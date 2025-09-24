import argparse, os, glob
import torch
import torchaudio
import soundfile as sf
from pyctcdecode import BeamSearchDecoderCTC, Alphabet
import librosa
import torch.nn.functional as F
import random, time
import csv
import numpy as np
from tqdm import tqdm
import utils
import difflib
import matplotlib.pyplot as plt

ctc_decoder = None 

# Loads audio
def load_audio(path):
    waveform, sr = librosa.load(path, sr=16000)
    return torch.tensor(waveform).float().unsqueeze(0)  # shape: [1, T]

# labels must include the CTC blank at index 0
# Example: labels = ['<pad>', 'a', 'b', ..., '|'] where '|' is used for space
def build_ctc_decoder(labels):
    labels = list(labels)
    # Convert '|' to space for pyctcdecode and build alphabet
    labels = [l if l != '|' else ' ' for l in labels]

    alphabet = Alphabet.build_alphabet(labels)

    # print("Alphabet contents:", alphabet.labels)
    return BeamSearchDecoderCTC(alphabet)

# Applying reverberations
def apply_rir_torch(audio_tensor, rir_tensor):
    # audio_tensor: [1, T]
    # rir_tensor: [1, 1, L]
    audio_tensor = audio_tensor.unsqueeze(1)  # [1, 1, T]
    return F.conv1d(audio_tensor, rir_tensor, padding=rir_tensor.shape[-1]-1).squeeze(1)

def transcribe(audio_tensor, model, labels, device):
    """
    Transcribe a waveform using Wav2Vec2 with CTC beam decoding for clean output.
    """
    global ctc_decoder
    if ctc_decoder is None:
        ctc_decoder = build_ctc_decoder(labels)

    model.eval()
    with torch.no_grad():
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(device)

        # Run model and get log probabilities
        emissions = model(audio_tensor)[0]  # shape: [1, T, C]
        emissions = F.log_softmax(emissions, dim=-1).cpu().squeeze(0).numpy()  # shape: [T, C]
        
        if emissions.shape[1] + 1 == len(ctc_decoder._alphabet.labels):
            # Add a fake column of -infinity logits (won't affect prediction)
            pad_column = np.full((emissions.shape[0], 1), -1e9)
            emissions = np.concatenate([emissions, pad_column], axis=1)
            
        # Decode using beam search
        decoded = ctc_decoder.decode(emissions)

        return decoded.strip().lower().replace("-", "").replace("  ", " ")   

def similarity(a: str, b: str) -> float:
    """
    Returns a similarity percentage between two strings.
    100.0  → exact match
     90.0  → 10 % characters differ / inserted / deleted
    """
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100.0
   
def get_args():
    parser = argparse.ArgumentParser(description="Robust Audio Adversarial Attack with Wav2Vec2")
    parser.add_argument("-d", "--read_data", type=str, default="read_data.txt", help="Path to read_data.txt")
    parser.add_argument("-i", "--num_iterations", type=int, default=1000, help="Number of optimization steps")
    parser.add_argument("-e", "--epsilon", type=float, default=0.03, help="Epsilon constraint")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate for attack")
    parser.add_argument("-r", "--use_reverb", action="store_true", help="Apply room reverberation")
    parser.add_argument("-o", "--output_dir", type=str, default="./adv_outputs", help="Where to save .wav outputs")
    parser.add_argument("-ut", "--use_temp", action="store_true", help="Use temporary stage 1 file to save time")
    return parser.parse_args()   
   
class PGDAttack:
    def __init__(self, model, labels, step_size=0.001, epsilon=0.02, num_steps=100, 
                 use_reverb=False, rir_files=None, save_dir="./adv_outputs", device='cpu', use_temp=False):
        self.model = model
        self.labels = labels
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.use_reverb = use_reverb
        self.rir_files = rir_files or []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = device
        self.current_name = None
        self.use_temp = use_temp

    def text_to_indices(self, text):
        label_map = {ch: i for i, ch in enumerate(self.labels)}
        text = text.upper().replace(" ", "|")  # match model format
        return [label_map.get(c, 0) for c in text]
    
    def _indices_to_text(self, indices):
        prev = -1
        result = []
        for idx in indices:
            if idx != prev and idx != 0:  # Skip blank token (index 0)
                result.append(self.labels[idx])
            prev = idx
        return "".join(result).replace("|", " ")
    
    def _apply_rir_and_crop(self, wav, rir, L):
        """
        wav : (1, 1, L)  clean + δ  in range [-1, 1]
        rir : (1, 1, K)  raw room-impulse response
        L   : original signal length
        -------------------------------------------------
        Returns conv(wav, rir_trimmed) cropped to length L
        """
        #1. trim leading zeros / delay so direct path is at t = 0 ----
        #     use the first sample ≥ 1 % of the peak as the onset
        onset = (rir.abs() >= 0.01 * rir.abs().max()).nonzero(as_tuple=False)[0, -1]
        rir_trim = rir[..., onset:]                    # shape (1,1,K_on)
        
        conv = torch.nn.functional.conv1d(
            wav, rir_trim, padding=rir_trim.size(-1) - 1
        )

        return conv[..., :L]   
    
    def save_perturbation_audio(self, delta, path, sr=16000):
        perturbation = delta.detach().cpu().numpy().squeeze()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Remove previous file
        if path and os.path.exists(path):
            os.remove(path)

        save_path = utils.get_unique_filename(path)
        # Create new path
        sf.write(save_path, perturbation, sr)
        print(f"Saved perturbation to: {path}")

    def attack(self, clean_audio, target_phrase):
        """
        Two-Stage Robust Adversarial Attack.
        """
        device = next(self.model.parameters()).device
        self.device = device
        clean_audio = clean_audio.to(device)
        delta = torch.nn.Parameter(0.001 * torch.randn_like(clean_audio).to(device))

        # Silence mask to avoid perturbing silent regions
        silence_mask = (clean_audio.abs() > 0.01).float().to(device)
        masked = (silence_mask == 0).sum().item()
        total = silence_mask.numel()
        print(f"Masked samples: {masked}/{total}")

        if target_phrase is not None:
            # Targeted attack: tokenize target
            tokens = self.text_to_indices(target_phrase.replace(" ", "|"))
            target = torch.tensor(tokens).long().unsqueeze(0).to(device)
            print(f"Target phrase: {target_phrase}\nTarget: {target}")
            wav_path = os.path.join(self.save_dir, "stage1_temp.wav")
            if self.use_temp:
                wav_np, _ = sf.read(wav_path)
                wav = torch.tensor(wav_np, dtype=torch.float32, device=self.device)
                delta = wav.view(1, -1).clamp(-self.epsilon, self.epsilon)
                delta = torch.nn.Parameter(delta)
            else:
                delta = self._stage1_attack(clean_audio, delta, target, silence_mask)
                delta = torch.nn.Parameter(delta.clone().detach().to(self.device))
                
            self.save_perturbation_audio(delta, wav_path)
            stage2_delta = self._stage2_refinement(clean_audio, delta, target, target_phrase, silence_mask)
        else:
            print(f"Error: target phrase is None. Please try again with a target phrase")
            return []
        
        adv_audio = (clean_audio + stage2_delta).detach().cpu()
        return adv_audio

    def random_rir(self):
        rir_path = random.choice(self.rir_files)
        rir_np, _ = librosa.load(rir_path, sr=16000)
        rir_tensor = torch.tensor(rir_np / np.sqrt(np.sum(rir_np ** 2)),
            dtype=torch.float32).view(1, 1, -1).to(self.device)
        
        return rir_tensor

    def _stage1_attack(self, clean_audio, delta, target, silence_mask):
        self.model.eval()
        # Metrics for sample graphs
        metrics = utils.MetricsLogger(keys=['iterations', 'ctc_loss', 'snr', 'max_delta', 'mean_delta'])
        optimizer = torch.optim.Adam([delta], lr=self.step_size)
        clock = 0
        prev_path = os.path.join(self.save_dir, "temp_perturb.wav")
        
        for step in range(1, self.num_steps + 1):
            start_time = time.time()
            
            # Add perturbation
            perturbed = torch.clamp(clean_audio + delta, -1.0, 1.0)

            if self.use_reverb and self.rir_files:
                rir_tensor = self.random_rir()
                perturbed = apply_rir_torch(perturbed, rir_tensor)
                
                # Trim or pad to fixed length (match clean_audio length)
                if perturbed.size(1) > clean_audio.size(1):
                    perturbed = perturbed[:, :clean_audio.size(1)]
                elif perturbed.size(1) < clean_audio.size(1):
                    pad = clean_audio.size(1) - perturbed.size(1)
                    perturbed = F.pad(perturbed, (0, pad), mode='constant', value=0)

            emissions = self.model(perturbed)[0]
            input_lengths = torch.tensor([emissions.size(1)]).to(self.device)
            target_lengths = torch.tensor([len(target.squeeze())]).to(self.device)
            
            # CTC loss for Stage 1
            log_probs = F.log_softmax(emissions, dim=-1)
            loss = F.ctc_loss(
                log_probs.transpose(0, 1),
                target,
                input_lengths,
                target_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=True,
            )
            
            # Update delta via FGSM-style step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            
            with torch.no_grad():
                delta.mul_(silence_mask).clamp_(-self.epsilon, self.epsilon)

            # Logging
            clock += time.time() - start_time
            
            if step % 10 == 0:
                max_delta = delta.abs().max().item()
                mean_delta = (delta).abs().mean().item()
                snr = 10 * torch.log10(clean_audio.pow(2).mean() / (delta.pow(2).mean() + 1e-8)).item()
                print(f"[Step {step}] Loss: {loss.item():.4f} | Max Δ: {max_delta:.5f} | Mean Δ: {mean_delta:.5f} | SNR: {snr:.2f} dB | Time for last 10 steps: {clock:.2f}s")
                metrics.log(iterations=step, ctc_loss=loss.item(), snr=snr,
                    max_delta=max_delta, mean_delta=mean_delta)
                clock = 0
                
            if step % 50 == 0:
                self.save_perturbation_audio(delta, prev_path)
                current_trans = transcribe(perturbed, self.model, self.labels, self.device)
                print(f"Current transcription: {current_trans}")
        
        utils.plot_robust_stage1_metrics(metrics.get_metrics(), f"{self.current_name}_robust_stage1_visualization.png")
        metrics.reset()
        return delta.detach()

    def _stage2_refinement(self, clean_audio, delta, target, target_phrase, silence_mask):
        metrics = utils.MetricsLogger(keys=[
            'iterations', 'total_loss', 'clean_ctc', 'reverb_ctc',
            'clean_sim', 'reverb_sim', 'snr'
        ])
        
        lr  = self.step_size         # 30 % of Stage-1 LR
        optimizer = torch.optim.Adam([delta], lr=lr, betas=(0.9, 0.98))

        L = clean_audio.size(-1)
        for step in range(1, self.num_steps // 2 + 1):
            t0 = time.time()

            views = [torch.clamp(clean_audio + delta, -1, 1)]  
            if self.use_reverb and self.rir_files:
                rir = torch.tensor(self.random_rir(), device=self.device).view(1, 1, -1)
                rev = self._apply_rir_and_crop(views[0], rir, L)
                views.append(rev)

            losses = []
            for v in views:
                logits = self.model(v.squeeze(1))[0]    
                lp = F.log_softmax(logits, -1)
                ilen = torch.tensor([logits.size(1)], device=self.device)
                tlen = torch.tensor([len(target.squeeze())], device=self.device)
                losses.append(F.ctc_loss(lp.transpose(0,1), target, ilen, tlen,
                                        blank=0, reduction='mean', zero_infinity=True))

            total_loss = torch.stack(losses).mean()   # equal weight
            
            # Update delta via FGSM-style step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                delta.clamp_(-self.epsilon, self.epsilon)
                delta.mul_(silence_mask)    
                
            # progress prints
            if step % 10 == 0:
                print(f"[Stage-2 | {step:4d}] loss={total_loss:.4f}  "
                  f"LR={lr:.2e}  time/10={time.time()-t0:.2f}s")
                
                clean_ctc = losses[0].item()
                reverb_ctc = losses[1].item() if len(losses) > 1 else clean_ctc
                clean_pred = transcribe(views[0], self.model, self.labels, self.device)
                clean_sim = similarity(clean_pred, target_phrase)
                reverb_sim = 0
                if len(views) > 1:
                    reverb_pred = transcribe(views[1], self.model, self.labels, self.device)
                    reverb_sim = similarity(reverb_pred, target_phrase)
                snr = 10 * torch.log10(clean_audio.pow(2).mean() / (delta.pow(2).mean() + 1e-8)).item()

                metrics.log(
                    iterations=step,
                    total_loss=total_loss.item(),
                    clean_ctc=clean_ctc,
                    reverb_ctc=reverb_ctc,
                    clean_sim=clean_sim,
                    reverb_sim=reverb_sim,
                    snr=snr
                )

            # *optional* early-stop check
            if step % 50 == 0:
                pred = transcribe(views[0], self.model, self.labels, self.device)
                similar = similarity(pred, target_phrase)
                print(f"Prediction: {pred}\nSimilarity: {round(similar, 2)}")
                
                if pred.strip().lower() == target_phrase.strip().lower():
                    if not self.use_reverb:
                        return delta.detach()
                    
                    pred_r = transcribe(views[1], self.model, self.labels, self.device)
                    reverb_sim = similarity(pred_r, target_phrase)
                    print(f"Prediction with reverb: {pred_r}\nReverb similarity: {round(reverb_sim, 2)}")
                    if pred_r.strip().lower() == target_phrase.strip().lower():
                        print("Clean & reverb match target — done.")
                        utils.plot_robust_stage2_metrics(metrics.get_metrics(), self.current_name)
                        metrics.reset()
                        return delta.detach()

        utils.plot_robust_stage2_metrics(metrics.get_metrics(), self.current_name)
        metrics.reset()
        print("Stage 2 finished; returning best candidate.")
        return delta.detach()
    
    def save(self, adv_audio, filename, sample_rate=16000):
        path = os.path.join(self.save_dir, filename)
        audio_np = adv_audio.squeeze().numpy()
        sf.write(path, audio_np, samplerate=sample_rate)
        print(f"[✓] Saved: {path}")
        
    def run_all(self, read_data_path):
        with open(read_data_path, "r") as f:
            lines = f.readlines()

        assert len(lines) >= 3, "Need 3 rows in read_data.txt: wav paths, transcriptions, target phrases"
        wav_paths = [p.strip() for p in lines[0].split(",")]
        targets = [t.strip().lower() for t in lines[1].split(",")]
        target_phrases = [t.strip().lower() for t in lines[2].split(",")]

        if not (len(wav_paths) == len(targets) == len(target_phrases)):
            print(f"Length wav: {len(wav_paths)}, Length targets: {len(targets)}, Length TP: {len(target_phrases)}")
            raise AssertionError("Mismatch in lengths of wav_paths, targets, and target_phrases.")

        log_path = os.path.join(self.save_dir, "attack_results.csv")
        fieldnames = [
            "file", "clean_pred", "adv_pred", "target", "clean_matches",
            "adv_matches_target", "max_delta", "snr", "robust_success_rate"
        ]

        for path, transcription, target_phrase in zip(wav_paths, targets, target_phrases):
            audio = load_audio(path)
            filename = os.path.splitext(os.path.basename(path))[0]
            wav_out = f"{filename}_robust_adv.wav"
            self.current_name = filename
            print(f"Current file name: {self.current_name}")
            print(f"Attacking file: {path} with target: {target_phrase}")

            adv_audio = self.attack(audio, target_phrase)
            self.save(adv_audio, wav_out)

            clean_pred = transcribe(audio, self.model, self.labels, self.device)
            adv_pred = transcribe(adv_audio, self.model, self.labels, self.device)
            print(f"Clean prediction: {clean_pred}")
            print(f"Adversarial prediction: {adv_pred}")

            delta = (adv_audio - audio).squeeze()
            max_delta = delta.abs().max().item()
            snr = 10 * torch.log10(audio.pow(2).mean() / (delta.pow(2).mean() + 1e-8)).item()

            clean_success = int(clean_pred.strip() == transcription.strip())
            adv_success = int(adv_pred.strip() == transcription.strip())
            target_hit = int(adv_pred.strip() == target_phrase.strip()) if target_phrase else adv_success

            if self.use_reverb and self.rir_files:
                sampled_rirs = random.sample(self.rir_files, min(100, len(self.rir_files)))
                sim_scores = []

                for rir_path in tqdm(sampled_rirs, desc="Evaluating RIRs", leave=False):
                    rir_np, _ = librosa.load(rir_path, sr=16000)
                    rir_np = rir_np / np.sqrt(np.sum(rir_np ** 2))
                    rir_tensor = torch.tensor(rir_np, dtype=torch.float32).view(1, 1, -1).to(adv_audio.device)

                    rev = apply_rir_torch(adv_audio, rir_tensor)[..., :adv_audio.size(-1)]
                    pred = transcribe(rev, self.model, self.labels, self.device)
                    gold = target_phrase or transcription
                    sim_scores.append(similarity(pred.strip(), gold.strip()))

                visualize_rir_scores(sim_scores, filename)
                robust_success_rate = round(float(np.mean(sim_scores)), 2)
            else:
                robust_success_rate = "N/A"

            row = {
                "file": wav_out,
                "clean_pred": clean_pred,
                "adv_pred": adv_pred,
                "target": target_phrase,
                "clean_matches": clean_success,
                "adv_matches_target": target_hit,
                "max_delta": round(max_delta, 6),
                "snr": round(snr, 2),
                "robust_success_rate": robust_success_rate
            }
            log_results(log_path, row, fieldnames)
 
def log_results(csv_path, row_dict, fieldnames):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def visualize_rir_scores(sim_scores, filename, output_dir="./graphs"):
    """
    Visualize similarity scores across RIRs for a given adversarial example.

    Args:
        sim_scores (list of float): Similarity scores (0–100).
        filename (str): Name of the audio file.
        output_dir (str): Where to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sim_scores)), sim_scores, color='skyblue')
    plt.axhline(y=sum(sim_scores)/len(sim_scores), color='red', linestyle='--', label='Mean Similarity')
    plt.xlabel("RIR Sample Index")
    plt.ylabel("Similarity (%)")
    plt.title(f"RIR Robustness for {filename}")
    plt.legend()
    save_path = os.path.join(output_dir, f"{filename}_rir_scores.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"RIR visualization saved to {save_path}")

    
        
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    # Load Wav2Vec2 model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    labels = bundle.get_labels()

    # Load RIRs if needed
    rir_files = glob.glob("./rir_bank/*.wav") if args.use_reverb else []
    if args.use_reverb and not rir_files:
        raise FileNotFoundError("No RIR files found in ./rir_bank")

    # Run the attack loop
    attacker = PGDAttack(
        model=model,
        labels=labels,
        step_size=args.learning_rate,
        epsilon=args.epsilon,
        num_steps=args.num_iterations,
        use_reverb=args.use_reverb,
        rir_files=rir_files,
        save_dir=args.output_dir,
        device=device,
        use_temp=args.use_temp,
    )
    attacker.run_all(args.read_data)