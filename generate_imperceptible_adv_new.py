import os
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import torchaudio
from generate_robust_adv_new import build_ctc_decoder, transcribe
import argparse
import librosa
import utils
import matplotlib.pyplot as plt
import numpy as np
import time

print(torch.cuda.is_available())         # Should be True if CUDA is installed
print(torch.version.cuda)                # Shows CUDA version your torch is built for

ctc_decoder = None 

def get_args():
    parser = argparse.ArgumentParser(
        description="Imperceptible adversarial attack generator (PyTorch + Wav2Vec2)"
    )

    parser.add_argument("-d", "--read_data", default="read_data.txt",
                        help="Path to read_data.txt (1st line = wav files, 2nd line = targets)")

    parser.add_argument("-i", "--num_iterations", type=int, default=300,
                        help="Number of optimisation steps per audio sample")

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
                        help="Learning rate for Stage-1 optimiser")
    
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Output directories for perturbed files")

    parser.add_argument("-a", "--alpha", type=float, default=1e-3,
                        help="Weight for masking-loss term in Stage-2")
    
    parser.add_argument("-e", "--epsilon", type=float, default=0.03,
                        help="Epsilon value for setting the range of the delta values")

    parser.add_argument("-t", "--target_phrase", action="store_true",
                        help="If set, perform a *targeted* attack toward this phrase")

    parser.add_argument("-ns", "--no_stage2", action="store_true",
                        help="Skip perceptual masking refinement (Stage 2)")

    parser.add_argument("-ut", "--use_temp", action="store_true", help="Use temporary stage 1 file to save time")
    return parser.parse_args()

def compute_silence_mask(clean_audio, frame_size=512, hop_length=256, energy_threshold=0.0005):
    # Compute short-time energy
    frames = clean_audio.unfold(1, frame_size, hop_length)  # (1, num_frames, frame_size)
    energy = (frames ** 2).mean(dim=2)                      # shape: (1, num_frames)

    # Create binary mask where energy > threshold
    speech_frames = (energy > energy_threshold).float()

    # Stretch back to original waveform length
    expanded = torch.repeat_interleave(speech_frames, hop_length, dim=1)

    # Pad if needed
    pad_length = clean_audio.shape[1] - expanded.shape[1]
    if pad_length > 0:
        expanded = F.pad(expanded, (0, pad_length), value=0)

    return expanded.squeeze(0)

class ImperceptibleAttacker:
    def __init__(self, args, device, model, labels):
        self.args = args
        self.device = device
        self.model = model
        self.labels = labels
        
        self.wav_paths, self.transcriptions, self.target_phrases = self._load_metadata(args.read_data)
        self.output_dir = self.args.output_dir
        self.current_name = None
        os.makedirs(self.output_dir, exist_ok=True)
        
    def text_to_indices(self, text):
        label_map = {ch: i for i, ch in enumerate(self.labels)}
        text = text.upper().replace(" ", "|")  # match model format
        return [label_map.get(c, 0) for c in text]
    
    def _load_metadata(self, path):
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        wavs = lines[0].split(",")
        transcripts = lines[1].split(",")
        targets = lines[2].split(",") if self.args.target_phrase else []
        return wavs, transcripts, targets
    
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
        
    def compute_masking_loss(self, clean, delta):
        if isinstance(clean, np.ndarray):
            clean = torch.from_numpy(clean)

        clean = clean.to(delta.device, dtype=torch.float32)
        if clean.ndim == 1:
            clean  = clean.unsqueeze(0)          # (1, T)
        if delta.ndim == 1:
            delta  = delta.unsqueeze(0)

        perturbed = clean + delta

        clean_stft = torch.stft(clean, n_fft=512, return_complex=True)
        perturbed_stft = torch.stft(perturbed, n_fft=512, return_complex=True)

        # magnitude (avoid log(0) with tiny epsilon)
        eps = 1e-7
        clean_mag     = clean_stft.abs()     + eps
        perturbed_mag = perturbed_stft.abs() + eps

        # Δ in dB:  20 log10( |P| / |C| )
        delta_db = 20.0 * torch.log10(perturbed_mag / clean_mag)

        # penalise bins that rise above the threshold (–40 dB)
        loss = F.relu(delta_db - (-40.0)).mean()
        return loss
    
    def attack_stage1(self, clean_audio, delta, target, target_phrase, silence_mask):
        delta.data *= silence_mask

        optimizer = torch.optim.Adam([delta], lr=self.args.learning_rate)
        target_flat = target.squeeze(0).to(self.device)       # (L,)  – remove the batch dim
        target_lengths = torch.tensor([target_flat.numel()],  # [L]
                                    dtype=torch.long, device=self.device)
        
        # Metrics for graphing learning metrics
        metrics = utils.MetricsLogger(keys=['iterations', 'snr', 'max_delta', 'mean_delta'])
        
        # start time to calculate speed
        start_time = time.time()
        match_count = 0
        for step in range(self.args.num_iterations):
            optimizer.zero_grad()
            perturbed = torch.clamp(clean_audio + delta, -1.0, 1.0)
            normed = (perturbed - perturbed.mean()) / (perturbed.std() + 1e-7)
            emissions = self.model(normed)[0]
            log_probs = F.log_softmax(emissions, dim=-1)

            input_length = torch.tensor([emissions.size(1)]).to(self.device)
            
            if not self.args.target_phrase:
                loss = -torch.mean(torch.sum(F.softmax(emissions, dim=-1) * log_probs, dim=-1))
                mask_loss = torch.tensor(0.0).to(self.device)
            else:
                loss = F.ctc_loss(
                    log_probs.transpose(0, 1),   # (T, B, C)
                    target_flat,                 # (L,)
                    input_length,               # (B,)
                    target_lengths,              # (B,)
                    blank=0,
                    reduction='mean',
                    zero_infinity=True,
                )
                mask_loss = torch.tensor(0.0).to(self.device)

            total_loss = loss + mask_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data *= silence_mask
                delta.data.clamp_(-self.args.epsilon, self.args.epsilon)

            if step % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[Stage 1 | Step {step}] CTC Loss: {loss.item():.4f}")
                print(f"10 iterations took {elapsed:.2f} seconds on device: {self.device}")
                max_delta = delta.abs().max().item()
                mean_delta = delta.abs().mean().item()
                snr = utils.compute_snr(clean_audio, delta)

                metrics.log(iterations=step, snr=snr, max_delta=max_delta, mean_delta=mean_delta)
                
                start_time = time.time()

            if step % 50 == 0:
                current_trans = transcribe(clean_audio + delta, self.model, self.labels, self.device)
                print(f"Transcription after {step} steps: {current_trans}")

                if current_trans.strip().lower() == target_phrase.strip().lower():
                    match_count += 1
                    if match_count >= 5:
                        print(f"[✓] Early stopping: matched target 5 times at step {step}")
                        utils.plot_stage1_metrics_imperceptible(metrics.get_metrics(),
                            f"{self.current_name}_imperceptible_stage1_progress.png")
                        metrics.reset()
                        return torch.clamp(clean_audio + delta, -1.0, 1.0), \
                            torch.nn.Parameter(delta.clone().detach().to(self.device))
                else:
                    match_count = 0  # Reset if it breaks the streak


        utils.plot_stage1_metrics_imperceptible(metrics.get_metrics(), f"{self.current_name}_attack_progress.png")
        metrics.reset()
        
        delta = delta.clone().detach().to(self.device).requires_grad_()
        return torch.clamp(clean_audio + delta, -1.0, 1.0), delta
    
    def attack_stage2(self, clean_audio, delta, target, target_phrase, silence_mask):
        """
        Perceptual-masking refinement (Stage 2).

        Args
        ----
        clean_audio  : (T,)  original waveform  @self.device
        delta        : (T,)  adversarial perturbation from Stage 1
        target_ids   : (1, L) OR (L,) tensor of token IDs              ← NEW
        silence_mask : (T,)  0/1 mask (don’t perturb silence)
        """
        alpha = self.args.alpha
        lr2 = self.args.learning_rate * 0.3
        optimizer = torch.optim.Adam([delta], lr=lr2)
        # Timer for how long code runs
        clock = 0
        
        target_flat = target.squeeze(0).to(self.device)   # (L,)
        target_lengths = torch.tensor([target_flat.numel()],
                                      dtype=torch.long,
                                      device=self.device)
        
        # Measuring best delta and snr
        best_delta = delta.clone().detach()
        best_snr = -float("inf")
        best_trans = ""

        metrics = utils.MetricsLogger(keys=['iterations', 'snr', 'ctc_loss', 'mask_loss'])
        for step in range(self.args.num_iterations // 2 + 1):
            start_time = time.time()
            optimizer.zero_grad()
            
            # Forward
            perturbed = torch.clamp(clean_audio + delta, -1.0, 1.0)
            normed = (perturbed - perturbed.mean()) / (perturbed.std() + 1e-7)

            emissions = self.model(normed)[0]
            log_probs = F.log_softmax(emissions, dim=-1)

            input_length = torch.tensor([emissions.size(1)]).to(self.device)
            
            loss_ctc = F.ctc_loss(
                log_probs.transpose(0, 1),   # (T, B, C)
                target_flat,                 # (L,)
                input_length,                   # (B,)
                target_lengths,              # (B,)
                blank=0,
                reduction='mean',
                zero_infinity=True,
            )
            
            loss_mask = self.compute_masking_loss(clean_audio, delta)
            
            total_loss = loss_ctc + alpha * loss_mask
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data *= silence_mask
                delta.data.clamp_(-self.args.epsilon, self.args.epsilon)

            # Logging
            clock += time.time() - start_time
            
            if step % 10 == 0:
                mean_delta = (delta).abs().mean().item()
                snr = utils.compute_snr(clean_audio, delta)
                print(f"[Stage 2 | Step {step}] Mask Loss: {loss_mask.item():.4f} | CTC: {loss_ctc.item():.4f} | Mean Δ: {mean_delta:.5f}\nMask: {loss_mask.item():.4f} | SNR: {snr:.2f} dB | Time for last 10 steps: {clock:.2f}s")
                metrics.log(iterations=step, snr=snr, ctc_loss=loss_ctc.item(), mask_loss=loss_mask.item())
                clock = 0
            
            if step % 100 == 0:
                current_adv = torch.clamp(clean_audio + delta, -1.0, 1.0)
                adv_transcription = transcribe(current_adv, self.model, self.labels, self.device)
                snr = utils.compute_snr(clean_audio, delta)
                
                if adv_transcription.strip().lower() == target_phrase.strip().lower():
                    if snr > best_snr:
                        print(f"[✓] New best SNR {snr:.2f} dB at step {step}")
                        best_delta = delta.clone().detach()
                        best_snr = snr
                        best_trans = adv_transcription
                print(f"Current transcription after {step} steps: {adv_transcription}")

        utils.plot_stage2_metrics_imperceptible(metrics.get_metrics(), f"{self.current_name}_stage2_attack.png")
        metrics.reset()
        
        print(f"Returning best transcription: {best_trans}")
        return torch.clamp(clean_audio + best_delta, -1.0, 1.0)
    
    # Visually determine if audio was properly masked/preturbed
    def visualize_perturbation(self, clean_np, adv_np, file_id, output_dir):
        delta_np = adv_np - clean_np

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        axs[0].plot(clean_np, color='blue')
        axs[0].set_title("Original Audio")

        axs[1].plot(adv_np, color='orange')
        axs[1].set_title("Adversarial Audio")

        axs[2].plot(delta_np, color='red')
        axs[2].set_title("Perturbation (Delta)")

        plt.xlabel("Sample Index")
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"{file_id}_visualization.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to: {save_path}")

    def run_all(self):
        for wav_path, transcript, target_phrase in zip(self.wav_paths, self.transcriptions, self.target_phrases):
            if not os.path.exists(wav_path):
                print(f"Skipping missing file: {wav_path}")
                continue

            # Load audio
            clean_np, _ = librosa.load(wav_path, sr=16000)
            clean_audio = torch.tensor(clean_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            silence_mask = compute_silence_mask(clean_audio)
            filename = os.path.splitext(os.path.basename(wav_path))[0]
            self.current_name = filename

            # Determine target
            target = None
            if self.args.target_phrase:
                tokens = self.text_to_indices(target_phrase.replace(" ", "|"))
                target = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                print(f"Target phrase: {target_phrase}\nTarget: {target}")
            else:
                print(f"\nUntargeted attack: {wav_path}")
            print(f"Clean transcription: {transcript}")

            # Initialize delta
            delta = torch.nn.Parameter(0.001 * torch.randn_like(clean_audio))

            temp_path = os.path.join(self.output_dir, "temp1_delta.wav")
            if self.args.use_temp and os.path.exists(temp_path):
                wav_np, _ = sf.read(temp_path)
                delta = torch.tensor(wav_np, dtype=torch.float32, device=self.device)
                delta = torch.nn.Parameter(delta.view(1, -1).clamp(-self.args.epsilon, self.args.epsilon))
                adv1 = clean_audio + delta
            else:
                adv1, new_delta = self.attack_stage1(clean_audio, delta, target, target_phrase, silence_mask)
                self.save_perturbation_audio(new_delta, temp_path)

            # Stage 2 refinement
            final_adv = adv1
            if self.args.target_phrase and not self.args.no_stage2:
                final_adv = self.attack_stage2(clean_audio, new_delta, target, target_phrase, silence_mask)

            # Evaluate and save
            adversarial_pred = transcribe(final_adv, self.model, self.labels, self.device)
            print(f"Final adversarial prediction: {adversarial_pred}")

            save_path = os.path.join(self.output_dir, f"{filename}_imperceptible.wav")
            self.save_perturbation_audio(final_adv, save_path)

            # Visualization
            self.visualize_perturbation(
                clean_audio.squeeze(0).cpu().numpy(),
                final_adv.squeeze(0).detach().cpu().numpy(),
                filename,
                self.output_dir
            )
            print(f"Saved to {save_path}")

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device being used: {device}")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    labels = bundle.get_labels()
    ctc_decoder = build_ctc_decoder(labels)
    attacker = ImperceptibleAttacker(args, device, model, labels)
    attacker.run_all()