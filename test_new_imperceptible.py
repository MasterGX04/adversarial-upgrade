import argparse
import os
import csv
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoConfig

# Load audio
def load_audio(path, sr=16000):
    waveform, _ = librosa.load(path, sr=sr)
    return torch.tensor(waveform).float().unsqueeze(0)  # [1, T]

# Transcribe audio using Wav2Vec2
def transcribe(model, processor, audio_tensor, device):
    input_values = processor(audio_tensor.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]  # this line is key!
    return transcription.strip()
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial examples using Wav2Vec2")
    parser.add_argument("-d", "--read_data", type=str, default="read_data.txt",
                        help="CSV-style text file: line 1 = filenames, line 2 = true or target transcriptions")
    parser.add_argument("-a", "--adv_stage", type=str, default="stage2", help="Suffix for adversarial file (e.g. '_stage2.wav')")
    parser.add_argument("--clean", action="store_true", help="Test clean audio instead of adversarial")
    parser.add_argument("-p", "--adv_dir", type=str, default="./adv_outputs", help="Directory where adversarial WAVs are stored")
    parser.add_argument("-o", "--output_csv", type=str, default="transcription_results.csv", help="CSV output path")
    args = parser.parse_args()
    
    cfg = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")
    print(cfg.architectures)

    # Load Wav2Vec2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    
    # Read read_data.txt
    with open(args.read_data, "r") as f:
        lines = f.readlines()
    wav_paths = [p.strip() for p in lines[0].split(",")]
    transcriptions = [t.strip().lower() for t in lines[2].split(",")] if not args.clean else [t.strip().lower() for t in lines[1].split(",")]
    assert len(wav_paths) == len(transcriptions), "Mismatch between paths and transcriptions"
    
    results = []
    correct = 0
    total = len(wav_paths)
    
    for original_path, expected in zip(wav_paths, transcriptions):
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        test_path = original_path if args.clean else os.path.join(args.adv_dir, f"{base_name}_adv.wav")

        if not os.path.exists(test_path):
            print(f"[!] Skipping missing file: {test_path}")
            continue

        audio = load_audio(test_path)
        predicted = transcribe(model, processor, audio, device).lower()
        results.append([test_path, expected, predicted])
        total += 1

        if predicted == expected:
            correct += 1
            
    # Save results
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "expected_transcription", "predicted_transcription"])
        writer.writerows(results)
    
    print(f"[✓] Saved results to {args.output_csv}")
    print(f"Accuracy: {correct}/{total} = {100 * correct / total:.2f}%")
    
if __name__ == "__main__":
    main()