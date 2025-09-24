# adversarial-upgrade
Adversarial Attack upgrade from 2019 Tensorflow to 2025 PyTorch

Migration: TensorFlow ➜ PyTorch (Wav2Vec2 + modern tooling)
This repo was upgraded from TensorFlow/Lingvo to PyTorch + Hugging Face Wav2Vec2 for simpler deps, faster iteration, and easier decoding.

What changed (at a glance)

- ASR backend: Lingvo (TF) → Wav2Vec2-base-960h (PyTorch) for 16 kHz speech.
- Decoder: Greedy/TF → CTC beam search via pyctcdecode (optional KenLM support). 
- Attacks:
    - Imperceptible attack: re-implemented with PyTorch autograd + psychoacoustic masking.
    - Robust (EoT) attack: PGD-style optimization with optional room RIR augmentation (PyRoomAcoustics).
- Room simulation: kept PyRoomAcoustics; added quality-of-life helpers for trimming/cropping RIR-convolved audio.
- Files (old → new):
- generate_imperceptible_adv.py (TF) → generate_imperceptible_adv_new.py (PyTorch).
- generate_robust_adv.py (TF) → generate_robust_adv_new.py (PyTorch).
- room_simulator.py retained (RIR generation). (See “RIR tools” below.)
- Heads-up: Wav2Vec2-base-960h expects 16 kHz input; the pipeline normalizes/clamps waveforms accordingly.

Why the change
- Lighter stack & maintenance: swap heavy TF/Lingvo graph plumbing for lean PyTorch eager mode.
- Better decoding: plug-and-play CTC beam search + LM with pyctcdecode.
- **Repro-friendly:** Hugging Face model cards & torchaudio ecosystem make it easy to reproduce transcripts and metrics.

- # 1) Create env & install
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# If you have CUDA, match the torch wheel to your CUDA version.

# 2) (Optional) Generate Room Impulse Responses (RIRs) for robust attacks
python room_simulator.py        # writes RIR .wav files (16 kHz) to ./rir_bank
# (You can adjust counts/paths inside the script.)

If in Visual Studio Code, use ' instead of \
# 3) Imperceptible attack (targeted example)
python generate_imperceptible_adv_new.py \
  -d read_data.txt \
  -i 300 \
  -lr 1e-3 \
  -o ./adv_outputs \
  -a 1e-3 \
  -e 0.03 \
  -t

# 4) Robust (EoT) attack with reverb augmentation
python generate_robust_adv_new.py \
  -d read_data.txt \
  -i 1000 \
  -lr 1e-3 \
  -e 0.05 \
  -r \
  -o ./adv_outputs
