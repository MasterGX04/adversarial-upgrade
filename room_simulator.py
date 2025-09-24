import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os, sys

def generate_rirs(
    output_dir="./rir_bank",
    num_rirs=20,
    fs=16000,
    room_dim_range=((4, 5, 2.5), (8, 9, 4)),
    mic_pos=(2, 1.5, 1.5),
    source_distance_range=(0.5, 3.0),
    max_order=15
):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_rirs):
        # Random room size
        room_dim = np.random.uniform(low=room_dim_range[0], high=room_dim_range[1])

        # Create shoebox room
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            absorption=0.4,
            max_order=max_order
        )

        # Mic at fixed or random location
        mic = np.array(mic_pos).reshape(3, 1)
        room.add_microphone_array(mic)

        # Source at random location/distance
        azimuth = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(*source_distance_range)
        src = mic[:, 0] + distance * np.array([np.cos(azimuth), np.sin(azimuth), 0])
        src = np.clip(src, 0.5, np.array(room_dim) - 0.5)
        room.add_source(src)

        room.compute_rir()

        rir = room.rir[0][0]
        rir = rir / np.abs(rir).max()

        rir_path = os.path.join(output_dir, f"rir_{i:03d}.wav")
        sf.write(rir_path, rir, samplerate=fs)
        print(f"✅ Saved RIR {i+1}/{num_rirs}: {rir_path}")

def main():
    # Check if a command-line argument is provided
    if len(sys.argv) > 1:
        try:
            num_rirs = int(sys.argv[1])
        except ValueError:
            print("Invalid number provided. Using default of 100 RIRs.")
            num_rirs = 100
        
    generate_rirs(num_rirs=num_rirs)

if __name__ == "__main__":
    main()