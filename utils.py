import torch
import os
import matplotlib.pyplot as plt

def compute_snr(clean, delta):
    return 10 * torch.log10(clean.pow(2).mean() / (delta.pow(2).mean() + 1e-8)).item()

def get_unique_filename(path):
    """
    Returns a unique file path by appending (1), (2), ... if file already exists.
    Works like Windows file renaming behavior.
    """
    base, ext = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = f"{base} ({counter}){ext}"
        counter += 1
    return path

class MetricsLogger:
    """
    A simple class to log and store metrics during attack iterations.
    """
    def __init__(self, keys=None):
        # Default keys if none are provided
        if keys is None:
            keys = ['iterations', 'snr', 'ctc_loss', 'mask_loss', 'max_delta', 'mean_delta']
        self.data = {k: [] for k in keys}

    def log(self, **kwargs):
        """
        Log values to the metrics dictionary.
        Example: logger.log(iteration=10, snr=12.5, max_delta=0.05)
        """
        for k, v in kwargs.items():
            if k in self.data:
                self.data[k].append(v)
            else:
                raise KeyError(f"Metric '{k}' not initialized in MetricsLogger.")

    def get_metrics(self):
        return self.data

    def reset(self):
        for key in self.data:
            self.data[key] = []


def plot_stage1_metrics_imperceptible(metrics, file_name="attack_progress.png"):
    """
    Plot SNR, max_delta, and mean_delta over iterations.

    Args:
        metrics (dict): Dictionary with keys:
            - 'iterations': list of iteration indices
            - 'snr': list of SNR values (in dB)
            - 'max_delta': list of max delta values
            - 'mean_delta': list of mean delta values
        output_path (str): Path to save the resulting plot.
    """
    os.makedirs("./graphs", exist_ok=True)
    iters = metrics['iterations']
    snr = metrics['snr']
    max_d = metrics['max_delta']
    mean_d = metrics['mean_delta']

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot SNR on the left axis
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("SNR (dB)", color='tab:blue')
    ax1.plot(iters, snr, label="SNR (dB)", color='tab:blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)

    # Plot max_delta and mean_delta on the right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Delta Values", color='tab:red')
    ax2.plot(iters, max_d, label="Max Δ", color='tab:red', linestyle='--')
    ax2.plot(iters, mean_d, label="Mean Δ", color='tab:orange', linestyle='-.')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Attack Progress: SNR, Max Δ, Mean Δ")
    plt.tight_layout()
    save_path = get_unique_filename(os.path.join("graphs", file_name))
    plt.savefig(save_path)
    plt.close()
    print(f"Attack progress plot saved to: {save_path}")

def plot_stage2_metrics_imperceptible(metrics, file_name="stage2_progress.png"):
    """
    Plot SNR, CTC Loss, and Masking Loss during Stage 2.
    Args:
        metrics (dict): {'iterations': [], 'snr': [], 'ctc_loss': [], 'mask_loss': []}
        output_path (str): Path to save the plot.
    """
    os.makedirs("./graphs", exist_ok=True)
    steps = metrics['iterations']
    plt.figure(figsize=(12, 6))

    # Plot SNR
    plt.plot(steps, metrics['snr'], label="SNR (dB)", color='blue', linewidth=2)

    # Plot loss values (CTC and Masking)
    plt.plot(steps, metrics['ctc_loss'], label="CTC Loss", color='red', linestyle='--')
    plt.plot(steps, metrics['mask_loss'], label="Masking Loss", color='green', linestyle='-.')

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Stage 2 Attack Progress (SNR and Losses)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_path = get_unique_filename(os.path.join("graphs", file_name))
    plt.savefig(save_path)
    plt.close()
    print(f"Stage 2 metrics plot saved to {save_path}")
    
def plot_robust_stage1_metrics(metrics, file_name="robust_stage1_progress.png"):
    os.makedirs("./graphs", exist_ok=True)
    steps = metrics['iterations']
    _, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(steps, metrics['ctc_loss'], color='tab:green', linewidth=2)
    axes[0].set_ylabel("CTC Loss", color='tab:green')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title("Stage 1 Robust Attack Metrics")

    axes[1].plot(steps, metrics['snr'], color='tab:blue', linewidth=2)
    axes[1].set_ylabel("SNR (dB)", color='tab:blue')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(steps, metrics['mean_delta'], color='tab:orange', linewidth=2)
    axes[2].set_ylabel("Mean Δ", color='tab:orange')
    axes[2].set_xlabel("Iteration")
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = get_unique_filename(os.path.join("graphs", file_name))
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Stage 1 robust attack metrics plot saved to {save_path}")

def plot_robust_stage2_metrics(metrics, output_prefix):
    steps = metrics['iterations']
    os.makedirs("./graphs", exist_ok=True)

    # CTC Loss
    plt.figure(figsize=(10, 5))
    plt.plot(steps, metrics['clean_ctc'], label="Clean View CTC", linestyle='--')
    plt.plot(steps, metrics['reverb_ctc'], label="Reverb View CTC", linestyle='-.')
    plt.title("Stage 2: CTC Loss (Clean vs Reverb)")
    plt.xlabel("Iteration")
    plt.ylabel("CTC Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = get_unique_filename(os.path.join("graphs", f"{output_prefix}_ctc_loss.png"))
    plt.savefig(save_path)
    plt.close()

    # SNR
    if 'snr' in metrics and metrics['snr']:
        plt.figure(figsize=(10, 5))
        plt.plot(steps, metrics['snr'], label="SNR (dB)")
        plt.title("Stage 2: SNR Progress")
        plt.xlabel("Iteration")
        plt.ylabel("SNR (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = get_unique_filename(os.path.join("graphs", f"{output_prefix}_snr.png"))
        plt.savefig(save_path)
        plt.close()

    # Similarity
    plt.figure(figsize=(10, 5))
    plt.plot(steps, metrics['clean_sim'], label="Clean Similarity")
    plt.plot(steps, metrics['reverb_sim'], label="Reverb Similarity", linestyle='-.')
    plt.title("Stage 2: Clean vs Reverb Similarity")
    plt.xlabel("Iteration")
    plt.ylabel("Similarity (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = get_unique_filename(os.path.join("graphs", f"{output_prefix}_similarity.png"))
    plt.savefig(save_path)
    plt.close()
    