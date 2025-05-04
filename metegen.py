import os
from pathlib import Path

def generate_protocol_file(audio_dir, output_file, is_train=True):
    """
    Generate a protocol file for an anti-spoofing model using audio files.

    Args:
        audio_dir (str or Path): Directory containing 'real' and 'fake' subdirectories with .wav files.
        output_file (str or Path): Path to save the protocol file.
        is_train (bool): Whether this is for training data (adds labels and tags).
    """
    audio_dir = Path(audio_dir)
    output_file = Path(output_file)

    # Ensure audio directory exists
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # Prepare output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect and sort audio files
    real_files = sorted((audio_dir / 'real').glob('*.wav'))
    fake_files = sorted((audio_dir / 'fake').glob('*.wav'))

    with output_file.open('w') as f:
        # Process real audio files
        for i, file_path in enumerate(real_files):
            file_name = file_path.stem  # Get filename without extension
            speaker_id = f"SPEAKER_{i:03d}"
            if is_train:
                f.write(f"{speaker_id} {file_name} bonafide 1\n")
            else:
                f.write(f"{speaker_id} {file_name} - -\n")

        # Process fake audio files
        for i, file_path in enumerate(fake_files):
            file_name = file_path.stem
            speaker_id = f"SPEAKER_{i + len(real_files):03d}"
            if is_train:
                f.write(f"{speaker_id} {file_name} spoof 0\n")
            else:
                f.write(f"{speaker_id} {file_name} - -\n")


# === Usage example ===
# Update 'your_dataset_path' below with the actual dataset path on your system.

dataset_base = Path('/home/pm_students/SSL_Anti-spoofing/archive/for-norm/for-norm')

# Generate training protocol
generate_protocol_file(
    dataset_base / 'ASVspoof2019_LA_train',
    dataset_base / 'ASVspoof_LA_cm_protocols' / 'ASVspoof2019.LA.cm.train.trn.txt',
    is_train=True
)

# Generate development protocol
generate_protocol_file(
    dataset_base / 'ASVspoof2019_LA_dev',
    dataset_base / 'ASVspoof_LA_cm_protocols' / 'ASVspoof2019.LA.cm.dev.trl.txt',
    is_train=False
)

# Generate evaluation protocol
generate_protocol_file(
    dataset_base / 'ASVspoof2021_LA_eval',
    dataset_base / 'ASVspoof_LA_cm_protocols' / 'ASVspoof2021.LA.cm.eval.trl.txt',
    is_train=False
)

