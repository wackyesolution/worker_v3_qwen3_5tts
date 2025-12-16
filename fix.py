import os
import subprocess
from pathlib import Path
import glob


def fix_corrupted_wav_files(directory):
    """
    Fix all corrupted WAV files in a directory by re-encoding them to proper PCM WAV format.
    """
    # Find all .wav files
    wav_files = glob.glob(os.path.join(directory, "*.wav"))

    print(f"Found {len(wav_files)} WAV files to check/fix")

    for wav_file in wav_files:
        try:
            # Create temporary output file
            temp_file = wav_file + ".temp.wav"

            print(f"\nProcessing: {os.path.basename(wav_file)}")

            # Re-encode to proper PCM WAV format
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-i', wav_file,
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                '-ar', '24000',  # Match your sample rate
                '-ac', '1',  # Mono
                temp_file
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Replace original with fixed file
                os.remove(wav_file)
                os.rename(temp_file, wav_file)
                print(f"✓ Fixed: {os.path.basename(wav_file)}")
            else:
                print(f"✗ Failed to fix: {os.path.basename(wav_file)}")
                print(f"Error: {result.stderr[:200]}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except Exception as e:
            print(f"✗ Error processing {wav_file}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    print("\n✓ Batch fixing complete!")


# Usage
directory = r"C:\ebooks"
fix_corrupted_wav_files(directory)