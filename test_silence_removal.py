import unittest
import subprocess
import os
from pathlib import Path

class TestSilenceRemoval(unittest.TestCase):
    def setUp(self):
        self.cli_path = "C:/dev/Chatterblez/cli.py"
        self.epubs_path = "C:/dev/Chatterblez/test_epubs"
        self.output_path = "C:/dev/Chatterblez/test_output"
        Path(self.output_path).mkdir(exist_ok=True)

    def test_no_silence_removal(self):
        # Test that the CLI can create an audiobook without removing silence
        epub_path = os.path.join(self.epubs_path, "Journey-Through-Time.epub")
        output_file = Path(self.output_path) / "Journey-Through-Time.m4b"

        # Run the CLI to create the audiobook
        subprocess.run([".\.venv\Scripts\python.exe", self.cli_path, "-f", epub_path, "-o", self.output_path], check=True)

        # Check that the output file was created and has a reasonable size
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)

    def test_silence_removal(self):
        # Test that the CLI can remove silence from an existing M4B file
        epub_path = os.path.join(self.epubs_path, "the-digital-explorer.epub")
        original_m4b = Path(self.output_path) / "the-digital-explorer.m4b"
        silence_removed_m4b = Path(self.output_path) / "the-digital-explorer_no_silence.m4b"

        # First, create the audiobook
        subprocess.run([".\.venv\Scripts\python.exe", self.cli_path, "-f", epub_path, "-o", self.output_path], check=True)
        self.assertTrue(original_m4b.exists())
        from core import probe_duration
        original_duration = probe_duration(str(original_m4b))

        # Now, remove the silence
        subprocess.run([".\.venv\Scripts\python.exe", self.cli_path, "-rs", str(original_m4b), "-o", self.output_path], check=True)
        self.assertTrue(silence_removed_m4b.exists())
        new_duration = probe_duration(str(silence_removed_m4b))

        # Check that the new file is shorter than the original
        self.assertLess(new_duration, original_duration)

if __name__ == "__main__":
    unittest.main()
