# Chatterblez 🗣️📖✨

## 🚀 Transform Your PDFs & EPUBS into Engaging Audiobooks! 🎧

Ever wished your favorite books could talk to you? 🤩 Chatterblez is here to make that dream a reality! 🪄 We leverage the cutting-edge **Next-gen AI Chatterbox-tts from Resemble-AI** ([check them out!](https://github.com/resemble-ai/chatterbox)) to generate high-quality audiobooks directly from your PDF or EPUB files. 📚➡️🔊

Inspired by the awesome work of [audiblez](https://github.com/santinic/audiblez), Chatterblez takes text-to-speech to the next level, offering a seamless and delightful listening experience. 💖

---

### 💻 Compatibility 🧑‍💻

Tested and running smoothly on:

* Windows 11 🪟
* Python 3.12 🐍
* **NVIDIA CUDA 12.4:** Required for GPU acceleration and optimal performance. Please ensure you have a compatible NVIDIA graphics card and the necessary CUDA drivers installed. 🚀

---

### 🛠️ Installation & Setup 🚀

Ready to dive in? Here's how to get Chatterblez up and running on your machine:

#### 1. Clone the Repository 📥

```bash
git clone https://github.com/cpttripzz/Chatterblez
```

#### 2. Install CUDA (NVIDIA Graphics Cards Only!) ⚡️

If you have an NVIDIA GPU, install CUDA for optimal performance. This significantly speeds up the AI processing!

* Download CUDA 12.4:
  [https://developer.nvidia.com/cuda-12-4-0-download-archive?target\_os=Windows\&target\_arch=x86\_64\&target\_version=11\&target\_type=exe\_local](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
* Follow the installation instructions provided by NVIDIA. 🧑‍💻

#### 3. Install Python Dependencies 📦

Navigate into the cloned directory and install the required Python packages:

```bash
pyenv install 3.11.9
pyenv local 3.11.9

# Create a new one with Python 3.11
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Now, install the requirements
pip install --upgrade setuptools wheel

pip install llvmlite==0.41.1 numba==0.58.1 numpy==1.25.2


# Reinstall PyQt6 (this will automatically install the correct sip version)
pip install PyQt6
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# Torch/torchaudio are installed separately to match your CUDA build, so they are no longer pinned in `requirements.txt`.
pip install -r requirements.txt

# Use the upstream repo instead of the PyPI release to avoid the stricter (torch==2.6.0) pin.
pip install git+https://github.com/resemble-ai/chatterbox.git

```

#### 👉 Prefer a one-shot install?

```bash
./install.sh
```

The script bootstraps a `.venv`, installs FFmpeg on apt-based systems (`apt update && apt upgrade -y && apt install -y ffmpeg && ffmpeg -version`), pulls in the CUDA-enabled `torch`/`torchaudio` wheels (default **2.4.0/cu124**), and finishes by grabbing the latest `chatterbox` straight from GitHub. You can tweak it via environment variables:

* `PYTHON_BIN=python3.11 ./install.sh` – choose a specific interpreter (project is tested on **Python 3.11.9**).
* `SKIP_VENV=1 ./install.sh` – install into the currently active environment (useful on managed images like Runpod).
* `TORCH_VERSION=2.4.1 TORCHAUDIO_VERSION=2.4.1 ./install.sh` – override the default CUDA wheel versions if needed.
* `SKIP_FFMPEG=1 ./install.sh` – skip the apt-based FFmpeg installation if it’s already present or you’re on a non-Debian system.

**DANIEL AGGIUNTE:** Se usi Runpod con il template “PyTorch Environment – Ready-to-use PyTorch + Python development environment with JupyterLab...”, clona il tuo fork e lancia direttamente lo script (puoi tenere tutto nell’ambiente globale del pod):

```bash
git clone <your fork>
cd Chatterblez
SKIP_VENV=1 ./install.sh   # ometti SKIP_VENV per far creare la .venv
```

Lo script esegue automaticamente `apt update && apt upgrade -y && apt install -y ffmpeg` (senza sudo) e stampa `ffmpeg -version`, così l’audio è pronto subito anche sul template Runpod.

Questo setup assume **Python 3.11.9** e i binari PyTorch/cu124 **2.4.0** del template.

#### 🕹️ Avvio guidato (Runpod friendly)

Usa il nuovo launcher interattivo per convertire rapidamente i libri dalla cartella `DD_book` o per fare un test vocalico di ~30 secondi con il tuo timbro:

```bash
python dd_launcher.py
```

Il menu (ora impostato di default sul modello multilingue con lingua **it**) ti chiederà:

1. Quale PDF/EPUB di `DD_book/` vuoi leggere.
2. Quale timbro WAV di `DD_timbro/` vuoi usare (c’è anche l’opzione `[Voce predefinita – nessun timbro]`).
3. Quale profilo voce applicare:
   * **Rilassato / Emotivo** – più lento (speed 0.88), `cfg_weight` 0.32 e `exaggeration` 0.72 come suggerito dalla guida originale.
   * **Bilanciato** – i valori “stock” (speed 1.0, `exaggeration=0.5`, `cfg_weight=0.5`).
   * **Energetico** – ritmo e tono più vivaci.
4. `Converti libro` ➜ avvia direttamente `core.main` con l’output in `DD_Output/`.
5. `Test timbro` ➜ genera un file `voice_test_*.wav` (circa 30s) usando `ChatterboxMultilingualTTS` in italiano per capire se procedere prima dell’intero audiobook.

Entrambe le opzioni utilizzano il modello `ChatterboxMultilingualTTS` con `language_id="it"` e applicano il profilo selezionato; se non scegli un WAV rimane la voce predefinita.

This might take a moment, so grab a coffee! ☕

#### 4. Install FFMPEG 🔊

FFmpeg is required for audio processing. Here's how to install it:

**🔵 Windows:**

1. Download a static build from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2. Extract the `.zip` to a location like `C:\ffmpeg`
3. Add the `C:\ffmpeg\bin` path to your **System Environment Variables**:

   * Search *"Edit the system environment variables"* from the Start Menu
   * Click "Environment Variables..."
   * Under "System Variables", find `Path`, click **Edit...**, then **New**, and paste the `bin` folder path
4. Open a new Command Prompt and run:

   ```bash
   ffmpeg -version
   ```

   You should see FFmpeg version info.

**🟢 Linux (Ubuntu):**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install ffmpeg -y
ffmpeg -version
```

**🟣 macOS (with Homebrew):**

```bash
brew install ffmpeg
ffmpeg -version
```

---

### 🚀 Usage (Coming Soon!)

Detailed usage instructions, including how to convert your first PDF or EPUB, will be added here shortly! Stay tuned! ⏳

---

### 🙏 Acknowledgements

* **Resemble-AI** for their incredible [Chatterbox-tts](https://github.com/resemble-ai/chatterbox) project. They're making AI voices sound truly human! 🗣️
* **santinic** for the inspiration from [audiblez](https://github.com/santinic/audiblez). Great minds think alike! 💡

---

### 💌 Contributing

Got ideas? Found a bug? Want to make Chatterblez even better? We'd love your contributions! Please feel free to open an issue or submit a pull request. Let's build something amazing together! 🤝

---

### 📜 License

\[Add your license information here, e.g., MIT License]

---

Made with ❤️ by cpttripzz ✨
Happy listening! 🎧📖💖

---

Let me know if you’d like to add demo commands, screenshots, or a `chatterblez.py` usage example next.
