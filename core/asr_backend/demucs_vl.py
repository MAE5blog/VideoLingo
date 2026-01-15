import os
import gc
import subprocess
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich import print as rprint
from torch.cuda import is_available as is_cuda_available
from core.utils.models import *


def _load_demucs_modules():
    try:
        from demucs.pretrained import get_model
        from demucs.audio import save_audio
        from demucs.api import Separator
    except Exception:
        return None
    return get_model, save_audio, Separator


def _convert_audio_to_mp3(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_path),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dst_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _demucs_cli_separate(raw_audio_path: Path):
    console = Console()
    out_dir = Path(_AUDIO_DIR) / "demucs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = "htdemucs"
    device = "cuda" if is_cuda_available() else "cpu"
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model_name,
        "--two-stems",
        "vocals",
        "-o",
        str(out_dir),
        "-d",
        device,
        str(raw_audio_path),
    ]
    console.print(f"[yellow]⚠️ Falling back to demucs CLI: {' '.join(cmd)}[/yellow]")
    subprocess.run(cmd, check=True)

    base = raw_audio_path.stem
    expected_dir = out_dir / model_name / base
    candidates = list(expected_dir.glob("vocals.*")) if expected_dir.exists() else list(out_dir.rglob("vocals.*"))
    if not candidates:
        raise FileNotFoundError("Demucs CLI finished, but vocals stem not found.")

    vocal_src = candidates[0]
    bg_src = vocal_src.with_name("no_vocals" + vocal_src.suffix)
    if not bg_src.exists():
        # Fallback: try other sources merged by demucs CLI (if any)
        bg_candidates = list(vocal_src.parent.glob("no_vocals.*"))
        if not bg_candidates:
            raise FileNotFoundError("Demucs CLI finished, but no_vocals stem not found.")
        bg_src = bg_candidates[0]

    _convert_audio_to_mp3(vocal_src, Path(_VOCAL_AUDIO_FILE))
    _convert_audio_to_mp3(bg_src, Path(_BACKGROUND_AUDIO_FILE))

def demucs_audio():
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return

    demucs_modules = _load_demucs_modules()
    if demucs_modules is None:
        _demucs_cli_separate(Path(_RAW_AUDIO_FILE))
        return

    get_model, save_audio, Separator = demucs_modules

    class PreloadedSeparator(Separator):
        def __init__(self, model, shifts: int = 1, overlap: float = 0.25,
                     split: bool = True, segment=None, jobs: int = 0):
            self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
            device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                                  segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)

    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    
    console.print("🤖 Loading <htdemucs> model...")
    model = get_model('htdemucs')
    separator = PreloadedSeparator(model=model, shifts=1, overlap=0.25)
    
    console.print("🎵 Separating audio...")
    _, outputs = separator.separate_audio_file(_RAW_AUDIO_FILE)
    
    kwargs = {"samplerate": model.samplerate, "bitrate": 128, "preset": 2, 
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}
    
    console.print("🎤 Saving vocals track...")
    save_audio(outputs['vocals'].cpu(), _VOCAL_AUDIO_FILE, **kwargs)
    
    console.print("🎹 Saving background music...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), _BACKGROUND_AUDIO_FILE, **kwargs)
    
    # Clean up memory
    del outputs, background, model, separator
    gc.collect()
    
    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_audio()
