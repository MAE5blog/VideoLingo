import os
import gc
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
    except Exception as exc:
        raise ModuleNotFoundError(
            "Demucs is required for vocal separation. "
            "Please install demucs>=4.0.1 (e.g. `pip install -U demucs`)."
        ) from exc
    return get_model, save_audio, Separator

def demucs_audio():
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return

    get_model, save_audio, Separator = _load_demucs_modules()

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
