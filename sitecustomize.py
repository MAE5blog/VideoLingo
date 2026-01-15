try:
    import torchaudio

    if not hasattr(torchaudio, "set_audio_backend"):
        def _noop_backend(_backend):
            return None

        torchaudio.set_audio_backend = _noop_backend
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
except Exception:
    pass

try:
    import numpy as np
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
except Exception:
    pass
