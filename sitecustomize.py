try:
    import torchaudio
    import sys
    import types

    if not hasattr(torchaudio, "set_audio_backend"):
        def _noop_backend(_backend):
            return None

        torchaudio.set_audio_backend = _noop_backend
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if "torchaudio.backend" not in sys.modules:
        backend_mod = types.ModuleType("torchaudio.backend")
        backend_mod.set_audio_backend = torchaudio.set_audio_backend
        backend_mod.get_audio_backend = torchaudio.get_audio_backend
        backend_mod.list_audio_backends = torchaudio.list_audio_backends
        sys.modules["torchaudio.backend"] = backend_mod
except Exception:
    pass

try:
    import numpy as np
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
except Exception:
    pass
