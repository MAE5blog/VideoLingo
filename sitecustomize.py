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

    backend_mod = sys.modules.get("torchaudio.backend")
    if backend_mod is None:
        backend_mod = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend_mod
    if not hasattr(backend_mod, "__path__"):
        backend_mod.__path__ = []
    backend_mod.set_audio_backend = torchaudio.set_audio_backend
    backend_mod.get_audio_backend = torchaudio.get_audio_backend
    backend_mod.list_audio_backends = torchaudio.list_audio_backends

    if "torchaudio.backend.common" not in sys.modules:
        common_mod = types.ModuleType("torchaudio.backend.common")
        common_mod.set_audio_backend = torchaudio.set_audio_backend
        common_mod.get_audio_backend = torchaudio.get_audio_backend
        common_mod.list_audio_backends = torchaudio.list_audio_backends
        class AudioMetaData:
            def __init__(self, sample_rate=None, num_frames=None, num_channels=None, bits_per_sample=None, encoding=None):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

            def __iter__(self):
                return iter((self.sample_rate, self.num_frames, self.num_channels, self.bits_per_sample, self.encoding))

        common_mod.AudioMetaData = AudioMetaData
        sys.modules["torchaudio.backend.common"] = common_mod
except Exception:
    pass

try:
    import numpy as np
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
except Exception:
    pass
