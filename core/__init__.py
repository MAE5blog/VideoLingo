import importlib
import sys

_MODULE_NAMES = {
    '_1_ytdlp',
    '_2_asr',
    '_3_1_split_nlp',
    '_3_2_split_meaning',
    '_4_1_summarize',
    '_4_2_translate',
    '_5_split_sub',
    '_6_gen_sub',
    '_7_sub_into_vid',
    '_8_1_audio_task',
    '_8_2_dub_chunks',
    '_9_refer_audio',
    '_10_gen_audio',
    '_11_merge_audio',
    '_12_dub_to_vid',
}

_UTIL_EXPORTS = {
    'ask_gpt',
    'load_key',
    'update_key',
    'cleanup',
    'delete_dubbing_files',
    'except_handler',
    'check_file_exists',
    'rprint',
    'get_joiner',
}

def __getattr__(name):
    if name in _MODULE_NAMES:
        module = importlib.import_module(f"{__name__}.{name}")
        setattr(sys.modules[__name__], name, module)
        return module
    if name in _UTIL_EXPORTS:
        if name == 'cleanup':
            from .utils.onekeycleanup import cleanup
            setattr(sys.modules[__name__], name, cleanup)
            return cleanup
        if name == 'delete_dubbing_files':
            from .utils.delete_retry_dubbing import delete_dubbing_files
            setattr(sys.modules[__name__], name, delete_dubbing_files)
            return delete_dubbing_files
        from . import utils as _utils
        value = getattr(_utils, name)
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'ask_gpt',
    'load_key',
    'update_key',
    'cleanup',
    'delete_dubbing_files',
    '_1_ytdlp',
    '_2_asr',
    '_3_1_split_nlp',
    '_3_2_split_meaning',
    '_4_1_summarize',
    '_4_2_translate',
    '_5_split_sub',
    '_6_gen_sub',
    '_7_sub_into_vid',
    '_8_1_audio_task',
    '_8_2_dub_chunks',
    '_9_refer_audio',
    '_10_gen_audio',
    '_11_merge_audio',
    '_12_dub_to_vid'
]
