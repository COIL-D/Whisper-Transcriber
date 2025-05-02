"""
Whisper Transcriber - A library for transcribing audio files using Whisper models
"""

__version__ = "0.1.0"

from .transcriber import WhisperTranscriber
from .audio_processing import analyze_audio_levels, analyze_full_audio_for_silence
from .utils import format_timestamp, format_srt_timestamp

__all__ = [
    'WhisperTranscriber',
    'analyze_audio_levels',
    'analyze_full_audio_for_silence',
    'format_timestamp',
    'format_srt_timestamp'
]
