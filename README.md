# Whisper Transcriber

A Python library for transcribing audio files using Whisper models with intelligent silence detection and segmentation.

## Installation

```bash
pip install whisper-transcriber
```

## Requirements

- Python 3.7 or higher
- ffmpeg and ffprobe installed on your system

## Features

- Intelligent silence detection for natural segmentation
- Adaptive audio analysis for optimal threshold detection
- High-quality transcription using Whisper models
- Support for various audio formats
- SRT subtitle output

## Usage

### Command Line

```bash
# Basic usage
whisper-transcribe audio_file.mp3

# Advanced usage
whisper-transcribe audio_file.mp3 --model openai/whisper-smal --output transcript.srt --min-segment 3 --max-segment 12
```

### Python Library

```python
from whisper_transcriber import WhisperTranscriber

# Initialize the transcriber
transcriber = WhisperTranscriber(model_name="openai/whisper-small", hf_token="YOUR_HF_TOKEN")

# Transcribe an audio file
results = transcriber.transcribe(
    "audio_file.mp3",
    output="transcript.srt",
    min_segment=5,
    max_segment=15,
    normalize_text=True
)

# Access the transcription results
for i, segment in enumerate(results):
    print(f"Segment {i+1}: {segment['transcript']}")
```

## License

MIT
