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
- Optional SRT subtitle output

## Usage

### Command Line

```bash
# Basic usage
whisper-transcribe audio_file.mp3

# Advanced usage
whisper-transcribe audio_file.mp3 -m openai/whisper-small \
  --min-segment 5 \
  --max-segment 15 \
  --silence-duration 0.2 \
  --sample-rate 16000 \
  --batch-size 8 \
  --normalize \
  --hf-token YOUR_HF_TOKEN \
  --no-timestamps
```

#### Available Arguments:

- `input`: Input audio file or directory (required)
- `-o, --output`: Output file path (optional)
- `-m, --model`: Whisper model to use (default: openai/whisper-small)
- `--hf-token`: HuggingFace API token
- `--min-segment`: Minimum segment length in seconds (default: 5)
- `--max-segment`: Maximum segment length in seconds (default: 15)
- `--silence-duration`: Minimum silence duration in seconds (default: 0.2)
- `--sample-rate`: Audio sample rate (default: 16000)
- `--batch-size`: Batch size for transcription (default: 8)
- `--normalize`: Normalize audio volume
- `--no-text-normalize`: Skip text normalization
- `--no-timestamps`: Don't print timestamps during processing

### Python Library

```python
from whisper_transcriber import WhisperTranscriber

# Initialize the transcriber
transcriber = WhisperTranscriber(model_name="openai/whisper-small", hf_token="YOUR_HF_TOKEN")

# Transcribe an audio file
results = transcriber.transcribe(
    "audio_file.mp3",
    min_segment=5,
    max_segment=15,
    silence_duration=0.2,
    sample_rate=16000,
    batch_size=8,
    normalize=True,
    normalize_text=True,
    print_timestamps=True
)

# Optionally save to an SRT file
# If you want to save the transcription, provide an output path
results = transcriber.transcribe(
    "audio_file.mp3",
    output="transcript.srt"
)

# Access the transcription results
for i, segment in enumerate(results):
    print(f"Segment {i+1}: {segment['transcript']}")
```

## License

MIT
