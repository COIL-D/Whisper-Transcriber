"""Command-line interface for Whisper Transcriber."""

import argparse
import sys
from .transcriber import WhisperTranscriber
from .utils import check_dependencies

def main():
    """Command line interface for the whisper transcriber."""
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper models")
    
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("-o", "--output", help="Output file path (default: input filename with .srt extension)")
    parser.add_argument("-m", "--model", default="openai/whisper-small", 
                        help="Whisper model to use (default: openai/whisper-small)")
    parser.add_argument("--hf-token", help="HuggingFace API token")
    parser.add_argument("--min-segment", type=float, default=5, 
                        help="Minimum segment length in seconds (default: 5)")
    parser.add_argument("--max-segment", type=float, default=15, 
                        help="Maximum segment length in seconds (default: 15)")
    parser.add_argument("--silence-duration", type=float, default=0.2, 
                        help="Minimum silence duration in seconds (default: 0.2)")
    parser.add_argument("--sample-rate", type=int, default=16000, 
                        help="Audio sample rate (default: 16000)")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Batch size for transcription (default: 8)")
    parser.add_argument("--normalize", action="store_true", 
                        help="Normalize audio volume")
    parser.add_argument("--no-text-normalize", action="store_true", 
                        help="Skip text normalization")
    
    args = parser.parse_args()
    
    # If no output specified, use input filename with .srt extension
    if not args.output:
        args.output = args.input.rsplit(".", 1)[0] + ".srt"
    
    try:
        transcriber = WhisperTranscriber(args.model, args.hf_token)
        
        transcriber.transcribe(
            args.input,
            output=args.output,
            min_segment=args.min_segment,
            max_segment=args.max_segment,
            silence_duration=args.silence_duration,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            normalize=args.normalize,
            normalize_text=not args.no_text_normalize
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
