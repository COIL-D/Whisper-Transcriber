"""Main transcriber module for the Whisper Transcriber package."""

import os
import torch
import numpy as np
import librosa
import time
import secrets
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login

from .audio_processing import (
    get_audio_duration,
    analyze_full_audio_for_silence,
    create_segments_from_silence
)
from .utils import check_dependencies, format_timestamp, format_srt_timestamp, normalize_kannada, validate_file_path


class WhisperTranscriber:
    """
    A class for transcribing audio files using Whisper models.
    """
    
    def __init__(self, model_name="openai/whisper-small", hf_token=None):
        """
        Initialize the transcriber with a Whisper model.
        
        Args:
            model_name (str): The name of the Whisper model to use
            hf_token (str, optional): HuggingFace API token
        """
        # Sanitize model name to prevent injection
        self._validate_model_name(model_name)
        self.model_name = model_name
        
        # Securely handle the token
        self.hf_token = self._secure_token_handling(hf_token)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        # Check dependencies
        if not check_dependencies():
            raise RuntimeError("Missing required dependencies. Please install ffmpeg.")
            
        # Load model
        self._load_model()
    
    def _secure_token_handling(self, token):
        """
        Securely handle the API token.
        
        Args:
            token (str): The API token to handle
            
        Returns:
            str: The API token
        """
        if token is None:
            # Try to get from environment variable
            token = os.environ.get("HF_TOKEN")
            
        # Don't store the token directly as attribute in plaintext
        # Instead, store a reference to it that will be used only when needed
        # This is a simple obfuscation, not true security - tokens should ideally
        # be managed by a secure credential store
        if token:
            # Generate a random key for simple obfuscation
            random_key = secrets.token_bytes(32)
            # XOR the token with the key for obfuscation
            token_bytes = token.encode('utf-8')
            token_bytes_padded = token_bytes + b'\0' * (32 - len(token_bytes) % 32)
            obfuscated = bytes(a ^ b for a, b in zip(token_bytes_padded, random_key * (len(token_bytes_padded) // len(random_key) + 1)))
            
            return {
                "obfuscated": obfuscated,
                "key": random_key,
                "length": len(token_bytes)
            }
        
        return None
    
    def _get_token(self):
        """
        Retrieve the original token when needed.
        
        Returns:
            str: The original API token or None
        """
        if self.hf_token is None:
            return None
            
        # Deobfuscate the token
        deobfuscated = bytes(a ^ b for a, b in zip(
            self.hf_token["obfuscated"], 
            self.hf_token["key"] * (len(self.hf_token["obfuscated"]) // len(self.hf_token["key"]) + 1)
        ))
        
        return deobfuscated[:self.hf_token["length"]].decode('utf-8')
    
    def _validate_model_name(self, model_name):
        """
        Validate the model name for basic security checks.
        
        Args:
            model_name (str): The model name to validate
            
        Raises:
            ValueError: If the model name is invalid
        """
        # Check type
        if not isinstance(model_name, str):
            raise ValueError("Model name must be a string")
            
        # Only check for dangerous characters that could be used for injection
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./")
        
        # Basic check for bad characters
        if not all(c in allowed_chars for c in model_name):
            raise ValueError(f"Invalid characters in model name: {model_name}")
        
        # All models are allowed as long as they pass the basic security check
    
    def _load_model(self):
        """Load the Whisper model and processor."""
        # Retrieve the deobfuscated token if it exists
        token = self._get_token()
        
        if token:
            login(token=token)
            
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            print(f"Successfully loaded model from {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def transcribe(self, input_file, output=None, min_segment=5, max_segment=15, 
                  silence_duration=0.2, sample_rate=16000, batch_size=8, 
                  normalize=False, normalize_text=True, print_timestamps=False, **kwargs):
        """
        Transcribe an audio file and optionally save the results to a file.
        
        Args:
            input_file (str): Path to the input audio file
            output (str, optional): Path to save the transcription (optional)
            min_segment (float): Minimum segment length in seconds
            max_segment (float): Maximum segment length in seconds
            silence_duration (float): Minimum silence duration to consider as a segment boundary
            sample_rate (int): Audio sample rate
            batch_size (int): Batch size for transcription
            normalize (bool): Whether to normalize audio
            normalize_text (bool): Whether to normalize transcription text
            print_timestamps (bool): Whether to print timestamps during processing
            **kwargs: Additional parameters for future compatibility
            
        Returns:
            list: List of transcription results
        """
        # Start timing the process
        script_start_time = time.time()
        
        # Sanitize inputs
        self._validate_parameters(min_segment, max_segment, silence_duration, sample_rate, batch_size)
        
        # Check if input file exists and is safe
        if not os.path.isfile(input_file) or not validate_file_path(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found or not safe to use")
        
        # Check output path if specified
        if output and not validate_file_path(output):
            raise ValueError(f"Invalid output file path: {output}")
        
        # Get audio duration
        total_duration = get_audio_duration(input_file)
        if total_duration <= 0:
            raise ValueError(f"Could not determine duration of {input_file}")
        
        print(f"Processing file: {input_file} (duration: {total_duration:.2f} seconds)")
        
        # Analyze audio for silence points
        silence_data = analyze_full_audio_for_silence(
            input_file,
            silence_duration=silence_duration,
            adaptive=True,
            min_silence_points=6
        )
        
        # Create segment boundaries from silence data
        segment_boundaries = create_segments_from_silence(
            silence_data,
            total_duration,
            min_segment_length=min_segment,
            max_segment_length=max_segment
        )
        
        # Transcribe using segment boundaries
        print("Starting transcription...")
        results = self._transcribe_audio(
            input_file,
            segment_boundaries,
            sample_rate=sample_rate,
            normalize=normalize,
            batch_size=batch_size,
            normalize_text=normalize_text,
            print_timestamps=print_timestamps
        )
        
        if not results:
            raise ValueError("No transcription results")

        # Format and save results if output is specified
        if output:
            self.save_transcription(results, output)
        
        # Calculate and display the total processing time
        end_time = time.time()
        elapsed_time = end_time - script_start_time
        
        # Format the elapsed time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        
        time_format = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}" if hours > 0 else f"{minutes:02d}:{seconds:06.3f}"
        print(f"\nTotal processing time: {time_format}")
        print("\nTranscription complete!")
        
        return results
    
    def _validate_parameters(self, min_segment, max_segment, silence_duration, sample_rate, batch_size):
        """
        Validate the parameters to prevent attacks or errors.
        
        Args:
            min_segment (float): Minimum segment length in seconds
            max_segment (float): Maximum segment length in seconds
            silence_duration (float): Minimum silence duration in seconds
            sample_rate (int): Audio sample rate
            batch_size (int): Batch size for transcription
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Check types
        if not isinstance(min_segment, (int, float)) or min_segment <= 0:
            raise ValueError(f"Invalid min_segment: {min_segment}")
        
        if not isinstance(max_segment, (int, float)) or max_segment <= 0:
            raise ValueError(f"Invalid max_segment: {max_segment}")
            
        if not isinstance(silence_duration, (int, float)) or silence_duration <= 0:
            raise ValueError(f"Invalid silence_duration: {silence_duration}")
            
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {sample_rate}")
            
        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 64:
            raise ValueError(f"Invalid batch_size: {batch_size}")
            
        # Check relationships
        if min_segment >= max_segment:
            raise ValueError(f"min_segment ({min_segment}) must be less than max_segment ({max_segment})")
            
        # Check reasonable ranges
        if min_segment < 0.5:
            print("Warning: Very small min_segment may cause excessive segmentation")
            
        if max_segment > 30:
            print("Warning: Very large max_segment may cause memory issues")
            
        if silence_duration < 0.05:
            print("Warning: Very small silence_duration may detect too many silence points")
            
        # Check sample rate in standard ranges
        valid_sample_rates = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
        if sample_rate not in valid_sample_rates:
            print(f"Warning: Unusual sample_rate {sample_rate}. Standard rates are {valid_sample_rates}")
    
    def _transcribe_audio(self, input_file, segment_boundaries, sample_rate=16000, 
                         normalize=False, batch_size=8, normalize_text=True, print_timestamps=False):
        """
        Transcribes audio using segment boundaries for timing information.
        
        Args:
            input_file (str): Path to the audio file
            segment_boundaries (list): List of segment boundaries
            sample_rate (int): Audio sample rate
            normalize (bool): Whether to normalize audio
            batch_size (int): Batch size for transcription
            normalize_text (bool): Whether to normalize transcription text
            print_timestamps (bool): Whether to print timestamps during processing
            
        Returns:
            list: List of transcription results
        """
        # Split segments into batches for transcription
        segment_pairs = []
        min_valid_duration = 0.2  # Minimum valid segment duration in seconds
        
        for i in range(len(segment_boundaries) - 1):
            start = segment_boundaries[i]
            end = segment_boundaries[i+1]
            duration = end - start
            
            # Skip segments with zero or extremely short duration
            if duration >= min_valid_duration:
                segment_pairs.append((start, end))
            else:
                print(f"Warning: Skipping invalid segment with duration {duration:.3f}s [{format_timestamp(start)} --> {format_timestamp(end)}]")
        
        print(f"Processing {len(segment_pairs)} audio segments...")
        results = []
        
        # Load the full audio file
        try:
            full_audio, _ = librosa.load(input_file, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return []
        
        # Process in batches to optimize memory usage and GPU utilization
        
        for batch_start in range(0, len(segment_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(segment_pairs))
            current_batch = segment_pairs[batch_start:batch_end]
            
            # Extract segments from the full audio
            batch_segments = []
            for i, (segment_start, segment_end) in enumerate(current_batch):
                # Calculate sample positions
                start_sample = int(segment_start * sample_rate)
                end_sample = int(segment_end * sample_rate)
                
                # Extract segment from the full audio
                if end_sample <= len(full_audio):
                    segment_audio = full_audio[start_sample:end_sample]
                    batch_segments.append({
                        "start": segment_start,
                        "end": segment_end,
                        "duration": segment_end - segment_start,
                        "audio": segment_audio,
                        "index": batch_start + i  # Track original position
                    })
            
            # Transcribe the batch if we have segments
            if batch_segments:
                batch_audio = [segment["audio"] for segment in batch_segments]
                
                try:
                    with torch.no_grad():
                        # Process the batch
                        batch_input_features = self.processor(
                            batch_audio,
                            sampling_rate=sample_rate,
                            return_tensors="pt"
                        ).input_features.to(self.device)
                        
                        predicted_ids = self.model.generate(
                            batch_input_features,
                            max_length=448,
                            num_beams=5,
                            early_stopping=True
                        )
                        
                        transcripts = self.processor.batch_decode(
                            predicted_ids,
                            skip_special_tokens=True
                        )
                        
                        # Process and store results
                        for j, transcript in enumerate(transcripts):
                            segment = batch_segments[j]
                            
                            # Apply normalization if requested
                            if normalize_text:
                                transcript = normalize_kannada(transcript)
                            
                            result = {
                                "start": segment["start"],
                                "end": segment["end"],
                                "duration": segment["duration"],
                                "transcript": transcript,
                                "index": segment["index"]
                            }
                            
                            results.append(result)
                            
                            # Print transcript with optional timestamps
                            output_line = "\n"
                            
                            # Only add timestamps if requested
                            if print_timestamps:
                                segment_start_time_str = format_timestamp(segment["start"])
                                segment_end_time_str = format_timestamp(segment["end"])
                                output_line += f"[{segment_start_time_str} --> {segment_end_time_str}] "
                            
                            # Always add transcript (default behavior)
                            output_line += transcript
                                
                            print(output_line)
                except Exception as e:
                    print(f"Error transcribing batch: {str(e)}")
            
        
        # Sort results by their original index
        results.sort(key=lambda x: x["index"])
        return results

    def save_transcription(self, results, output_file):
        """
        Save the transcription results to a file.
        
        Args:
            results (list): List of transcription results
            output_file (str): Path to save the transcription
        """
        # Validate output path
        if not validate_file_path(output_file):
            raise ValueError(f"Invalid output file path: {output_file}")
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise OSError(f"Could not create output directory: {str(e)}")
        
        # Ensure output has .srt extension
        if not output_file.lower().endswith('.srt'):
            output_file = os.path.splitext(output_file)[0] + '.srt'
            
        output_lines = []
        for i, result in enumerate(results):
            # Format in SRT format
            srt_entry = [
                str(i + 1),
                f"{format_srt_timestamp(result['start'])} --> {format_srt_timestamp(result['end'])}",
                result["transcript"],
                ""  # Empty line between entries
            ]
            output_lines.extend(srt_entry)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
            print(f"\nSubtitles saved in SRT format to {output_file}")
        except Exception as e:
            print(f"Error saving output: {str(e)}")
