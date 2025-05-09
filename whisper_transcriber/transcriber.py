"""Main transcriber module for the Whisper Transcriber package."""

import os
import torch
import numpy as np
import librosa
import time
import secrets
import re
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login
import concurrent.futures
from pathlib import Path
import tempfile

from .audio_processing import (
    get_audio_duration,
    analyze_full_audio_for_silence,
    create_segments_from_silence,
    extract_audio_segment
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
                  normalize=False, normalize_text=True, print_timestamps=False, 
                  verbose=True, two_pass=False, chunk_size=600, parallel_jobs=None,
                  temperature=0.0, top_p=None, beam_size=5, **kwargs):
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
            verbose (bool): Whether to print processing information and transcripts during processing
            two_pass (bool): Whether to use two-pass transcription for improved segment boundaries
            chunk_size (int): Size of audio chunks in seconds for memory-efficient processing
            parallel_jobs (int): Number of parallel jobs for silence detection (None for auto)
            temperature (float): Temperature for sampling during generation (0.0 for deterministic)
            top_p (float): Top-p probability threshold for nucleus sampling
            beam_size (int): Beam size for beam search during generation
            **kwargs: Additional parameters for future compatibility
            
        Returns:
            list: List of transcription results
        """
        # Start timing the process
        script_start_time = time.time()
        
        # Check for parameter inconsistency before validation
        if min_segment >= max_segment:
            if verbose:
                print(f"Error: min_segment ({min_segment}) must be less than max_segment ({max_segment})")
                
            # Try to prompt for corrected values
            try:
                # Check if running in interactive environment
                import sys
                is_interactive = hasattr(sys, 'ps1') or sys.flags.interactive
                
                if is_interactive:
                    print("Please enter corrected values:")
                    try:
                        new_min = float(input(f"Enter min_segment (current: {min_segment}, must be < {max_segment}): "))
                        if new_min >= max_segment:
                            new_max = float(input(f"Enter max_segment (current: {max_segment}, must be > {new_min}): "))
                        else:
                            new_max = max_segment
                            
                        print(f"Using corrected values: min_segment={new_min}, max_segment={new_max}")
                        min_segment = new_min
                        max_segment = new_max
                    except ValueError as e:
                        print(f"Invalid input: {e}")
                        print("Using default values: min_segment=5, max_segment=15")
                        min_segment = 5
                        max_segment = 15
                else:
                    # Not in interactive mode, use defaults
                    print("Not in interactive environment. Using default values: min_segment=5, max_segment=15")
                    min_segment = 5
                    max_segment = 15
            except Exception:
                # Fall back to defaults if anything goes wrong
                print("Error detecting environment. Using default values: min_segment=5, max_segment=15")
                min_segment = 5
                max_segment = 15
        
        # Sanitize inputs
        self._validate_parameters(min_segment, max_segment, silence_duration, sample_rate, batch_size, 
                                 temperature, beam_size, chunk_size)
        
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
        
        if verbose:
            print(f"Processing file: {input_file} (duration: {total_duration:.2f} seconds)")
        
        # Handle short audio files directly
        if total_duration < min_segment and total_duration < max_segment:
            if verbose:
                print(f"Audio duration ({total_duration:.2f}s) is less than min_segment ({min_segment}s) and max_segment ({max_segment}s). Transcribing as a single segment.")
            segment_boundaries = [0.0, total_duration]
            
            # Skip silence detection and directly transcribe the short audio
            results = self._transcribe_audio(
                input_file,
                segment_boundaries,
                sample_rate=sample_rate,
                normalize=normalize,
                batch_size=batch_size,
                normalize_text=normalize_text,
                print_timestamps=print_timestamps,
                verbose=verbose,
                chunk_size=chunk_size,
                temperature=temperature,
                top_p=top_p,
                beam_size=beam_size
            )
        else:
            # For longer audio, do the full silence detection and transcription
            # Determine parallel jobs if not specified
            if parallel_jobs is None:
                parallel_jobs = min(os.cpu_count() or 2, 8)  # Default to min(CPU count, 8)
            
            # Analyze audio for silence points with parallel processing
            if verbose:
                print(f"Using {parallel_jobs} parallel jobs for silence detection")
                
            silence_data = analyze_full_audio_for_silence(
                input_file,
                silence_duration=silence_duration,
                adaptive=True,
                min_silence_points=6,
                parallel_jobs=parallel_jobs,
                chunk_size=chunk_size
            )
            
            # Create segment boundaries from silence data
            segment_boundaries = create_segments_from_silence(
                silence_data,
                total_duration,
                min_segment_length=min_segment,
                max_segment_length=max_segment
            )
        
            # Two-pass approach for improved segment boundaries
            if two_pass and total_duration > max_segment * 2:
                if verbose:
                    print("Performing two-pass transcription for improved segment boundaries...")
                
                # First pass: Initial transcription with current segment boundaries
                initial_results = self._transcribe_audio(
                    input_file,
                    segment_boundaries,
                    sample_rate=sample_rate,
                    normalize=normalize,
                    batch_size=batch_size,
                    normalize_text=normalize_text,
                    verbose=False,  # No verbose output for first pass
                    print_timestamps=False,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_p=top_p,
                    beam_size=beam_size
                )
                
                # Refine segment boundaries based on linguistic analysis
                if verbose:
                    print("Refining segment boundaries based on initial transcription...")
                    
                refined_boundaries = self._refine_segment_boundaries(
                    initial_results,
                    segment_boundaries,
                    total_duration,
                    min_segment_length=min_segment,
                    max_segment_length=max_segment
                )
                
                # Use refined segment boundaries
                segment_boundaries = refined_boundaries
                
                if verbose:
                    print(f"Refined segment count: {len(segment_boundaries) - 1}")
            
            # Perform final transcription with (potentially refined) segment boundaries
            if verbose:
                print("Starting transcription...")
            
            results = self._transcribe_audio(
                input_file,
                segment_boundaries,
                sample_rate=sample_rate,
                normalize=normalize,
                batch_size=batch_size,
                normalize_text=normalize_text,
                print_timestamps=print_timestamps,
                verbose=verbose,
                chunk_size=chunk_size,
                temperature=temperature,
                top_p=top_p,
                beam_size=beam_size
            )
        
        if not results:
            raise ValueError("No transcription results")

        # Format and save results if output is specified
        if output:
            self.save_transcription(results, output)
        
        # Calculate and display the total processing time
        if verbose:
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

    def _validate_parameters(self, min_segment, max_segment, silence_duration, sample_rate, batch_size,
                          temperature=0.0, beam_size=5, chunk_size=600):
        """
        Validate the parameters to prevent attacks or errors.
        
        Args:
            min_segment (float): Minimum segment length in seconds
            max_segment (float): Maximum segment length in seconds
            silence_duration (float): Minimum silence duration in seconds
            sample_rate (int): Audio sample rate
            batch_size (int): Batch size for transcription
            temperature (float): Temperature parameter for generation
            beam_size (int): Beam size for beam search
            chunk_size (int): Size of audio chunks in seconds
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Original parameter checks
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
            
        # New parameter checks
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 1.5:
            raise ValueError(f"Invalid temperature: {temperature}. Must be between 0.0 and 1.5")
            
        if not isinstance(beam_size, int) or beam_size < 1 or beam_size > 10:
            raise ValueError(f"Invalid beam_size: {beam_size}. Must be between 1 and 10")
            
        if not isinstance(chunk_size, int) or chunk_size < 10 or chunk_size > 3600:
            raise ValueError(f"Invalid chunk_size: {chunk_size}. Must be between 10 and 3600 seconds")
    
    def _transcribe_audio(self, input_file, segment_boundaries, sample_rate=16000, 
                        normalize=False, batch_size=8, normalize_text=True, 
                        print_timestamps=False, verbose=True, chunk_size=600,
                        temperature=0.0, top_p=None, beam_size=5):
        """
        Transcribes audio using segment boundaries for timing information with memory-efficient chunking.
        
        Args:
            input_file (str): Path to the audio file
            segment_boundaries (list): List of segment boundaries
            sample_rate (int): Audio sample rate
            normalize (bool): Whether to normalize audio
            batch_size (int): Batch size for transcription
            normalize_text (bool): Whether to normalize transcription text
            print_timestamps (bool): Whether to print timestamps during processing
            verbose (bool): Whether to print progress and transcripts during processing
            chunk_size (int): Size of audio chunks in seconds for memory-efficient processing
            temperature (float): Temperature for sampling (0.0 for deterministic)
            top_p (float): Top-p probability threshold for nucleus sampling
            beam_size (int): Beam size for beam search
            
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
                if verbose:
                    print(f"Warning: Skipping invalid segment with duration {duration:.3f}s [{format_timestamp(start)} --> {format_timestamp(end)}]")
        
        if verbose:
            print(f"Processing {len(segment_pairs)} audio segments...")
        
        # Set up progress bar if verbose
        if verbose:
            pbar = tqdm(total=len(segment_pairs), desc="Transcribing segments", unit="segment")
            
        results = []
        
        # Process in batches, with memory efficiency (don't load entire file at once)
        for batch_start in range(0, len(segment_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(segment_pairs))
            current_batch = segment_pairs[batch_start:batch_end]
            
            # Extract segments from audio file directly rather than loading the entire file
            batch_segments = []
            for i, (segment_start, segment_end) in enumerate(current_batch):
                try:
                    # Extract audio segment directly using librosa's offset and duration parameters
                    segment_audio = extract_audio_segment(
                        input_file, 
                        segment_start, 
                        segment_end - segment_start,
                        sample_rate=sample_rate,
                        normalize=normalize
                    )
                    
                    if segment_audio is not None:
                        batch_segments.append({
                            "start": segment_start,
                            "end": segment_end,
                            "duration": segment_end - segment_start,
                            "audio": segment_audio,
                            "index": batch_start + i  # Track original position
                        })
                except Exception as e:
                    if verbose:
                        print(f"Error extracting audio segment [{format_timestamp(segment_start)} --> {format_timestamp(segment_end)}]: {str(e)}")
            
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
                        
                        # Prepare generation config with enhanced parameters
                        generation_kwargs = {
                            "max_length": 448,
                            "num_beams": beam_size,
                        }
                        
                        # Only set early_stopping when using beam search (beam_size > 1)
                        if beam_size > 1:
                            generation_kwargs["early_stopping"] = True
                        
                        # Add temperature if non-zero (for non-deterministic output)
                        if temperature > 0:
                            generation_kwargs["temperature"] = temperature
                            generation_kwargs["do_sample"] = True
                            
                            # Add top_p if specified (only relevant with do_sample=True)
                            if top_p is not None and top_p > 0 and top_p < 1:
                                generation_kwargs["top_p"] = top_p
                        # If temperature=0 (deterministic) and top_p is specified, warn user
                        elif top_p is not None and top_p > 0 and top_p < 1:
                            print("Warning: top_p is only used when temperature > 0. Ignoring top_p value.")
                        
                        predicted_ids = self.model.generate(
                            batch_input_features,
                            **generation_kwargs
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
                            
                            # Only print if verbose mode is enabled
                            if verbose:
                                output_line = "\n"
                                
                                # Only add timestamps if requested
                                if print_timestamps:
                                    segment_start_time_str = format_timestamp(segment["start"])
                                    segment_end_time_str = format_timestamp(segment["end"])
                                    output_line += f"[{segment_start_time_str} --> {segment_end_time_str}] "
                                
                                # Add transcript only in verbose mode
                                output_line += transcript
                                print(output_line)
                                
                        # Update progress bar if verbose
                        if verbose:
                            pbar.update(len(batch_segments))
                            
                except Exception as e:
                    if verbose:
                        print(f"Error transcribing batch: {str(e)}")
                    
                # Clear GPU memory between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        # Close progress bar if verbose
        if verbose:
            pbar.close()
        
        # Sort results by their original index
        results.sort(key=lambda x: x["index"])
        return results
        
    def _refine_segment_boundaries(self, initial_results, original_boundaries, total_duration,
                                min_segment_length=5, max_segment_length=15):
        """
        Refines segment boundaries based on linguistic analysis of initial transcription.
        
        Args:
            initial_results (list): List of initial transcription results
            original_boundaries (list): Original segment boundaries
            total_duration (float): Total audio duration
            min_segment_length (float): Minimum segment length
            max_segment_length (float): Maximum segment length
            
        Returns:
            list: Refined segment boundaries
        """
        if not initial_results or len(initial_results) < 2:
            return original_boundaries
            
        refined_boundaries = [0.0]  # Always start at 0
        
        # Analyzing transcript patterns to find natural breaks
        for result in initial_results:
            transcript = result["transcript"]
            segment_start = result["start"]
            segment_end = result["end"]
            segment_duration = result["duration"]
            
            # Skip very short segments as they're unlikely to need refinement
            if segment_duration < min_segment_length:
                continue
                
            # If segment is already at or below max length, keep its end boundary
            if segment_duration <= max_segment_length:
                if segment_end not in refined_boundaries and segment_end <= total_duration:
                    refined_boundaries.append(segment_end)
                continue
                
            # For long segments, look for natural language break points
            sentences = self._split_into_sentences(transcript)
            
            if len(sentences) <= 1:
                # No clear sentence breaks, divide evenly
                num_parts = max(1, int(segment_duration / max_segment_length))
                part_duration = segment_duration / num_parts
                
                for i in range(1, num_parts):
                    new_boundary = segment_start + (i * part_duration)
                    if new_boundary not in refined_boundaries:
                        refined_boundaries.append(new_boundary)
                
                # Always add the segment end if not already in boundaries
                if segment_end not in refined_boundaries:
                    refined_boundaries.append(segment_end)
            else:
                # Use sentence breaks to guide segmentation
                total_chars = sum(len(s) for s in sentences)
                if total_chars == 0:  # Safeguard against empty transcripts
                    if segment_end not in refined_boundaries:
                        refined_boundaries.append(segment_end)
                    continue
                    
                char_per_second = total_chars / segment_duration
                char_count = 0
                last_boundary = segment_start
                
                for i, sentence in enumerate(sentences):
                    char_count += len(sentence)
                    estimated_time = segment_start + (char_count / char_per_second)
                    
                    # Add boundary if enough time has passed
                    if (estimated_time - last_boundary >= min_segment_length or 
                        i == len(sentences) - 1):  # Always include the last sentence end
                        
                        # Don't exceed segment_end
                        boundary = min(estimated_time, segment_end)
                        
                        if (boundary not in refined_boundaries and 
                            boundary > refined_boundaries[-1] and
                            boundary <= total_duration):
                            refined_boundaries.append(boundary)
                            last_boundary = boundary
        
        # Ensure we include the total_duration as the final boundary
        if total_duration not in refined_boundaries:
            refined_boundaries.append(total_duration)
            
        # Final validation pass: remove boundaries that are too close together
        min_gap = 0.5  # Minimum 0.5s gap between segments
        filtered_boundaries = [refined_boundaries[0]]  # Always keep the first boundary
        
        for i in range(1, len(refined_boundaries)):
            if refined_boundaries[i] - filtered_boundaries[-1] >= min_gap:
                filtered_boundaries.append(refined_boundaries[i])
        
        return sorted(filtered_boundaries)
        
    def _split_into_sentences(self, transcript):
        """
        Splits transcript text into sentences based on punctuation.
        
        Args:
            transcript (str): The transcript to split
            
        Returns:
            list: List of sentences
        """
        # Basic sentence splitting on punctuation
        # More sophisticated NLP could be used here with external libraries
        sentence_pattern = r'[.!?।॥]+\s*'
        sentences = re.split(sentence_pattern, transcript)
        
        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

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
