"""Main transcriber module for the Whisper Transcriber package."""

import os
import torch
import numpy as np
import librosa
import time
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login

from .audio_processing import (
    get_audio_duration,
    analyze_full_audio_for_silence,
    create_segments_from_silence
)
from .utils import check_dependencies, format_timestamp, format_srt_timestamp, normalize_kannada


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
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        # Check dependencies
        if not check_dependencies():
            raise RuntimeError("Missing required dependencies. Please install ffmpeg.")
            
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model and processor."""
        if self.hf_token:
            login(token=self.hf_token)
            
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            print(f"Successfully loaded model from {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def transcribe(self, input_file, output=None, min_segment=5, max_segment=15, 
                  silence_duration=0.2, sample_rate=16000, batch_size=8, 
                  normalize=False, normalize_text=True):
        """
        Transcribe an audio file and optionally save the results to a file.
        
        Args:
            input_file (str): Path to the input audio file
            output (str, optional): Path to save the transcription
            min_segment (float): Minimum segment length in seconds
            max_segment (float): Maximum segment length in seconds
            silence_duration (float): Minimum silence duration to consider as a segment boundary
            sample_rate (int): Audio sample rate
            batch_size (int): Batch size for transcription
            normalize (bool): Whether to normalize audio
            normalize_text (bool): Whether to normalize transcription text
            
        Returns:
            list: List of transcription results
        """
        # Start timing the process
        script_start_time = time.time()
        
        # Check if input file exists
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
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
            normalize_text=normalize_text
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
    
    def _transcribe_audio(self, input_file, segment_boundaries, sample_rate=16000, 
                         normalize=False, batch_size=8, normalize_text=True):
        """
        Transcribes audio using segment boundaries for timing information.
        
        Args:
            input_file (str): Path to the audio file
            segment_boundaries (list): List of segment boundaries
            sample_rate (int): Audio sample rate
            normalize (bool): Whether to normalize audio
            batch_size (int): Batch size for transcription
            normalize_text (bool): Whether to normalize transcription text
            
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
        progress_bar = tqdm(total=len(segment_pairs), desc="Processing segments", unit="segment")
        
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
                            
                            # Print each result immediately
                            segment_start_time_str = format_timestamp(segment["start"])
                            segment_end_time_str = format_timestamp(segment["end"])
                            print(f"\nSegment {segment['index']+1}: [{segment_start_time_str} --> {segment_end_time_str}] {transcript}")
                except Exception as e:
                    print(f"Error transcribing batch: {str(e)}")
            
            progress_bar.update(len(current_batch))
        
        progress_bar.close()
        
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
