import subprocess
import re
import os
import argparse
import sys
import torch
import numpy as np
from tqdm import tqdm
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login
import regex as re
import multiprocessing
import time  # Add time module for duration tracking

# List of supported audio extensions
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.m4a', '.flac')

def normalize_kannada(text):
    """Normalize Kannada text for consistent evaluation"""
    text = text.lower().strip()
    text = re.sub(r'[–—-]', ' ', text)
    text = re.sub(r'[.,!?।॥]', '', text)
    text = re.sub(r'\u0CBC', '', text)
    text = re.sub(r'[\u0C82\u0C83]', '\u0C82', text)
    
    vowel_marks = {
        '\u0CBE\u0CBE': '\u0CBE', '\u0CBF\u0CBF': '\u0CBF',
        '\u0CC0\u0CC0': '\u0CC0', '\u0CC1\u0CC1': '\u0CC1',
        '\u0CC2\u0CC2': '\u0CC2', '\u0CC3\u0CC3': '\u0CC3',
        '\u0CC6\u0CC6': '\u0CC6', '\u0CC7\u0CC7': '\u0CC7',
        '\u0CC8\u0CC8': '\u0CC8', '\u0CCA\u0CCA': '\u0CCA',
        '\u0CCB\u0CCB': '\u0CCB', '\u0CCC\u0CCC': '\u0CCC'
    }
    
    for old, new in vowel_marks.items():
        text = text.replace(old, new)
    
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def check_dependencies():
    """
    Check if ffmpeg and ffprobe are installed and available.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        print("Error: ffmpeg and/or ffprobe not found. Please install ffmpeg.")
        return False

def get_audio_duration(input_file):
    """
    Use ffprobe to obtain the duration (in seconds) of the input audio file.
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        duration = float(result.stdout.decode().strip())
        return duration
    except (subprocess.SubprocessError, ValueError) as e:
        print(f"Error getting audio duration: {e}")
        return 0.0

def analyze_audio_levels(input_file):
    """
    Analyze audio file to get mean and peak volume levels for better silence detection threshold.
    Returns a dynamically calculated silence threshold based on audio characteristics.
    """
    command = [
        "ffmpeg",
        "-i", input_file,
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    
    try:
        process = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
        stderr_output = process.stderr.decode()
        
        # Look for mean_volume and max_volume in the output
        mean_match = re.search(r"mean_volume:\s*([-\d\.]+)\s*dB", stderr_output)
        max_match = re.search(r"max_volume:\s*([-\d\.]+)\s*dB", stderr_output)
        
        mean_volume = float(mean_match.group(1)) if mean_match else -25
        max_volume = float(max_match.group(1)) if max_match else -5
        
        # Calculate dynamic ratio based on the difference between max and mean
        dynamic_range = max_volume - mean_volume
        
        # Adjust threshold more intelligently based on audio characteristics
        if dynamic_range > 40:  # High dynamic range (music or mixed content)
            # For high dynamic range, set threshold closer to mean to catch more silence points
            threshold_offset = min(30, dynamic_range * 0.5)
        elif dynamic_range > 20:  # Medium dynamic range (typical speech)
            threshold_offset = min(25, dynamic_range * 0.6)
        else:  # Low dynamic range (compressed audio or consistent speech)
            threshold_offset = min(20, dynamic_range * 0.7)
        
        # Ensure threshold is at least 15dB below mean and not too extreme
        silence_threshold = max(mean_volume - threshold_offset, -60)
        
        print(f"Audio analysis: Mean volume: {mean_volume:.2f}dB, Max volume: {max_volume:.2f}dB")
        print(f"Dynamic range: {dynamic_range:.2f}dB, Calculated silence threshold: {silence_threshold:.2f}dB")
        
        return silence_threshold
    except subprocess.SubprocessError as e:
        print(f"Error analyzing audio levels: {e}")
        return -30  # Default if analysis fails

def analyze_full_audio_for_silence(input_file, silence_threshold=-30, silence_duration=0.2, adaptive=True, min_silence_points=6):
    """
    Analyze the entire audio file for silence points with enhanced adaptive threshold algorithm.
    Returns a list of silence start and end points.
    
    Parameters:
    - input_file: Path to the audio file
    - silence_threshold: Initial silence threshold in dB (only used if adaptive=False)
    - silence_duration: Minimum silence duration to consider as a segment boundary
    - adaptive: Whether to adaptively determine the silence threshold
    - min_silence_points: Minimum number of silence points needed for good segmentation
    """
    if adaptive:
        # Get adaptive threshold with improved algorithm
        silence_threshold = analyze_audio_levels(input_file)
        print(f"Using adaptive silence threshold: {silence_threshold:.2f}dB")
    
    print("Performing full audio silence analysis...")
    
    # First attempt with initial threshold
    silence_data = _detect_silence_points(input_file, silence_threshold, silence_duration)
    
    # If not enough silence points found, try with progressively more lenient thresholds
    # using a more sophisticated approach
    if len(silence_data) < min_silence_points:
        # Calculate the number of seconds per silence point we'd expect
        audio_duration = get_audio_duration(input_file)
        expected_seconds_per_point = audio_duration / (min_silence_points + 1)
        
        # Determine how aggressive we need to be with threshold adjustment
        if audio_duration > 300:  # Long audio (>5min)
            adjustment_steps = [5, 8, 12, 15]
        elif audio_duration > 120:  # Medium length (2-5min)
            adjustment_steps = [4, 7, 10, 14]
        else:  # Short audio (<2min)
            adjustment_steps = [3, 6, 9, 12]
        
        # Try increasingly lenient thresholds until we get enough points
        # or exhaust our adjustment options
        for step in adjustment_steps:
            new_threshold = silence_threshold + step
            print(f"Few silence points detected ({len(silence_data)}). Trying with more lenient threshold: {new_threshold:.2f}dB")
            silence_data = _detect_silence_points(input_file, new_threshold, silence_duration)
            
            if len(silence_data) >= min_silence_points:
                print(f"Successfully found {len(silence_data)} silence points with threshold {new_threshold:.2f}dB")
                break
    
    # If we still don't have enough silence points, try with shorter silence duration
    if len(silence_data) < min_silence_points and silence_duration > 0.1:
        shorter_duration = max(0.05, silence_duration / 2)
        print(f"Still insufficient silence points. Trying with shorter silence duration: {shorter_duration:.2f}s")
        silence_data = _detect_silence_points(input_file, silence_threshold + 10, shorter_duration)
    
    print(f"Found {len(silence_data)} silence points")
    return silence_data

def _detect_silence_points(input_file, silence_threshold, silence_duration):
    """
    Helper function to detect silence points with a given threshold.
    Enhanced with noise detection parameters for better accuracy.
    """
    # Add noise parameter for better filtering
    command = [
        "ffmpeg",
        "-i", input_file,
        "-af", f"silencedetect=noise={silence_threshold}dB:d={silence_duration}:mono=true",
        "-f", "null", "-"
    ]
    
    try:
        process = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
        stderr_output = process.stderr.decode()
        
        # Store both silence starts and ends
        silence_data = []
        
        # Process line by line with improved pattern matching
        for line in stderr_output.splitlines():
            # Extract silence end times with improved regex
            end_match = re.search(r"silence_end:\s*([\d\.]+)(?:\s*\|\s*silence_duration:\s*([\d\.]+))?", line)
            if end_match:
                silence_end = float(end_match.group(1))
                duration = float(end_match.group(2)) if end_match.group(2) else None
                silence_data.append({"type": "end", "time": silence_end, "duration": duration})
            
            # Extract silence_start times
            start_match = re.search(r"silence_start:\s*([\d\.]+)", line)
            if start_match:
                silence_start = float(start_match.group(1))
                silence_data.append({"type": "start", "time": silence_start})
        
        # Sort by time and filter out any duplicate points (within 10ms)
        silence_data.sort(key=lambda x: x["time"])
        if len(silence_data) > 1:
            filtered_data = [silence_data[0]]
            for i in range(1, len(silence_data)):
                if abs(silence_data[i]["time"] - filtered_data[-1]["time"]) > 0.01:
                    filtered_data.append(silence_data[i])
            silence_data = filtered_data
            
        return silence_data
    except subprocess.SubprocessError as e:
        print(f"Error analyzing audio: {e}")
        return []

def create_segments_from_silence(silence_data, total_duration, min_segment_length=5, max_segment_length=15):
    """
    Create segment boundaries from silence detection data without extracting audio.
    """
    print("Creating segments from silence data...")
    
    # Initialize with start of file
    segment_boundaries = [0.0]
    
    # Track the last added boundary and current position
    last_boundary = 0.0
    current_position = 0.0
    
    # Track if we're in a silence or not
    in_silence = False
    silence_start = 0.0
    
    # Process silence data to create meaningful segments
    for point in silence_data:
        time = point["time"]
        point_type = point["type"]
        
        # Skip very short segments
        if time - last_boundary < 0.5:
            continue
            
        if point_type == "start":
            in_silence = True
            silence_start = time
        elif point_type == "end":
            in_silence = False
            
            # Only add a boundary if this creates a segment of reasonable length
            if time - last_boundary >= min_segment_length:
                # If segment would be too long, add intermediate boundaries
                if time - last_boundary > max_segment_length:
                    # Calculate exactly how many segments we need to stay under max_segment_length
                    segment_duration = time - last_boundary
                    # Use ceiling division to ensure we have enough segments
                    steps = int(np.ceil(segment_duration / max_segment_length)) - 1
                    step_size = segment_duration / (steps + 1)
                    
                    for i in range(1, steps + 1):
                        boundary = last_boundary + (i * step_size)
                        segment_boundaries.append(boundary)
                        # Don't update last_boundary here to maintain reference to segment start
                
                # Add the silence end as a boundary
                segment_boundaries.append(time)
                last_boundary = time
    
    # Ensure the end of the file is included
    if segment_boundaries[-1] < total_duration:
        remaining = total_duration - segment_boundaries[-1]
        
        # Check if remaining audio is less than max_segment_length - if so, don't chop it further
        if remaining <= max_segment_length:
            # Just add the end of file as the final boundary without further segmentation
            segment_boundaries.append(total_duration)
        else:
            # If remaining segment is too long, add intermediate boundaries
            # Calculate exactly how many segments we need
            steps = int(np.ceil(remaining / max_segment_length)) - 1
            if steps > 0:
                step_size = remaining / (steps + 1)
                
                for i in range(1, steps + 1):
                    boundary = segment_boundaries[-1] + (i * step_size)
                    segment_boundaries.append(boundary)
            
            # Add final boundary
            segment_boundaries.append(total_duration)
    
    # Remove any duplicates and ensure boundaries are ordered
    segment_boundaries = sorted(list(set(segment_boundaries)))
    
    # Additional step: Remove boundaries that are too close to each other (minimum 0.5s gap)
    min_gap = 0.5
    filtered_boundaries = [segment_boundaries[0]]  # Always keep the first boundary
    for i in range(1, len(segment_boundaries)):
        if segment_boundaries[i] - filtered_boundaries[-1] >= min_gap:
            filtered_boundaries.append(segment_boundaries[i])
    
    segment_boundaries = filtered_boundaries
    print(f"Created {len(segment_boundaries)-1} segments from silence data")
    return segment_boundaries

def transcribe_audio(input_file, model, processor, device, segment_boundaries,
                    sample_rate=16000, normalize=False, 
                    batch_size=8, normalize_text=True):
    """
    Transcribes audio using segment boundaries for timing information.
    Uses the full audio file for transcription without extracting segments.
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
                    batch_input_features = processor(
                        batch_audio,
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    ).input_features.to(device)
                    
                    predicted_ids = model.generate(
                        batch_input_features,
                        max_length=448,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    transcripts = processor.batch_decode(
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

def format_timestamp(seconds):
    """Format seconds into HH:MM:SS.mmm format"""
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def format_srt_timestamp(seconds):
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def load_whisper_model(model_name, hf_token=None, device=None):
    """
    Load Whisper model and processor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if hf_token:
        login(token=hf_token)
    
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        print(f"Successfully loaded model from {model_name}")
        return model, processor, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, device

def main():
    """Main function to process audio files"""
    # Start timing the process
    script_start_time = time.time()
    
    # Hard-coded configuration values (formerly command-line arguments)
    input_file = "/content/trimmed_audio_segment_040.wav"  # Change this to your actual input file path
    model_name = "loko99/whisper_small_kannada" # Change this to your actual model name
    hf_token = "YOUR_HUGGINGFACE_TOKEN"  # Replace with your actual Hugging Face token
    min_segment = 5
    max_segment_length = 15  # Maximum segment length in seconds
    silence_duration = 0.2  # Minimum silence duration to consider as a segment boundary
    sample_rate = 16000
    batch_size = 8
    normalize = True  # Set to True if you want to normalize audio volume
    no_text_normalize = False  # Set to True to skip Kannada text normalization
    output = "/content/transcription.srt"  # Changed extension to .srt

    # Check if ffmpeg is installed
    if not check_dependencies():
        sys.exit(1)

    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)

    # Step 1: Load model first so it's ready for transcription
    print("Loading model...")
    model, processor, device = load_whisper_model(model_name, hf_token)
    if model is None or processor is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Step 2: Get audio duration
    total_duration = get_audio_duration(input_file)
    if total_duration <= 0:
        print(f"Error: Could not determine duration of {input_file}")
        sys.exit(1)
    
    print(f"Processing file: {input_file} (duration: {total_duration:.2f} seconds)")
    
    # Step 3 & 4: Analyze audio for silence points in a single pass with adaptive threshold
    silence_data = analyze_full_audio_for_silence(
        input_file, 
        silence_duration=silence_duration,
        adaptive=True,
        min_silence_points=6
    )
    
    # Step 5: Create segment boundaries from silence data
    segment_boundaries = create_segments_from_silence(
        silence_data, 
        total_duration,
        min_segment_length=min_segment,
        max_segment_length=max_segment_length
    )
    
    # Step 6: Transcribe using segment boundaries
    print("Starting transcription...")
    results = transcribe_audio(
        input_file,
        model, processor, device,
        segment_boundaries,
        sample_rate=sample_rate,
        normalize=normalize,
        batch_size=batch_size,
        normalize_text=not no_text_normalize
    )
    
    if not results:
        print("No transcription results. Exiting.")
        sys.exit(1)

    # Step 7: Output the results to file
    output_lines = []
    print("\nPreparing output file...")
    for i, result in enumerate(results):
        # Format in SRT format for the output file
        srt_entry = [
            str(i + 1),
            f"{format_srt_timestamp(result['start'])} --> {format_srt_timestamp(result['end'])}",
            result["transcript"],
            ""  # Empty line between entries
        ]
        output_lines.extend(srt_entry)
    
    # Save to file if specified
    if output:
        # Ensure output has .srt extension
        if not output.lower().endswith('.srt'):
            output = os.path.splitext(output)[0] + '.srt'
            
        try:
            with open(output, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
            print(f"\nSubtitles saved in SRT format to {output}")
        except Exception as e:
            print(f"Error saving output: {str(e)}")
    
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

if __name__ == "__main__":
    main()