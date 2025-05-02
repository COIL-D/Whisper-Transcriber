"""Audio processing functions for the Whisper Transcriber package."""

import subprocess
import re
import numpy as np
import os
from .utils import validate_file_path


def get_audio_duration(input_file):
    """
    Use ffprobe to obtain the duration (in seconds) of the input audio file.
    
    Args:
        input_file (str): Path to the audio file
        
    Returns:
        float: Duration of the audio file in seconds
    """
    # Validate file path before using in subprocess
    if not os.path.exists(input_file) or not validate_file_path(input_file):
        print(f"Error: Invalid or unsafe file path: {input_file}")
        return 0.0
        
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              check=True, shell=False, text=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.SubprocessError, ValueError) as e:
        print(f"Error getting audio duration: {e}")
        return 0.0


def analyze_audio_levels(input_file):
    """
    Analyze audio file to get mean and peak volume levels for better silence detection threshold.
    
    Args:
        input_file (str): Path to the audio file
        
    Returns:
        float: Dynamically calculated silence threshold based on audio characteristics
    """
    # Validate file path before using in subprocess
    if not os.path.exists(input_file) or not validate_file_path(input_file):
        print(f"Error: Invalid or unsafe file path: {input_file}")
        return -30.0  # Default if validation fails
        
    command = [
        "ffmpeg",
        "-i", input_file,
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    
    try:
        process = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, 
                               check=True, shell=False, text=True)
        stderr_output = process.stderr
        
        # Look for mean_volume and max_volume in the output
        mean_match = re.search(r"mean_volume:\s*([-\d\.]+)\s*dB", stderr_output)
        max_match = re.search(r"max_volume:\s*([-\d\.]+)\s*dB", stderr_output)
        
        mean_volume = float(mean_match.group(1)) if mean_match else -25
        max_volume = float(max_match.group(1)) if max_match else -5
        
        # Calculate dynamic ratio based on the difference between max and mean
        dynamic_range = max_volume - mean_volume
        
        # Adjust threshold more intelligently based on audio characteristics
        if dynamic_range > 40:  # High dynamic range (music or mixed content)
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


def _detect_silence_points(input_file, silence_threshold, silence_duration):
    """
    Helper function to detect silence points with a given threshold.
    
    Args:
        input_file (str): Path to the audio file
        silence_threshold (float): Silence threshold in dB
        silence_duration (float): Minimum silence duration in seconds
        
    Returns:
        list: List of dictionaries containing silence points
    """
    # Validate file path
    if not os.path.exists(input_file) or not validate_file_path(input_file):
        print(f"Error: Invalid or unsafe file path: {input_file}")
        return []
        
    command = [
        "ffmpeg",
        "-i", input_file,
        "-af", f"silencedetect=noise={silence_threshold}dB:d={silence_duration}:mono=true",
        "-f", "null", "-"
    ]
    
    try:
        process = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, 
                               check=True, shell=False, text=True)
        stderr_output = process.stderr
        
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


def analyze_full_audio_for_silence(input_file, silence_threshold=-30, silence_duration=0.2, adaptive=True, min_silence_points=6):
    """
    Analyze the entire audio file for silence points with enhanced adaptive threshold algorithm.
    
    Args:
        input_file (str): Path to the audio file
        silence_threshold (float): Initial silence threshold in dB
        silence_duration (float): Minimum silence duration in seconds
        adaptive (bool): Whether to adaptively determine the silence threshold
        min_silence_points (int): Minimum number of silence points needed for good segmentation
    
    Returns:
        list: List of silence points
    """
    # Validate file path
    if not os.path.exists(input_file) or not validate_file_path(input_file):
        print(f"Error: Invalid or unsafe file path: {input_file}")
        return []
        
    if adaptive:
        silence_threshold = analyze_audio_levels(input_file)
        print(f"Using adaptive silence threshold: {silence_threshold:.2f}dB")
    
    print("Performing full audio silence analysis...")
    
    # First attempt with initial threshold
    silence_data = _detect_silence_points(input_file, silence_threshold, silence_duration)
    
    # If not enough silence points found, try with progressively more lenient thresholds
    if len(silence_data) < min_silence_points:
        # Calculate the number of seconds per silence point we'd expect
        audio_duration = get_audio_duration(input_file)
        
        # Determine how aggressive we need to be with threshold adjustment
        if audio_duration > 300:  # Long audio (>5min)
            adjustment_steps = [5, 8, 12, 15]
        elif audio_duration > 120:  # Medium length (2-5min)
            adjustment_steps = [4, 7, 10, 14]
        else:  # Short audio (<2min)
            adjustment_steps = [3, 6, 9, 12]
        
        # Try increasingly lenient thresholds
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


def create_segments_from_silence(silence_data, total_duration, min_segment_length=5, max_segment_length=15):
    """
    Create segment boundaries from silence detection data without extracting audio.
    
    Args:
        silence_data (list): List of silence points
        total_duration (float): Total duration of the audio file
        min_segment_length (float): Minimum segment length in seconds
        max_segment_length (float): Maximum segment length in seconds
        
    Returns:
        list: List of segment boundaries
    """
    print("Creating segments from silence data...")
    
    # Initialize with start of file
    segment_boundaries = [0.0]
    
    # Track the last added boundary and current position
    last_boundary = 0.0
    
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
                
                # Add the silence end as a boundary
                segment_boundaries.append(time)
                last_boundary = time
    
    # Ensure the end of the file is included
    if segment_boundaries[-1] < total_duration:
        remaining = total_duration - segment_boundaries[-1]
        
        # Check if remaining audio is less than max_segment_length
        if remaining <= max_segment_length:
            segment_boundaries.append(total_duration)
        else:
            # If remaining segment is too long, add intermediate boundaries
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
