"""Audio processing functions for the Whisper Transcriber package."""

import subprocess
import re
import numpy as np
import os
import librosa
import concurrent.futures
import tempfile
import time
from tqdm import tqdm
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


def extract_audio_segment(input_file, start_time, duration, sample_rate=16000, normalize=False):
    """
    Extract a specific segment of audio from a file without loading the entire file.
    
    Args:
        input_file (str): Path to the audio file
        start_time (float): Start time of the segment in seconds
        duration (float): Duration of the segment in seconds
        sample_rate (int): Sample rate for the audio
        normalize (bool): Whether to normalize the audio
    
    Returns:
        numpy.ndarray: Audio segment data
    """
    # Validate file path
    if not os.path.exists(input_file) or not validate_file_path(input_file):
        print(f"Error: Invalid or unsafe file path: {input_file}")
        return None
        
    try:
        # Use librosa's offset and duration parameters to load only the segment
        audio_segment, _ = librosa.load(
            input_file,
            sr=sample_rate,
            mono=True,
            offset=start_time,
            duration=duration
        )
        
        # Apply normalization if requested
        if normalize and len(audio_segment) > 0:
            audio_segment = librosa.util.normalize(audio_segment)
            
        return audio_segment
    except Exception as e:
        print(f"Error loading audio segment: {e}")
        return None


def _process_audio_chunk(chunk_data):
    """
    Process a chunk of audio for silence detection in parallel.
    
    Args:
        chunk_data (dict): Data for the chunk to process
        
    Returns:
        list: Silence points detected in the chunk
    """
    input_file = chunk_data["input_file"]
    start_time = chunk_data["start_time"]
    duration = chunk_data["duration"]
    silence_threshold = chunk_data["silence_threshold"]
    silence_duration = chunk_data["silence_duration"]
    
    # Create a temporary file for the audio chunk
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
        
    try:
        # Extract chunk to temporary file
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(duration),
            "-c", "copy",
            temp_path
        ]
        
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, shell=False)
        
        # Run silence detection on the chunk
        chunk_silence = _detect_silence_points(temp_path, silence_threshold, silence_duration)
        
        # Adjust the times to account for the chunk's starting position
        for point in chunk_silence:
            point["time"] += start_time
            
        return chunk_silence
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        return []
    finally:
        # Remove the temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def analyze_full_audio_for_silence(input_file, silence_threshold=-30, silence_duration=0.2, 
                                adaptive=True, min_silence_points=6, parallel_jobs=None, 
                                chunk_size=600):
    """
    Analyze the entire audio file for silence points with parallel processing.
    
    Args:
        input_file (str): Path to the audio file
        silence_threshold (float): Initial silence threshold in dB
        silence_duration (float): Minimum silence duration in seconds
        adaptive (bool): Whether to adaptively determine the silence threshold
        min_silence_points (int): Minimum number of silence points needed for good segmentation
        parallel_jobs (int): Number of parallel jobs to use (None for auto)
        chunk_size (int): Size of audio chunks in seconds for parallel processing
    
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
    
    # Get audio duration
    total_duration = get_audio_duration(input_file)
    if total_duration <= 0:
        print(f"Error: Could not determine duration of {input_file}")
        return []
    
    # For short files, just do direct analysis
    if total_duration <= chunk_size * 2:
        return _detect_silence_points(input_file, silence_threshold, silence_duration)
    
    # Determine number of parallel jobs if not specified
    if parallel_jobs is None:
        parallel_jobs = min(os.cpu_count() or 2, 8)
    
    # Split audio into chunks for parallel processing
    chunks = []
    for chunk_start in range(0, int(total_duration), chunk_size):
        # Make chunks overlap by 2x silence_duration to avoid missing silence at chunk boundaries
        overlap = 2 * silence_duration
        adjusted_start = max(0, chunk_start - overlap) if chunk_start > 0 else 0
        
        # Last chunk should go to the end of audio
        if chunk_start + chunk_size >= total_duration:
            chunk_duration = total_duration - adjusted_start
        else:
            chunk_duration = chunk_size + (chunk_start - adjusted_start) + overlap
        
        chunks.append({
            "input_file": input_file,
            "start_time": adjusted_start,
            "duration": chunk_duration,
            "silence_threshold": silence_threshold,
            "silence_duration": silence_duration
        })
    
    # Process chunks in parallel
    silence_data = []
    with tqdm(total=len(chunks), desc="Analyzing audio chunks", unit="chunk") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            # Submit all tasks and create a future-to-chunk mapping
            future_to_chunk = {
                executor.submit(_process_audio_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    silence_data.extend(chunk_result)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
    
    # Sort by time and filter out duplicates
    silence_data.sort(key=lambda x: x["time"])
    
    # Remove duplicate points that might occur due to chunk overlaps (within 10ms)
    if silence_data:
        filtered_data = [silence_data[0]]
        for i in range(1, len(silence_data)):
            if abs(silence_data[i]["time"] - filtered_data[-1]["time"]) > 0.01:
                filtered_data.append(silence_data[i])
        silence_data = filtered_data
    
    print(f"Found {len(silence_data)} silence points")
    
    # If we didn't find enough silence points, try with more lenient thresholds
    if len(silence_data) < min_silence_points:
        # Calculate the number of seconds per silence point we'd expect
        # Determine how aggressive we need to be with threshold adjustment
        if total_duration > 300:  # Long audio (>5min)
            adjustment_steps = [5, 8, 12, 15]
        elif total_duration > 120:  # Medium length (2-5min)
            adjustment_steps = [4, 7, 10, 14]
        else:  # Short audio (<2min)
            adjustment_steps = [3, 6, 9, 12]
        
        # Try increasingly lenient thresholds
        for step in adjustment_steps:
            new_threshold = silence_threshold + step
            print(f"Few silence points detected ({len(silence_data)}). Trying with more lenient threshold: {new_threshold:.2f}dB")
            
            # Process in parallel again with new threshold
            for chunk in chunks:
                chunk["silence_threshold"] = new_threshold
                
            new_silence_data = []
            with tqdm(total=len(chunks), desc="Re-analyzing with new threshold", unit="chunk") as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
                    future_to_chunk = {
                        executor.submit(_process_audio_chunk, chunk): i 
                        for i, chunk in enumerate(chunks)
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            chunk_result = future.result()
                            new_silence_data.extend(chunk_result)
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error processing chunk {chunk_idx} with new threshold: {e}")
            
            # Sort and filter duplicates
            new_silence_data.sort(key=lambda x: x["time"])
            if new_silence_data:
                filtered_data = [new_silence_data[0]]
                for i in range(1, len(new_silence_data)):
                    if abs(new_silence_data[i]["time"] - filtered_data[-1]["time"]) > 0.01:
                        filtered_data.append(new_silence_data[i])
                new_silence_data = filtered_data
            
            silence_data = new_silence_data
            print(f"Found {len(silence_data)} silence points with threshold {new_threshold:.2f}dB")
            
            if len(silence_data) >= min_silence_points:
                break
    
    # If we still don't have enough silence points, try with shorter silence duration
    if len(silence_data) < min_silence_points and silence_duration > 0.1:
        shorter_duration = max(0.05, silence_duration / 2)
        print(f"Still insufficient silence points. Trying with shorter silence duration: {shorter_duration:.2f}s")
        
        # Update all chunks with shorter silence duration and more lenient threshold
        for chunk in chunks:
            chunk["silence_duration"] = shorter_duration
            chunk["silence_threshold"] = silence_threshold + 10
        
        new_silence_data = []
        with tqdm(total=len(chunks), desc="Re-analyzing with shorter duration", unit="chunk") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
                future_to_chunk = {
                    executor.submit(_process_audio_chunk, chunk): i 
                    for i, chunk in enumerate(chunks)
                }
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        new_silence_data.extend(chunk_result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing chunk {chunk_idx} with shorter duration: {e}")
        
        # Sort and filter duplicates
        if new_silence_data:
            new_silence_data.sort(key=lambda x: x["time"])
            filtered_data = [new_silence_data[0]]
            for i in range(1, len(new_silence_data)):
                if abs(new_silence_data[i]["time"] - filtered_data[-1]["time"]) > 0.01:
                    filtered_data.append(new_silence_data[i])
            silence_data = filtered_data
    
    print(f"Final silence point count: {len(silence_data)}")
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
    
    segment_boundaries = [0.0]
    last_boundary = 0.0
    # Keep track of added boundaries to prevent duplicates from different logic paths
    processed_times = {0.0}

    # Process silence data, prioritizing 'silence_end' as natural segment breaks
    for point in silence_data:
        time = point["time"]
        point_type = point["type"]

        if point_type == "end":
            current_segment_start = last_boundary
            current_segment_end = time

            # Skip if this point is too close to the last boundary, creating a tiny segment
            if current_segment_end - current_segment_start < 0.5 and current_segment_end < total_duration - 0.5:
                continue

            if current_segment_end - current_segment_start >= min_segment_length:
                # This segment [current_segment_start, current_segment_end] is valid.
                # Now, check if it's too long and needs splitting.
                
                temp_chunk_start = current_segment_start
                
                while current_segment_end - temp_chunk_start > max_segment_length:
                    ideal_split_time = temp_chunk_start + max_segment_length
                    best_silence_split_point = -1

                    candidate_silence_splits = []
                    for s_info in silence_data: # Check all silence points
                        s_time = s_info['time']
                        # Conditions for a valid split point:
                        # 1. Within the current chunk being split (temp_chunk_start, current_segment_end)
                        # 2. Ensures new segment [temp_chunk_start, s_time] is >= min_segment_length
                        # 3. Ensures remaining part [s_time, current_segment_end] is >= min_segment_length
                        if (temp_chunk_start < s_time < current_segment_end and
                            s_time - temp_chunk_start >= min_segment_length and
                            current_segment_end - s_time >= min_segment_length):
                            candidate_silence_splits.append(s_time)
                    
                    if candidate_silence_splits:
                        # Prefer silence points <= ideal_split_time, take the one closest (largest)
                        choices_le_ideal = [s for s in candidate_silence_splits if s <= ideal_split_time]
                        if choices_le_ideal:
                            best_silence_split_point = max(choices_le_ideal)
                        else:
                            # No points <= ideal_split_time, so take the smallest point > ideal_split_time
                            choices_gt_ideal = [s for s in candidate_silence_splits if s > ideal_split_time]
                            if choices_gt_ideal:
                                best_silence_split_point = min(choices_gt_ideal)
                    
                    if best_silence_split_point != -1:
                        # Found a good silence point to split at
                        if best_silence_split_point not in processed_times:
                            segment_boundaries.append(best_silence_split_point)
                            processed_times.add(best_silence_split_point)
                        temp_chunk_start = best_silence_split_point
                    else:
                        # No suitable silence point, force a split at max_segment_length
                        forced_split_time = temp_chunk_start + max_segment_length
                        # Ensure this forced split doesn't make the remaining part too small
                        if forced_split_time < current_segment_end - min_segment_length:
                            if forced_split_time not in processed_times:
                                segment_boundaries.append(forced_split_time)
                                processed_times.add(forced_split_time)
                            temp_chunk_start = forced_split_time
                        else:
                            # Cannot split further without violating min_segment_length for the remainder.
                            break 
                
                # Add the original 'time' (current_segment_end) from the silence_end event as a boundary.
                if current_segment_end not in processed_times:
                    segment_boundaries.append(current_segment_end)
                    processed_times.add(current_segment_end)
                last_boundary = current_segment_end
            # If segment (last_boundary, time) was < min_segment_length, 'time' is not added yet,
            # and last_boundary remains unchanged.
    
    # Ensure the end of the file is included and handle splitting if the last segment is too long
    if last_boundary < total_duration:
        remaining_segment_start = last_boundary
        remaining_segment_end = total_duration

        while remaining_segment_end - remaining_segment_start > max_segment_length:
            ideal_split_time = remaining_segment_start + max_segment_length
            best_silence_split_point = -1

            candidate_silence_splits = []
            for s_info in silence_data:
                s_time = s_info['time']
                if (remaining_segment_start < s_time < remaining_segment_end and
                    s_time - remaining_segment_start >= min_segment_length and
                    remaining_segment_end - s_time >= min_segment_length):
                    candidate_silence_splits.append(s_time)

            if candidate_silence_splits:
                choices_le_ideal = [s for s in candidate_silence_splits if s <= ideal_split_time]
                if choices_le_ideal:
                    best_silence_split_point = max(choices_le_ideal)
                else:
                    choices_gt_ideal = [s for s in candidate_silence_splits if s > ideal_split_time]
                    if choices_gt_ideal:
                        best_silence_split_point = min(choices_gt_ideal)
            
            if best_silence_split_point != -1:
                if best_silence_split_point not in processed_times:
                    segment_boundaries.append(best_silence_split_point)
                    processed_times.add(best_silence_split_point)
                remaining_segment_start = best_silence_split_point
            else:
                forced_split_time = remaining_segment_start + max_segment_length
                if forced_split_time < remaining_segment_end - min_segment_length:
                    if forced_split_time not in processed_times:
                        segment_boundaries.append(forced_split_time)
                        processed_times.add(forced_split_time)
                    remaining_segment_start = forced_split_time
                else:
                    break
        
        # Add the final total_duration boundary
        if total_duration not in processed_times:
            segment_boundaries.append(total_duration)
            # processed_times.add(total_duration) # Not strictly needed to add here as it's the end

    # Remove duplicates and ensure boundaries are ordered
    segment_boundaries = sorted(list(set(segment_boundaries)))
    
    # Additional step: Remove boundaries that are too close to each other (minimum 0.5s gap)
    # This helps prevent extremely short segments that might be artifacts.
    min_gap = 0.5
    if not segment_boundaries: # Should not happen if total_duration > 0
        return []
        
    filtered_boundaries = [segment_boundaries[0]]
    if len(segment_boundaries) > 1:
        for i in range(1, len(segment_boundaries)):
            # Add boundary if it's distinct and maintains min_gap
            if segment_boundaries[i] - filtered_boundaries[-1] >= min_gap:
                filtered_boundaries.append(segment_boundaries[i])
            # If current boundary is total_duration and it was skipped due to min_gap,
            # replace the last added boundary with total_duration if total_duration is greater.
            # This ensures total_duration is always the true end if it creates a very small last segment.
            elif i == len(segment_boundaries) - 1 and segment_boundaries[i] > filtered_boundaries[-1]:
                 # If the true last point (total_duration) was too close to the previous one
                 # ensure the list ends with total_duration.
                 filtered_boundaries[-1] = segment_boundaries[i] 

    # Ensure the list is not empty and contains at least 0.0 and total_duration if total_duration > 0
    if not filtered_boundaries and total_duration > 0:
        return [0.0, total_duration]
    if filtered_boundaries and filtered_boundaries[-1] < total_duration and total_duration > 0:
         if total_duration - filtered_boundaries[-1] < min_gap:
             filtered_boundaries[-1] = total_duration # Adjust last boundary to be total_duration
         elif total_duration not in filtered_boundaries:
              filtered_boundaries.append(total_duration)


    segment_boundaries = sorted(list(set(filtered_boundaries))) # Final sort and unique after adjustments

    print(f"Created {len(segment_boundaries)-1} segments from silence data")
    return segment_boundaries
