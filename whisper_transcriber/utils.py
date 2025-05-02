"""Utility functions for the Whisper Transcriber package."""

import subprocess
import re
import os
import shlex


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


def validate_file_path(file_path):
    """
    Validate that a file path is safe to use.
    
    Args:
        file_path (str): Path to validate
        
    Returns:
        bool: True if the path is safe, False otherwise
    """
    # Check if path contains shell control characters
    if any(c in file_path for c in ';&|`$(){}[]!#'):
        return False
    
    # Ensure the path is not trying to traverse directories
    normalized = os.path.normpath(file_path)
    if normalized.startswith('..') or '..' in normalized.split(os.path.sep):
        return False
        
    return True


def check_dependencies():
    """
    Check if ffmpeg and ffprobe are installed and available.
    
    Returns:
        bool: True if dependencies are available, False otherwise
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                      check=False, shell=False)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                      check=False, shell=False)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        print("Error: ffmpeg and/or ffprobe not found. Please install ffmpeg.")
        return False


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
