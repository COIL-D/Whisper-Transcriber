from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whisper_transcriber",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for transcribing audio files using Whisper models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisper_transcriber",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "tqdm",
        "librosa",
        "transformers",
        "huggingface_hub",
        "regex",
    ],
    entry_points={
        "console_scripts": [
            "whisper-transcribe=whisper_transcriber.cli:main",
        ],
    },
)
