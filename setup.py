from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whisper_transcriber",
    version="0.1.8",
    author="Ranjan Shettigar",
    author_email="theloko.dev@gmail.com",
    description="A library for transcribing audio files using Whisper models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/COILDOrg/whisper-transcriber",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "tqdm>=4.64.0",
        "librosa>=0.9.2",
        "transformers>=4.26.0",
        "huggingface_hub>=0.12.0",
        "regex>=2022.10.31",
        "pathlib>=1.0.1",
    ],
    entry_points={
        "console_scripts": [
            "whisper-transcribe=whisper_transcriber.cli:main",
        ],
    },
)
