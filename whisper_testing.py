import torch
from transformers import pipeline, WhisperConfig, WhisperModel
import requests
import tempfile
import os
import librosa
import soundfile as sf
import numpy as np

# Check if CUDA is available, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

# Download the audio file locally first to avoid FFmpeg dependency
def download_audio_file(url, filename):
    """Download audio file to local temp directory"""
    response = requests.get(url)
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    return filepath

def load_audio_with_librosa(filepath):
    """Load audio using librosa which doesn't require FFmpeg"""
    # Load audio and resample to 16kHz (Whisper's expected sample rate)
    audio, sr = librosa.load(filepath, sr=16000)
    return audio

try:
    # Create the pipeline with proper device handling
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        dtype=dtype,
        device=device
    )

    # Download audio file from huggingface dataset until we have our own data
    audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    audio_file = download_audio_file(audio_url, "mlk.flac")
    print(f"Downloaded audio file to: {audio_file}")

    # Load audio using librosa instead of letting transformers handle it
    audio_array = load_audio_with_librosa(audio_file)
    print(f"Loaded audio array with shape: {audio_array.shape}")

    # Process the audio array directly (no FFmpeg needed)
    result = asr_pipeline(audio_array)
    print("Transcription result:", result)

    # Clean up the temporary file
    os.remove(audio_file)
    print("Temporary file cleaned up")

except Exception as e:
    print(f"Error occurred: {e}")
    print("This might be due to model loading or other issues.")

# Create model configuration and model instance
configuration = WhisperConfig()
model = WhisperModel(configuration)