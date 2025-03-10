import torch
from TTS.api import TTS
import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Hello world!", speaker_wav="bark_voices/speaker/my_voice.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="bark_voices/speaker/my_voice.wav", language="en", file_path="output.wav")
