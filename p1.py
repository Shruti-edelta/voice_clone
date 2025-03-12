from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
from scipy.io.wavfile import write
import os
import warnings
import logging
import torch
 
torch.manual_seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
 
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
 
logging.getLogger("TTS").setLevel(logging.ERROR)
 
config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="bark/", eval=True)
model.to("cpu")
 
voice_dirs = "bark_voices/"
speaker_id = "speaker"
 
if not os.path.exists(voice_dirs):
    raise FileNotFoundError(f"Voice directory '{voice_dirs}' not found!")
 
text = " [laughs] my good name is darshan dhameliya."
 
output_dic = model.synthesize(text, config, speaker_id=speaker_id, voice_dirs=voice_dirs, temperature=0.1, tqdm_disable=True)
 
write("output_07.wav", 24000, output_dic["wav"])
 
 