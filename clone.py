from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark  #like transformer
from scipy.io.wavfile import write,read
import os
import soundfile as sf
import numpy as np
import warnings
import torch
import sys

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)

# data = read('bark_voices/speaker/sample_randomvoice.wav')
# print("Sample rate of reference:",data)

config=BarkConfig(SEMANTIC_RATE_HZ=48.6)
model=Bark.init_from_config(config)
model.load_checkpoint(config,checkpoint_dir="bark/",eval=True)
model.to("cpu")

# text="[laughter] my name is shruti"
# text="This is a beginning of the history. If you want to hear more, please continue."
text="This is a beginning of the history."
text = "My name is Darshan Dhameliya"

output_dic=model.synthesize(text,config,speaker_id="speaker",voice_dirs="bark_voices/",temperature=0.1)

output_dic["wav"] = output_dic["wav"]

print("out_dic: ",output_dic)
write("output.wav",25000,output_dic["wav"])


# from TTS.tts.configs.bark_config import BarkConfig
# from TTS.tts.models.bark import Bark
# from scipy.io.wavfile import write
# import os
# import warnings
# import torch
 
# # torch.manual_seed(42)
# warnings.filterwarnings("ignore")
 
# os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
 
# # logging.getLogger("TTS").setLevel(logging.ERROR)
 
# config = BarkConfig()
# model = Bark.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="bark/", eval=True)
# model.to("cpu")
 
# voice_dirs = "bark_voices/"
# speaker_id = "speaker" 

# text = "My name is Darshan Dhameliya"
 
# output_dic = model.synthesize(text, config, speaker_id=speaker_id, voice_dirs=voice_dirs, temperature=0.1, tqdm_disable=True)

# print("out_dic: ",output_dic)
# write("output.wav", 25000, output_dic["wav"])
