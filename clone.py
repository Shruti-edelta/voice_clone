from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark  #like transformer
from scipy.io.wavfile import write,read
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

rate, _ = read('bark_voices/speaker/sample.wav')
print("Sample rate of reference:", rate)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
config=BarkConfig()
model=Bark.init_from_config(config)
model.load_checkpoint(config,checkpoint_dir="bark/")
model.to("cpu")

# text="[laughter] my name is shruti"
text="This is a beginning of the history. If you want to hear more, please continue."

output_dic=model.synthesize(text,config,speaker_id="speaker",voice_dirs="bark_voices")

write("output.wav",rate,output_dic["wav"])

