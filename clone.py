from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark  #like transformer
from scipy.io.wavfile import write,read
import os

rate, _ = read('harvard.wav')
print("Sample rate of reference:", rate)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
config=BarkConfig()
model=Bark.init_from_config(config)
model.load_checkpoint(config,checkpoint_dir="bark/",eval=True)
model.to("cpu")

voice_dirs = "bark_voices/"
print("Available voices:", os.listdir(voice_dirs))
speaker_id = "speaker" 
# text="[laughter] my name is shruti"
text="This is a beginning of the history. If you want to hear more, please continue."

output_dic=model.synthesize(text,config,speaker_id=speaker_id,voice_dirs=voice_dirs,temperature=0.0)

write("output.wav",rate,output_dic["wav"])

