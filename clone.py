from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark  #like transformer
from scipy.io.wavfile import write,read
import os
import warnings

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
warnings.filterwarnings("ignore")
rate,data = read('bark_voices/speaker/LJ037-0171.wav')

# np.set_printoptions(threshold=sys.maxsize) # print all array not display...
# print("Sample rate of reference:",data)

# SEMANTIC_VOCAB_SIZE N_COARSE_CODEBOOKS
# config=BarkConfig(num_chars=100,SAMPLE_RATE =rate,
#                   SEMANTIC_VOCAB_SIZE=10000,
#                   USE_SMALLER_MODELS=True)   #SEMANTIC_VOCAB_SIZE =10000 12000 TEXT_SOS_TOKEN=10,050 7000

config=BarkConfig(USE_SMALLER_MODELS=True)
model=Bark.init_from_config(config)
model.load_checkpoint(config,checkpoint_dir="bark/")
model.eval()
model.to("cpu")

text="This is a beginning of the history."

output_dic=model.synthesize(text,config,speaker_id="speaker",voice_dirs="bark_voices/",temperature=0.5,max_gen_duration_s=30)
print("out_dic: ",output_dic["wav"].shape,output_dic)
write("output.wav",24000,output_dic["wav"]) #22050,230005


# text="This is the size of the vocabulary that the model uses to represent semantic concepts"
# text="If you find that the output is inconsistent across different runs even with the same settings, you can try saving the generated speech.If you want to hear more, please continue.If you find that the output is inconsistent."
# text="[laughter] my name is shruti"
# text="Mohandas Karamchand Gandhi was an Indian lawyer anti-colonial nationalist, and political ethicist who employed nonviolent. resistance to lead the successful campaign for India's independence from British rule. " 
# text2="He inspired movements for civil rights and freedom across the world. The honorific Mahātmā, first applied to him in South Africa in 1914, is now used throughout the world."
# text="The training helps system integrators, resellers, and distributors equip their people with the skills to deploy and maintain AudioCodes networking technology. first applied to him in South Africa in 1914. "

 