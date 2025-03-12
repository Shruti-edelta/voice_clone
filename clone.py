from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark  #like transformer
from scipy.io.wavfile import write,read
import os
import warnings

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
warnings.filterwarnings("ignore")
# np.set_printoptions(threshold=sys.maxsize) # print all array not display...
rate,data = read('bark_voices/speaker/LJ037-0171.wav')
print("Sample rate of reference:",data)

config=BarkConfig(num_chars=10000,SAMPLE_RATE =rate,SEMANTIC_VOCAB_SIZE=10000,USE_SMALLER_MODELS=True)   #SEMANTIC_VOCAB_SIZE =10000 12000 TEXT_SOS_TOKEN=10,050 7000
model=Bark.init_from_config(config)
model.load_checkpoint(config,checkpoint_dir="bark/")
model.eval()
model.to("cpu")

def split_text(text, max_length=500):
    """
    Splits the text into chunks of max_length. Adjust max_length according to model's limitation.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    print("words: ",words)
    for word in words:
        print(" ".join(current_chunk + [word]))
        if len(" ".join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            print("===")
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# text="This is a beginning of the history."
# text="This is the size of the vocabulary that the model uses to represent semantic concepts"
# text="If you find that the output is inconsistent across different runs even with the same settings, you can try saving the generated speech.If you want to hear more, please continue.If you find that the output is inconsistent."
# text="[laughter] my name is shruti"
text="Mohandas Karamchand Gandhi was an Indian lawyer anti-colonial nationalist, and political ethicist who employed nonviolent. resistance to lead the successful campaign for India's independence from British rule. He inspired movements for civil rights and freedom across the world. The honorific Mahātmā, first applied to him in South Africa in 1914, is now used throughout the world."

# chunks = split_text(text, max_length=500)
# final_output = []

# for chunk in chunks:
#     print("sdfsd")
#     output_dic = model.synthesize(chunk, config, speaker_id="speaker", voice_dirs="bark_voices/", temperature=0.5)
#     final_output.append(output_dic["wav"])

# # Concatenate the audio chunks
# import numpy as np
# final_audio = np.concatenate(final_output, axis=0)

# write("output.wav", rate, final_audio)
# print("Final audio shape:", final_audio.shape,final_audio)

output_dic=model.synthesize(text,config,speaker_id="speaker",voice_dirs="bark_voices/",temperature=0.5,max_gen_duration_s=30)
write("output.wav",24000,output_dic["wav"]) #22050,23000
print("out_dic: ",output_dic["wav"].shape,output_dic)

 