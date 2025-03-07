from transformers import AutoProcessor,BarkModel
import scipy
import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
processor=AutoProcessor.from_pretrained("suno/bark")
model=BarkModel.from_pretrained("suno/bark")
model.to("cpu")

def audio(text,preset,output):
    inputs = processor(text,voice_preset=preset)
    for k,v in inputs.items():
        inputs[k]=v.to("cpu")
    audio_arr=model.generate(**inputs)
    audio_arr=audio_arr.cpu().numpy().squeeze()
    sample_rate=model.generation_config.sample_rate
    scipy.io.wavfile.write(output,rate=sample_rate,data=audio_arr)

text="hello,how are you"
# text="你好吗"
text="एक समय की बात है, एक छोटे से प्यारे बिल्ली के बच्चे का नाम था मिलो।"
audio(text,preset="v2/hi_speaker_1",output="output.wav")


