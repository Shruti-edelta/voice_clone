# from transformers import AutoProcessor,BarkModel
# import scipy
# import os
# os.environ["SUNO_OFFLOAD_CPU"] = "True"
# os.environ["SUNO_USE_SMALL_MODELS"] = "True"
# processor=AutoProcessor.from_pretrained("suno/bark")
# model=BarkModel.from_pretrained("suno/bark")
# model.to("cpu")

# def audio(text,preset,output):
#     inputs = processor(text,voice_preset=preset)
#     for k,v in inputs.items():
#         inputs[k]=v.to("cpu")
#     audio_arr=model.generate(**inputs)
#     audio_arr=audio_arr.cpu().numpy().squeeze()
#     sample_rate=model.generation_config.sample_rate
#     scipy.io.wavfile.write(output,rate=sample_rate,data=audio_arr)

# text="hello,how are you"
# # text="你好吗"
# text="एक समय की बात है, एक छोटे से प्यारे बिल्ली के बच्चे का नाम था मिलो।"
# text="[laughter] my name is shruti"
# audio(text,preset="v2/en_speaker_9",output="output.wav")


from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  