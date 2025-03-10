import speech_recognition as sr
import sounddevice as sd
import numpy as np
from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark  #like transformer
from scipy.io.wavfile import write,read
import os
import re
import subprocess

model="deepseek-r1"

def remove_sample_file():
    file_path="bark_voices/speaker/my_voice.npz"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"The file {file_path} has been deleted.")
    else:
        print(f"The file {file_path} does not exist.")

def listen_for_input():  
    while True:
        text=speech_reco()
        if text:
            if text.upper() == "BREAK":
                print("Ending the conversation...")
                break
            else:
                text=auto_answer(text)
                voice_cloning(text)

# Use the microphone as the audio source
def speech_reco():
    global audio
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("wait for noise adjusting...")
        recognizer.adjust_for_ambient_noise(source)  #voice handle with threshod value(noise adjust)
        print("Please say something: ")
        audio = recognizer.listen(source)
        save_audio(audio)  # listen to the audio
        try:
            g_text = recognizer.recognize_google(audio)     #translate audio to text(google translator)
            print(g_text)
            return g_text
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said")
            return None
        except sr.RequestError as e:
            print("Error; {0}".format(e))
            return None

def save_audio(audio):
    with open("bark_voices/speaker/my_voice.wav", "wb") as file:
        file.write(audio.get_wav_data())
    

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
def voice_cloning(text):    
    print(text)
    remove_sample_file()
    config=BarkConfig()
    model=Bark.init_from_config(config)
    model.load_checkpoint(config,checkpoint_dir="bark/",eval=True)
    model.to("cpu")
    # output_dic=model.generate_audio(text)
    output_dic=model.synthesize(text,config,speaker_id="speaker",voice_dirs="bark_voices/",temperature=0.75,speed=0.9)
    print("ganarated_Audio: ",output_dic)
    write("output.wav",33000,output_dic["wav"])
    speak("output.wav")

def auto_answer(text):
    r=subprocess.run(['ollama','run',model,text],capture_output=True,text=True)
    cleaned_output = re.sub(r'<.*?>','',r.stdout)
    print(cleaned_output)
    return cleaned_output.strip()

def speak(file_path):
    rate, data = read(file_path)
    sd.play(data, rate)

listen_for_input()
# text="hello,how r u.what are you doing"
# text=auto_answer(text)
# voice_cloning(text)

