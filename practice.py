import speech_recognition as sr
import sounddevice as sd
import numpy as np
from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark  #like transformer
from scipy.io.wavfile import write,read
import os
import re
import subprocess
import pyttsx3

model="deepseek-r1"
engine = pyttsx3.init()
engine.setProperty('rate', 100)
voices=engine.getProperty('voices')
speaking=False    
speech_queue = []


def listen_for_input():  
    while True:
        text=speech_reco()
        if text:
            if text.upper() == "BREAK":
                print("Ending the conversation...")
                break
            else:
                response = auto_answer(text)
                speak(response)

# Use the microphone as the audio source
def speech_reco():
    global audio
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("wait for noise adjusting...")
        recognizer.adjust_for_ambient_noise(source)  #voice handle with threshod value(noise)
        print("Please say something: ")
        audio = recognizer.listen(source)  # listen to the audio
        try:
            g_text = recognizer.recognize_google(audio)     #translate audio to text(google translator)
            print(g_text)
            engine.stop()
            speech_queue.clear()
            return g_text
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said")
            return None
        except sr.RequestError as e:
            print("Error; {0}".format(e))
            return None

def auto_answer(text):
    r=subprocess.run(['ollama','run',model,text],capture_output=True,text=True)
    cleaned_output = re.sub(r'<.*?>','',r.stdout)
    speech_queue.append(cleaned_output)
    return cleaned_output.strip()

def speak(cleaned_output):
    print(f"Responding== {cleaned_output}")
    engine.say(cleaned_output)
    engine.runAndWait()

def save_audio():
    with open("bark_voices/speaker/my_voice.wav", "wb") as file:
        file.write(audio.get_wav_data())
    
def vice_cloning():

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
    
    text="hello,how are you"
    output_dic=model.synthesize(text,config,speaker_id=speaker_id,voice_dirs=voice_dirs,temperature=0.0)

    write("output.wav",rate,output_dic["wav"])


