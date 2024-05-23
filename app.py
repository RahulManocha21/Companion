import cv2
import numpy as np
import mss
import time
import speech_recognition as sr
from setuptools import distutils
import google.generativeai as genai
import streamlit as st
from collections import deque
from threading import Thread
import pyttsx3
import wave


class ScreenRecorder:
    def __init__(self, duration=15, fps=10):
        self.duration = duration
        self.fps = fps
        self.frame_buffer = deque(maxlen=self.duration * self.fps)
        self.running = False

    def start_recording(self):
        self.running = True
        self.record_thread = Thread(target=self._record_screen)
        self.record_thread.start()

    def stop_recording(self):
        self.running = False
        self.record_thread.join()

    def _record_screen(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while self.running:
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self.frame_buffer.append(img)
                time.sleep(1 / self.fps)

    def save_last_seconds(self):
        if not self.frame_buffer:
            return
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        height, width, _ = self.frame_buffer[0].shape
        out = cv2.VideoWriter("screen_recording.avi", fourcc, self.fps, (width, height))
        
        # Create a copy of the deque to avoid mutation during iteration
        frames_copy = list(self.frame_buffer)
        for frame in frames_copy:
            out.write(frame)
        out.release()


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # st.write("Listening for command...")
        audio = recognizer.listen(source)
        # Save the audio data as a WAV file
        audio_data = audio.get_wav_data()
        with wave.open("audio.wav", "wb") as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(audio.sample_width)
            wf.setframerate(audio.sample_rate)
            wf.writeframes(audio_data)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry"

def speak_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)    
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()


def get_gemini_vision():
    genai.configure(api_key=st.secrets["SecretKey"]["GOOGLE_API_KEY"])
    video_file_name = "screen_recording.avi"
    audio_file_name = "audio.wav"
    prompt = """You are good observer, study the audio and video inputs and give the suggestions 
    what else can user do better in whatever he is doing in the video"""
    # st.write("Please wait, I am uploading files and let gemini work from here...")
    speak_text("Please wait, I am uploading files and let gemini work from here...")
    video_file = genai.upload_file(path=video_file_name)
    audio_file = genai.upload_file(path=audio_file_name)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content([prompt, audio_file, video_file])
    genai.delete_file(video_file.name)
    genai.delete_file(audio_file.name)

    return response

def main():
    st.set_page_config(page_title="Gemini Companion", page_icon=":robot:", layout="wide")
    st.title("Gemini Companion")
    recorder = ScreenRecorder(duration=15, fps=10)
    recorder.start_recording()
    place, comment = st.columns(2)
    with place:
        placeholder = st.empty()
    while True:
        try:
            
            placeholder.image("img/listen.gif")
            user_prompt = recognize_speech()
            if user_prompt and user_prompt.lower() != "sorry":
                if user_prompt.lower() == "stop" or user_prompt.lower() == "exit":
                    speak_text("See you Later. Good Bye")
                    break
                else:
                    speak_text("Okay, Got it")
                    placeholder.image("img/think.gif")
                    comment.write(f"User said: {user_prompt}")
                    recorder.save_last_seconds()
                    response = get_gemini_vision()
                    comment.write(response.text)
                    placeholder.image("img/giphy.gif")
                    speak_text(response.text)
                    placeholder.empty()
        except Exception as e:
            st.error(f"Error occured as {e}")
            speak_text(f"Error occured as {e}")



if __name__ == "__main__":
    main()
