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

    def save_last_seconds(self, output_file):
        if not self.frame_buffer:
            return
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        height, width, _ = self.frame_buffer[0].shape
        out = cv2.VideoWriter(output_file, fourcc, self.fps, (width, height))
        
        # Create a copy of the deque to avoid mutation during iteration
        frames_copy = list(self.frame_buffer)
        for frame in frames_copy:
            out.write(frame)
        out.release()


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please speak...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."

def get_gemini_vision(input):
    genai.configure(api_key=st.secrets["SecretKey"]["GOOGLE_API_KEY"])
    video_file_name = "screen_recording.avi"
    st.write("Uploading file...")
    video_file = genai.upload_file(path=video_file_name)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content([input, video_file])
    genai.delete_file(video_file.name)

    return response

st.set_page_config(page_title="Gemini Companion", page_icon=":robot:", layout="wide")
st.title("Gemini Companion")

if st.button("Start recording"):
    recorder = ScreenRecorder(duration=15, fps=10)
    recorder.start_recording()
    time.sleep(20)
    recorder.save_last_seconds("screen_recording.avi")
    recorder.stop_recording()
if st.button("Speech and get response"):
    user_prompt = recognize_speech()
    st.write(f"User said: {user_prompt}")
    response = get_gemini_vision(user_prompt)
    st.write(response.text)
