import os
import numpy as np
import sounddevice as sd
import wavio
import tempfile
from groq import Groq
from datetime import datetime

class AudioRecorder:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.is_recording = False
        self.sample_rate = 44100
        self.audio_data = []

    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.audio_data = []  # Clear previous audio data
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback)
        self.stream.start()
        print("Recording started...")

    def audio_callback(self, indata, frames, time, status):
        """Callback function to store audio data."""
        if status:
            print(status)
        self.audio_data.append(indata.copy())

    def stop_recording(self):
        """Stop recording audio and return the audio file path."""
        self.stream.stop()
        self.stream.close()
        self.is_recording = False
        audio_file = self.save_audio()
        print("Recording stopped.")
        return audio_file

    def save_audio(self):
        """Save the recorded audio to a file."""
        audio_array = np.concatenate(self.audio_data, axis=0)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wavio.write(temp_file.name, audio_array, self.sample_rate, sampwidth=2)
        return temp_file.name

    def transcribe_audio(self, audio_file):
        """Transcribe the audio file using the Groq API."""
        # Implement transcription logic here
        # This is a placeholder for the actual transcription logic
        return "Transcription of the audio."

class AudioNoteManager:
    def __init__(self, api_key):
        self.recorder = AudioRecorder(api_key)
        self.notes = {}
        
    def start_recording(self):
        """Start a new audio recording"""
        self.recorder.start_recording()
        
    def stop_and_save(self, patient_id):
        """Stop recording and save transcription to patient's notes"""
        audio_file = self.recorder.stop_recording()
        transcription = self.recorder.transcribe_audio(audio_file)
        
        if patient_id not in self.notes:
            self.notes[patient_id] = []
            
        self.notes[patient_id].append({
            "timestamp": datetime.now().isoformat(),
            "transcription": transcription
        })
        return transcription
    
    def analyze_note(self, text):
        """Analyze clinical notes using Groq's Mixtral model"""
        try:
            response = self.recorder.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{
                    "role": "user",
                    "content": f"Analyze this clinical note and provide key insights:\n{text}"
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis error: {str(e)}"