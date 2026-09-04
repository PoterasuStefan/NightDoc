import os
import sys
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.transforms as T

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, "sound_radar_model.onnx")
from classifier import SoundRadarClassifier, CLASSES

classifier = SoundRadarClassifier()

def test_audio_file(file_path):
    print(f"\n==================================================")
    print(f"Test Fișier: {os.path.basename(file_path)}")
    
    data, sr = sf.read(file_path, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.t()
        
    res = classifier.classify_waveform(waveform, sr)
    print(f"RMS Energy: {res['rms_energy']:.4f} | Alert Status: {res['alert_status']} | Is Alert: {res['is_alert']}")
    print(f"Predicție: [{res['predicted_class'].upper()}] - {res['title']} (Încredere: {res['confidence']:.1f}%)")

    print("\nProbabilități Multi-Label Sigmoid (independente):")
    for cls, p in res['probabilities'].items():
        bar = "█" * int(p / 5)
        print(f"  - {cls:15s}: {p:5.1f}% | {bar}")

if __name__ == "__main__":
    audio_dir_man = os.path.join(BASE_DIR, "Test(junk)")
    bg_dir = os.path.join(BASE_DIR, "Additional_Manual_Training", "Background")
    conv_dir = os.path.join(BASE_DIR, "Additional_Manual_Training", "Conversatie")
    
    # 1. Alerte de urgență / bebeluș
    test_audio_file(os.path.join(audio_dir_man, "Baby Crying Sound Effect.wav"))
    test_audio_file(os.path.join(audio_dir_man, "Alarm Siren Sound Fx.wav"))
    
    # 2. Conversație / Vorbire (Română)
    if os.path.exists(conv_dir):
        test_audio_file(os.path.join(conv_dir, "Episode 1. Greetings in Romanian.wav"))
    
    # 3. Zgomot de fond / Chatter ambiental
    if os.path.exists(bg_dir):
        test_audio_file(os.path.join(bg_dir, "mixkit-restaurant-background-ambience-2502.wav"))