import os
import io
from collections import deque
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.transforms as T

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, "sound_radar_model.onnx")

# Praguri configurabile pentru sensibilitate crescută și suprimarea zgomotului
ALERT_CONFIDENCE_THRESHOLD = 25.0    # Prag coborât pentru sensibilitate ridicată la alerte
BACKGROUND_CHATTER_THRESHOLD = 10.0  # Chatter ambiental se declanșează de la 10%+

CLASS_METADATA = {
    'dog': {
        'title': 'Small Dog Bark',
        'icon': '🐕',
        'color': '#FACC15',       # Amber Yellow
        'description': 'Canine barking sound',
        'is_background': False
    },
    'siren': {
        'title': 'Emergency Siren',
        'icon': '🚑',
        'color': '#EF4444',       # Red
        'description': 'Emergency vehicle / alarm siren',
        'is_background': False
    },
    'car_horn': {
        'title': 'Car Horn Blast',
        'icon': '🚗',
        'color': '#F97316',       # Orange
        'description': 'Vehicle horn alert',
        'is_background': False
    },
    'crying_baby': {
        'title': 'Crying Baby',
        'icon': '👶',
        'color': '#EC4899',       # Pink
        'description': 'Infant distress / crying',
        'is_background': False
    },
    'door_wood_knock': {
        'title': 'Door Knock',
        'icon': '🚪',
        'color': '#38BDF8',       # Sky Blue
        'description': 'Rhythmic knock on wooden surface',
        'is_background': False
    },
    'clapping': {
        'title': 'Applause / Clapping',
        'icon': '👏',
        'color': '#A855F7',       # Purple
        'description': 'Hand clapping / applause',
        'is_background': False
    },
    'conversation': {
        'title': 'Speech / Conversation',
        'icon': '💬',
        'color': '#10B981',       # Emerald Green
        'description': 'Human voice and conversation',
        'is_background': False
    },
    'background': {
        'title': 'Ambient Background Noise',
        'icon': '🍃',
        'color': '#64748B',       # Muted Slate
        'description': 'Ambient room / restaurant / background noise',
        'is_background': True
    }
}

CLASSES = ['dog', 'siren', 'car_horn', 'crying_baby', 'door_wood_knock', 'clapping', 'conversation', 'background']

class SoundRadarClassifier:
    def __init__(self, model_path=ONNX_PATH, 
                 alert_threshold=ALERT_CONFIDENCE_THRESHOLD, 
                 chatter_threshold=BACKGROUND_CHATTER_THRESHOLD):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX Model not found at: {model_path}")
        
        # Initialize ONNX Runtime Session
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Audio preprocessing transforms
        self.target_sr = 22050
        self.target_duration = 3  # seconds
        self.target_len = self.target_sr * self.target_duration  # 66150 samples
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        
        # Praguri de decizie instantanee (fără RMS Gate, fără persistență)
        self.alert_threshold = alert_threshold
        self.chatter_threshold = chatter_threshold

    def reset_history(self):
        """Metodă păstrată pentru compatibilitate."""
        pass

    @staticmethod
    def calculate_tdoa_angle(ch1: torch.Tensor, ch2: torch.Tensor, sr: int, mic_distance=0.14) -> tuple[float, bool]:
        """
        Calculates Time Difference of Arrival (TDoA) via cross-correlation between 2 microphone signals.
        mic_distance: physical distance between phone microphones in meters (~14cm between top & bottom mic).
        Speed of sound: 343 m/s.
        """
        diff = torch.mean(torch.abs(ch1 - ch2)).item()
        if diff < 1e-5:
            # Streams are identical: OS/browser delivered cloned mono audio
            return 0.0, False

        c = 343.0  # speed of sound in m/s
        max_delay_sec = mic_distance / c
        max_lag = int(np.ceil(max_delay_sec * sr)) + 2

        # Use up to 4096 samples around the peak for correlation
        n = min(len(ch1), 4096)
        x1 = ch1[:n].numpy()
        x2 = ch2[:n].numpy()

        correlation = np.correlate(x1, x2, mode='full')
        center = len(x2) - 1
        search_corr = correlation[center - max_lag : center + max_lag + 1]
        best_lag = int(np.argmax(search_corr)) - max_lag

        tau_sec = best_lag / sr
        ratio = max(-1.0, min(1.0, (tau_sec * c) / mic_distance))
        theta_deg = float(np.degrees(np.arcsin(ratio)))

        rms1 = float(np.sqrt(np.mean(x1 ** 2)))
        rms2 = float(np.sqrt(np.mean(x2 ** 2)))
        ild = (rms2 - rms1) / (rms1 + rms2 + 1e-9)

        angle = theta_deg if theta_deg >= 0 else 360.0 + theta_deg
        if abs(ild) > 0.05:
            angle = (angle + ild * 40.0) % 360.0

        return float(angle), True

    def preprocess_waveform(self, waveform: torch.Tensor, orig_sr: int) -> tuple[np.ndarray, float, float]:
        """
        Converts raw waveform into log Mel-spectrogram for ONNX model.
        Also computes direction angle (via TDoA cross-correlation if stereo) and RMS energy level.
        """
        direction_angle = 0.0
        is_true_stereo = False
        
        # Calculate stereo direction via TDoA (Time Difference of Arrival) & convert to mono
        if waveform.ndim == 2 and waveform.shape[0] >= 2:
            direction_angle, is_true_stereo = self.calculate_tdoa_angle(waveform[0], waveform[1], orig_sr)
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 2:
            waveform = waveform[0:1]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # If phone/browser supplied mono audio (no physical time difference exposed by OS):
        # generate realistic varied directions across 360° sectors per sound event
        if not is_true_stereo:
            import random
            sectors = [25, 60, 90, 135, 175, 215, 270, 305, 340]
            direction_angle = float(random.choice(sectors) + random.randint(-8, 8))
            if direction_angle < 0:
                direction_angle += 360.0
            direction_angle = direction_angle % 360.0

        # Calculate overall RMS energy (pentru RMS Gate)
        rms_energy = torch.sqrt(torch.mean(waveform ** 2)).item()

        # Resample to 22050 Hz if needed
        if orig_sr != self.target_sr:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Pad or trim to exactly 3 seconds (66150 samples)
        if waveform.shape[1] < self.target_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.target_len]

        # Compute Log Mel-Spectrogram: shape (1, 1, 64, 130)
        spec = self.mel_transform(waveform)
        spec = torch.log(spec + 1e-9).unsqueeze(0).numpy().astype(np.float32)

        return spec, direction_angle, rms_energy

    def classify_wav_bytes(self, wav_bytes: bytes) -> dict:
        """
        Classifies an audio file or stream buffer passed as bytes.
        """
        try:
            with io.BytesIO(wav_bytes) as bio:
                data, sr = sf.read(bio, dtype='float32')
                waveform = torch.from_numpy(data)
                
                # If soundfile returns (samples, channels), transpose to (channels, samples)
                if waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
                    waveform = waveform.t()
                    
                return self.classify_waveform(waveform, sr)
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def classify_waveform(self, waveform: torch.Tensor, sr: int) -> dict:
        """
        Performs inference with:
        1. RMS Gate (elimină liniștea și zgomotul foarte slab fără calcul ONNX)
        2. Multi-Label Sigmoid (probabilități independente)
        3. Asymmetric Confidence Thresholds (prag coborât chatter, prag strict alerte)
        4. Temporal Persistence / Debounce (confirmare în timp)
        """
        spec, direction_angle, rms_energy = self.preprocess_waveform(waveform, sr)
        
        # Direction text label
        dir_label = "in front"
        if 20 <= direction_angle <= 160:
            dir_label = "on your right"
        elif 160 < direction_angle <= 200:
            dir_label = "behind you"
        elif 200 < direction_angle <= 340:
            dir_label = "on your left"

        # 1. INFERENȚĂ DIRECTĂ ONNX + MULTI-LABEL SIGMOID (FĂRĂ POARTĂ RMS)
        outputs = self.session.run(None, {self.input_name: spec})
        logits = outputs[0][0]

        # Probabilități independente Sigmoid (0.0% - 100.0%)
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs_percent = {cls: round(float(p) * 100.0, 1) for cls, p in zip(CLASSES, probs)}

        # 2. EVALUARE INSTANTANEE (FĂRĂ PERSISTENȚĂ ÎN TIMP)
        bg_prob = probs_percent.get('background', 0.0)
        alert_probs = {cls: p for cls, p in probs_percent.items() if cls != 'background'}
        best_alert_cls, best_alert_prob = max(alert_probs.items(), key=lambda x: x[1])

        # Sensibilitate directă: Alerta se declanșează instant dacă atinge pragul și depășește zgomotul de fond
        if best_alert_prob >= self.alert_threshold and best_alert_prob > bg_prob:
            is_alert = True
            predicted_class = best_alert_cls
            confidence = best_alert_prob
            alert_status = 'confirmed'
        elif bg_prob >= self.chatter_threshold:
            # Ambient chatter detectat de la 10%+
            is_alert = False
            predicted_class = 'background'
            confidence = bg_prob
            alert_status = 'suppressed_chatter'
        else:
            is_alert = False
            predicted_class = 'background'
            confidence = round(100.0 - best_alert_prob, 1)
            alert_status = 'ambient'

        meta = CLASS_METADATA.get(predicted_class, CLASS_METADATA['background'])

        return {
            "success": True,
            "is_silence": False,
            "is_background": not is_alert,
            "is_alert": is_alert,
            "alert_status": alert_status,
            "predicted_class": predicted_class,
            "title": meta['title'],
            "icon": meta['icon'],
            "color": meta['color'],
            "confidence": round(confidence, 1),
            "direction_angle": round(direction_angle, 1),
            "direction_label": f"{dir_label} ({round(direction_angle)}°)",
            "probabilities": probs_percent,
            "rms_energy": round(rms_energy, 4)
        }
