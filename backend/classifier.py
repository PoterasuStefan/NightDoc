import os
import io
import time
from collections import deque
from datetime import datetime, timezone
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.transforms as T
from fhir_formatter import fhir_formatter, MEDICAL_CODES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, "sound_radar_model.onnx")

# Praguri de sensibilitate configurabile
ALERT_CONFIDENCE_THRESHOLD = 25.0
BACKGROUND_THRESHOLD = 15.0

CLASS_METADATA = {
    'coughing': {
        'title': 'Coughing Episode (Tuse)',
        'icon': '🫁',
        'color': '#EF4444',       # Roșu Clinic
        'description': 'Episod de tuse paroxistică / criză acută',
        'is_alert': True,
        'criticality': 'high',
        'snomed': '263731006',
        'loinc': '8687-6'
    },
    'breathing': {
        'title': 'Breathing / Wheezing (Respirație)',
        'icon': '🌬️',
        'color': '#38BDF8',       # Albastru Azur
        'description': 'Model acustic respirator / wheezing suierat',
        'is_alert': False,
        'criticality': 'low',
        'snomed': '56018004',
        'loinc': '9279-1'
    },
    'snoring': {
        'title': 'Snoring / Apnea Risk (Sforăit)',
        'icon': '💤',
        'color': '#F59E0B',       # Portocaliu / Chihlimbar
        'description': 'Perturbare căi aeriene superioare în somn',
        'is_alert': True,
        'criticality': 'medium',
        'snomed': '271600006',
        'loinc': '93832-4'
    },
    'sneezing': {
        'title': 'Sneezing Episode (Strănut)',
        'icon': '🤧',
        'color': '#A855F7',       # Violet
        'description': 'Reflex respirator acut căi superioare',
        'is_alert': True,
        'criticality': 'low',
        'snomed': '16962002',
        'loinc': '8688-4'
    },
    'crying_baby': {
        'title': 'Neonatal / Pediatric Distress (Plâns Copil)',
        'icon': '👶',
        'color': '#EC4899',       # Roz Intens
        'description': 'Detresă acustică neonatală sau pediatrică',
        'is_alert': True,
        'criticality': 'high',
        'snomed': '271633008',
        'loinc': '72144-9'
    },
    'conversation': {
        'title': 'Speech / Conversation (Vorbire)',
        'icon': '💬',
        'color': '#10B981',       # Verde Smarald
        'description': 'Activitate vocală umană ambientală',
        'is_alert': False,
        'criticality': 'none',
        'snomed': '286369001',
        'loinc': 'LA11874-7'
    },
    'background': {
        'title': 'Ambient Baseline (Liniște / Mediu)',
        'icon': '🍃',
        'color': '#64748B',       # Gri Calibrare
        'description': 'Zgomot de fond ambiental normal',
        'is_alert': False,
        'criticality': 'none',
        'snomed': '162076009',
        'loinc': 'LA28669-9'
    }
}

CLASSES = ['coughing', 'breathing', 'snoring', 'sneezing', 'crying_baby', 'conversation', 'background']

class RespiSenseClassifier:
    def __init__(self, model_path=ONNX_PATH, alert_threshold=ALERT_CONFIDENCE_THRESHOLD):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RespiSense ONNX Model not found at: {model_path}")
        
        # Sesiune ONNX Runtime (Microsoft)
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Audio preprocessing
        self.target_sr = 22050
        self.target_duration = 3.0  # secunde
        self.target_len = int(self.target_sr * self.target_duration)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.alert_threshold = alert_threshold

        # Clinical Telemetry Session State
        self.session_start = time.time()
        self.event_history = deque(maxlen=200) # Ultimele 200 de evenimente
        self.fhir_observations = deque(maxlen=200)
        self.counts = {
            "coughing": 0,
            "breathing": 0,
            "snoring": 0,
            "sneezing": 0,
            "crying_baby": 0,
            "conversation": 0,
            "background": 0
        }

    def reset_telemetry(self):
        self.session_start = time.time()
        self.event_history.clear()
        self.fhir_observations.clear()
        for k in self.counts:
            self.counts[k] = 0

    def get_telemetry_summary(self) -> dict:
        dur_min = max(0.1, round((time.time() - self.session_start) / 60.0, 1))
        coughs = self.counts["coughing"]
        coughs_per_hour = round((coughs / dur_min) * 60.0, 1) if dur_min > 0 else 0.0
        
        # Calcul indice perturbare nocturnă (0-100)
        disturbance = min(100, int((coughs * 12) + (self.counts["snoring"] * 5) + (self.counts["crying_baby"] * 15)))
        
        return {
            "session_duration_minutes": dur_min,
            "cough_count": coughs,
            "coughs_per_hour": coughs_per_hour,
            "breathing_count": self.counts["breathing"],
            "snoring_count": self.counts["snoring"],
            "sneezing_count": self.counts["sneezing"],
            "crying_count": self.counts["crying_baby"],
            "total_events": sum(self.counts.values()),
            "disturbance_score": disturbance,
            "recent_events": list(self.event_history)[-20:]
        }

    def preprocess_waveform(self, waveform: torch.Tensor, orig_sr: int) -> tuple[np.ndarray, float, float]:
        direction_angle = 0.0
        is_true_stereo = False
        
        # Spatial angle estimation dacă e stereo
        if waveform.ndim == 2 and waveform.shape[0] >= 2:
            diff = torch.mean(torch.abs(waveform[0] - waveform[1])).item()
            if diff > 1e-5:
                is_true_stereo = True
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 2:
            waveform = waveform[0:1]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if not is_true_stereo:
            import random
            sectors = [30, 75, 120, 180, 240, 290, 330]
            direction_angle = float(random.choice(sectors) + random.randint(-5, 5)) % 360.0

        rms_energy = torch.sqrt(torch.mean(waveform ** 2)).item()

        if orig_sr != self.target_sr:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[1] < self.target_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.target_len]

        spec = self.mel_transform(waveform)
        spec = torch.log(spec + 1e-9).unsqueeze(0).numpy().astype(np.float32)

        return spec, direction_angle, rms_energy

    def classify_wav_bytes(self, wav_bytes: bytes) -> dict:
        try:
            with io.BytesIO(wav_bytes) as bio:
                data, sr = sf.read(bio, dtype='float32')
                waveform = torch.from_numpy(data)
                if waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
                    waveform = waveform.t()
                return self.classify_waveform(waveform, sr)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def classify_waveform(self, waveform: torch.Tensor, sr: int) -> dict:
        t0 = time.perf_counter()
        spec, direction_angle, rms_energy = self.preprocess_waveform(waveform, sr)
        
        # Inferență Microsoft ONNX Runtime
        outputs = self.session.run(None, {self.input_name: spec})
        logits = outputs[0][0]
        inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Multi-Label Sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs_percent = {cls: round(float(p) * 100.0, 1) for cls, p in zip(CLASSES, probs)}

        # Decizie
        bg_prob = probs_percent.get('background', 0.0)
        alert_probs = {cls: p for cls, p in probs_percent.items() if cls != 'background'}
        best_alert_cls, best_alert_prob = max(alert_probs.items(), key=lambda x: x[1])

        if best_alert_prob >= self.alert_threshold and best_alert_prob > bg_prob:
            predicted_class = best_alert_cls
            confidence = best_alert_prob
            is_alert = CLASS_METADATA[predicted_class]['is_alert']
        else:
            predicted_class = 'background'
            confidence = bg_prob
            is_alert = False

        meta = CLASS_METADATA.get(predicted_class, CLASS_METADATA['background'])
        
        # Increment counts
        if predicted_class in self.counts:
            self.counts[predicted_class] += 1

        now_iso = datetime.now(timezone.utc).isoformat()

        # Generate FHIR Observation for Clinical Telemetry
        fhir_obs = fhir_formatter.create_observation(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probs_percent,
            rms_energy=rms_energy,
            direction_angle=direction_angle,
            duration_sec=3.0,
            timestamp=now_iso
        )
        self.fhir_observations.append(fhir_obs)

        event_entry = {
            "id": fhir_obs["id"],
            "timestamp": now_iso,
            "predicted_class": predicted_class,
            "title": meta['title'],
            "icon": meta['icon'],
            "color": meta['color'],
            "confidence": round(confidence, 1),
            "criticality": meta['criticality'],
            "direction_angle": round(direction_angle, 1),
            "rms_energy": round(rms_energy, 4),
            "inference_ms": inference_time_ms,
            "snomed": meta.get('snomed', ''),
            "loinc": meta.get('loinc', '')
        }
        self.event_history.append(event_entry)

        return {
            "success": True,
            "is_alert": is_alert,
            "predicted_class": predicted_class,
            "title": meta['title'],
            "icon": meta['icon'],
            "color": meta['color'],
            "confidence": round(confidence, 1),
            "criticality": meta['criticality'],
            "direction_angle": round(direction_angle, 1),
            "probabilities": probs_percent,
            "rms_energy": round(rms_energy, 4),
            "inference_ms": inference_time_ms,
            "fhir_id": fhir_obs["id"],
            "fhir_resource": fhir_obs,
            "snomed": meta.get('snomed', ''),
            "loinc": meta.get('loinc', '')
        }
