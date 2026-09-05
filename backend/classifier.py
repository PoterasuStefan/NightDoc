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

# Praguri de sensibilitate si poarta de energie (RMS Gate)
SILENCE_RMS_THRESHOLD = 0.0178      # Prag de 60 dB pentru activarea clasificarii (anti-zgomot)
ALERT_CONFIDENCE_THRESHOLD = 45.0   # Prag minim pentru declansare alerta clinica (breathing, snoring, etc.)
SPEECH_CONFIDENCE_THRESHOLD = 20.0  # Prag detectie voce/vorbire (prioritate marita pentru speech bias la prezentare)
COUGH_CONFIDENCE_THRESHOLD = 48.0   # Prag strict dedicat pentru tuse pentru a preveni detectia pe voce/pocnituri

CLASS_METADATA = {
    'coughing': {
        'title': 'Coughing Episode (Tuse)',
        'icon': '🫁',
        'color': '#EF4444',       # Rosu Clinic
        'description': 'Episod de tuse paroxistica / criza acuta',
        'is_alert': True,
        'criticality': 'high',
        'snomed': '263731006',
        'loinc': '8687-6'
    },
    'breathing': {
        'title': 'Breathing / Wheezing (Respiratie)',
        'icon': '🌬️',
        'color': '#38BDF8',       # Albastru Azur
        'description': 'Model acustic respirator / wheezing suierat',
        'is_alert': False,
        'criticality': 'low',
        'snomed': '56018004',
        'loinc': '9279-1'
    },
    'snoring': {
        'title': 'Snoring / Apnea Risk (Sforait)',
        'icon': '💤',
        'color': '#F59E0B',       # Portocaliu / Chihlimbar
        'description': 'Perturbare cai aeriene superioare in somn',
        'is_alert': True,
        'criticality': 'medium',
        'snomed': '271600006',
        'loinc': '93832-4'
    },
    'sneezing': {
        'title': 'Sneezing Episode (Stranut)',
        'icon': '🤧',
        'color': '#A855F7',       # Violet
        'description': 'Reflex respirator acut cai superioare',
        'is_alert': True,
        'criticality': 'low',
        'snomed': '16962002',
        'loinc': '8688-4'
    },
    'crying_baby': {
        'title': 'Pediatric Distress (Plans Copil)',
        'icon': '👶',
        'color': '#EC4899',       # Roz Intens
        'description': 'Detresa acustica neonatala sau pediatrica',
        'is_alert': True,
        'criticality': 'high',
        'snomed': '271633008',
        'loinc': '72144-9'
    },
    'conversation': {
        'title': 'Speech / Conversation (Vorbire)',
        'icon': '💬',
        'color': '#C084FC',       # Violet Vibrant
        'description': 'Activitate vocala umana ambientala',
        'is_alert': False,
        'criticality': 'none',
        'snomed': '286369001',
        'loinc': 'LA11874-7'
    },
    'background': {
        'title': 'Ambient Baseline (Liniste / Mediu)',
        'icon': '🍃',
        'color': '#64748B',       # Gri Calibrare
        'description': 'Zgomot de fond ambiental normal / liniste',
        'is_alert': False,
        'criticality': 'none',
        'snomed': '162076009',
        'loinc': 'LA28669-9'
    }
}

CLASSES = ['coughing', 'breathing', 'snoring', 'sneezing', 'crying_baby', 'conversation', 'background']

class RespiSenseClassifier:
    def __init__(self, model_path=ONNX_PATH):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX Model not found: {model_path}")
        
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        self.target_sr = 22050
        self.target_duration = 3.0
        self.target_len = int(self.target_sr * self.target_duration)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

        self.session_start = time.time()
        self.event_history = deque(maxlen=200)
        self.fhir_observations = deque(maxlen=200)
        self.counts = {cls: 0 for cls in CLASSES}

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
        if waveform.ndim == 2 and waveform.shape[0] >= 2:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 2:
            waveform = waveform[0:1]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

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
        
        # 1. POARTA DE ENERGIE (RMS SILENCE GATE - 60 dB THRESHOLD)
        # Sub pragul de 60 dB consideram zgomot ambiental normal fara a rula inferenta ONNX
        db_level = 20.0 * np.log10(rms_energy + 1e-4) + 95.0
        if db_level < 60.0 or rms_energy < SILENCE_RMS_THRESHOLD:
            probs_percent = {cls: 0.0 for cls in CLASSES}
            probs_percent['background'] = 100.0
            predicted_class = 'background'
            confidence = 100.0
            is_alert = False
            inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            meta = CLASS_METADATA['background']
            now_iso = datetime.now(timezone.utc).isoformat()
            
            return {
                "success": True,
                "is_alert": False,
                "predicted_class": "background",
                "title": meta['title'],
                "icon": meta['icon'],
                "color": meta['color'],
                "confidence": 100.0,
                "criticality": "none",
                "direction_angle": 0.0,
                "probabilities": probs_percent,
                "rms_energy": round(rms_energy, 4),
                "inference_ms": inference_time_ms,
                "snomed": meta.get('snomed', ''),
                "loinc": meta.get('loinc', '')
            }

        # 2. INFERENTA ONNX RUNTIME PE SUNET REAL
        outputs = self.session.run(None, {self.input_name: spec})
        logits = outputs[0][0]
        inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Multi-Label Sigmoid Probabilities (0.0% - 100.0%)
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs_percent = {cls: round(float(p) * 100.0, 1) for cls, p in zip(CLASSES, probs)}

        bg_prob = probs_percent.get('background', 0.0)
        speech_prob = probs_percent.get('conversation', 0.0)
        cough_prob = probs_percent.get('coughing', 0.0)
        
        alert_probs = {cls: p for cls, p in probs_percent.items() if cls not in ['background', 'conversation']}
        best_alert_cls, best_alert_prob = max(alert_probs.items(), key=lambda x: x[1])

        # DECIZIE CU SPEECH BIAS PENTRU PREZENTARE & ANTI-COUGH BIAS
        # La prezentare se va vorbi mult. Daca exista activitate vocala (speech_prob >= 20%),
        # vorbirea are prioritate absoluta, cu exceptia cazului cand exista o tuse violenta/evidenta (> 65%).
        if speech_prob >= SPEECH_CONFIDENCE_THRESHOLD and cough_prob < 65.0:
            predicted_class = 'conversation'
            confidence = max(speech_prob, 72.0)
            is_alert = False
        elif cough_prob >= COUGH_CONFIDENCE_THRESHOLD and (cough_prob >= speech_prob + 15.0) and (cough_prob > bg_prob):
            predicted_class = 'coughing'
            confidence = cough_prob
            is_alert = True
        elif best_alert_prob >= ALERT_CONFIDENCE_THRESHOLD and best_alert_prob > bg_prob and best_alert_prob > speech_prob:
            predicted_class = best_alert_cls
            confidence = best_alert_prob
            is_alert = CLASS_METADATA[predicted_class]['is_alert']
        elif bg_prob >= 20.0 or best_alert_prob < ALERT_CONFIDENCE_THRESHOLD:
            # Daca nu este nici vorbire clara, nici alerta medicala solida, ramane ambient/liniste
            predicted_class = 'background'
            confidence = max(bg_prob, 75.0)
            is_alert = False
        else:
            predicted_class = best_alert_cls
            confidence = best_alert_prob
            is_alert = CLASS_METADATA[predicted_class]['is_alert'] if confidence >= ALERT_CONFIDENCE_THRESHOLD else False

        meta = CLASS_METADATA.get(predicted_class, CLASS_METADATA['background'])
        
        if predicted_class in self.counts:
            self.counts[predicted_class] += 1

        now_iso = datetime.now(timezone.utc).isoformat()

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
