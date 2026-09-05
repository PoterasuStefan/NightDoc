import os
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import soundfile as sf
import torch
import pandas as pd
from classifier import RespiSenseClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classifier = RespiSenseClassifier()

csv_path = os.path.join(BASE_DIR, "ESC-50-master", "meta", "esc50.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    cough_files = df[df['category'] == 'coughing']['filename'].tolist()
    snore_files = df[df['category'] == 'snoring']['filename'].tolist()
    breathe_files = df[df['category'] == 'breathing']['filename'].tolist()
    
    print("=== TEST 1: TUSE REALA (ESC-50) ===")
    if cough_files:
        sample_path = os.path.join(BASE_DIR, "ESC-50-master", "audio", cough_files[0])
        data, sr = sf.read(sample_path, dtype='float32')
        res = classifier.classify_waveform(torch.from_numpy(data), sr)
        print(f"Fisier: {cough_files[0]}")
        print(f"Detectat: {res['title']} | Incredere: {res['confidence']}% | Timp Inferenta ONNX: {res['inference_ms']}ms")
        print(f"Standarde: LOINC {res['loinc']} | SNOMED-CT {res['snomed']}")
        print(f"HL7 FHIR Observation ID: {res['fhir_id']}")

    print("\n=== TEST 2: SFORAIT / APNEE (ESC-50) ===")
    if snore_files:
        sample_path = os.path.join(BASE_DIR, "ESC-50-master", "audio", snore_files[0])
        data, sr = sf.read(sample_path, dtype='float32')
        res = classifier.classify_waveform(torch.from_numpy(data), sr)
        print(f"Fisier: {snore_files[0]}")
        print(f"Detectat: {res['title']} | Incredere: {res['confidence']}% | Timp Inferenta ONNX: {res['inference_ms']}ms")
        print(f"Standarde: LOINC {res['loinc']} | SNOMED-CT {res['snomed']}")
        print(f"HL7 FHIR Observation ID: {res['fhir_id']}")

    print("\n=== TEST 3: RESPIRATIE (ESC-50) ===")
    if breathe_files:
        sample_path = os.path.join(BASE_DIR, "ESC-50-master", "audio", breathe_files[0])
        data, sr = sf.read(sample_path, dtype='float32')
        res = classifier.classify_waveform(torch.from_numpy(data), sr)
        print(f"Fisier: {breathe_files[0]}")
        print(f"Detectat: {res['title']} | Incredere: {res['confidence']}% | Timp Inferenta ONNX: {res['inference_ms']}ms")
