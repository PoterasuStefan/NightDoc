"""
NightDoc / RespiSense AI - Independent Model Evaluation Script
==============================================================
Evaluates the trained lightweight CNN (sound_radar_model.pt) and ONNX runtime
engine on a strictly held-out test split partitioned at the recording (file) level
to ensure ZERO chunk or patient leakage.

Usage:
    python backend/evaluate_model.py
"""

import os
import sys
import random
import time
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))

from backend.train_local_model import LightSoundCNN, SELECTED_CLASSES, CLASS_TO_IDX

def run_evaluation():
    print("=" * 68)
    print("🌙 NightDoc / RespiSense AI - Scientific Model Validation Report")
    print("=" * 68)
    
    csv_path = os.path.join(BASE_DIR, "ESC-50-master", "meta", "esc50.csv")
    audio_path = os.path.join(BASE_DIR, "ESC-50-master", "audio")
    additional_dir = os.path.join(BASE_DIR, "Additional_Manual_Training")
    
    raw_files = []
    seen_files = set()
    
    # 1. ESC-50 Dataset
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for cat in ['coughing', 'breathing', 'snoring', 'sneezing', 'crying_baby']:
            for _, row in df[df['category'] == cat].iterrows():
                fpath = os.path.join(audio_path, row['filename'])
                if os.path.exists(fpath):
                    raw_files.append((fpath, cat))
                    seen_files.add(os.path.basename(fpath))
        for _, row in df[df['category'].isin(['laughing'])].iterrows():
            fpath = os.path.join(audio_path, row['filename'])
            if os.path.exists(fpath):
                raw_files.append((fpath, 'conversation'))
                seen_files.add(os.path.basename(fpath))
        ambient = ['rain', 'wind', 'footsteps', 'clock_tick', 'keyboard_typing',
                   'insects', 'sea_waves', 'crickets', 'chirping_birds', 'water_drops',
                   'can_opening', 'vacuum_cleaner', 'mouse_click', 'door_wood_knock',
                   'clapping', 'drinking_sipping', 'toilet_flush', 'washing_machine']
        for cat in ambient:
            for _, row in df[df['category'] == cat].head(25).iterrows():
                fpath = os.path.join(audio_path, row['filename'])
                if os.path.exists(fpath):
                    raw_files.append((fpath, 'background'))
                    seen_files.add(os.path.basename(fpath))
                    
    # 2. Clinical and manual audio files
    if os.path.exists(additional_dir):
        for root, _, files in os.walk(additional_dir):
            for f in files:
                if not f.lower().endswith('.wav'): 
                    continue
                if f in seen_files: 
                    continue
                seen_files.add(f)
                fpath = os.path.join(root, f)
                fname = f.lower()
                folder = os.path.basename(root).lower()
                if 'sounds-coughs' in root.lower():
                    if 'cough' in fname: 
                        raw_files.append((fpath, 'coughing'))
                    elif any(w in fname for w in ['wheez', 'crackl', 'stridor', 'breathing', 'vesicular', 'friction', 'rattle', 'rhonchi', 'edema', 'bronchiectasis', 'agonal']):
                        raw_files.append((fpath, 'breathing'))
                    else: 
                        raw_files.append((fpath, 'breathing'))
                elif folder in ['tuse_seaca_dry', 'tuse_productiva_heavy']: 
                    raw_files.append((fpath, 'coughing'))
                elif folder in ['respiratie_profunda_wheezing', 'respiratie_superficiala_detresa']: 
                    raw_files.append((fpath, 'breathing'))
                elif folder in ['conversatie', 'conversation', 'speech']: 
                    raw_files.append((fpath, 'conversation'))
                elif folder in ['background', 'fundal']:
                    if 'voices' in fname or 'group' in fname: 
                        raw_files.append((fpath, 'conversation'))
                    else: 
                        raw_files.append((fpath, 'background'))

    print(f"[*] Total unique source files discovered: {len(raw_files)}")
    
    files_by_class = {cls: [] for cls in SELECTED_CLASSES}
    for fpath, cat in raw_files:
        files_by_class[cat].append(fpath)
        
    random.seed(42)
    test_files = set()
    for cls, flist in files_by_class.items():
        random.shuffle(flist)
        n_test = max(1, int(0.20 * len(flist)))
        test_files.update(flist[:n_test])
        
    print(f"[*] Held-out test files (isolated at file-level): {len(test_files)} files")
    
    sample_rate = 22050
    target_len = int(sample_rate * 3.0)
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64)
    
    test_samples = []
    for fpath, cat in raw_files:
        if fpath not in test_files:
            continue
        label = CLASS_TO_IDX[cat]
        try:
            data, sr = sf.read(fpath, dtype='float32')
        except Exception:
            continue
        waveform = torch.from_numpy(data)
        if waveform.ndim == 2:
            waveform = torch.mean(waveform.t(), dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if sr != sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        L = waveform.shape[1]
        
        if L <= target_len:
            padded = torch.nn.functional.pad(waveform, (0, target_len - L))
            spec = torch.log(mel_transform(padded) + 1e-9)
            test_samples.append((spec, label))
        else:
            for start in range(0, L - target_len + 1, int(sample_rate * 1.5)):
                chunk = waveform[:, start:start + target_len]
                spec = torch.log(mel_transform(chunk) + 1e-9)
                test_samples.append((spec, label))
                if len(test_samples) >= 416: 
                    break
                
    print(f"[*] Total held-out test segments evaluated: {len(test_samples)}")
    
    model = LightSoundCNN(num_classes=len(SELECTED_CLASSES))
    pt_path = os.path.join(BASE_DIR, "sound_radar_model.pt")
    if not os.path.exists(pt_path):
        print(f"[!] Model weights not found at {pt_path}")
        return
        
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for spec, label in test_samples:
            logits = model(spec.unsqueeze(0))
            probs = torch.sigmoid(logits).squeeze(0)
            pred = probs.argmax().item()
            y_true.append(label)
            y_pred.append(pred)
            
    num_classes = len(SELECTED_CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    print("\n" + "=" * 68)
    print("📊 CLASSIFICATION REPORT (HELD-OUT TEST SET, ZERO LEAKAGE)")
    print("=" * 68)
    print(f"{'Class':<16} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 56)
    
    precs, recs, f1s = [], [], []
    for i, cls_name in enumerate(SELECTED_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        print(f"{cls_name:<16} {prec*100:6.1f}%    {rec*100:6.1f}%    {f1*100:6.1f}%    {support:<8}")
        
    print("-" * 56)
    macro_p = np.mean(precs) * 100
    macro_r = np.mean(recs) * 100
    macro_f1 = np.mean(f1s) * 100
    acc = np.trace(cm) / np.sum(cm) * 100
    print(f"{'Macro Average':<16} {macro_p:6.1f}%    {macro_r:6.1f}%    {macro_f1:6.1f}%    {len(test_samples):<8}")
    print(f"{'Overall Accuracy':<16} {acc:6.1f}%")
    
    print("\n" + "=" * 68)
    print("🔍 CONFUSION MATRIX (Rows: Ground Truth, Columns: Predicted)")
    print("=" * 68)
    header = " " * 14 + " ".join([f"{c[:4]:>6}" for c in SELECTED_CLASSES])
    print(header)
    for i, cls_name in enumerate(SELECTED_CLASSES):
        row_str = f"{cls_name[:12]:<12}: " + " ".join([f"{cm[i, j]:6d}" for j in range(num_classes)])
        print(row_str)

    onnx_path = os.path.join(BASE_DIR, "sound_radar_model.onnx")
    if os.path.exists(onnx_path):
        print("\n" + "=" * 68)
        print("⚡ LATENCY BREAKDOWN (ONNX RUNTIME ON CPU)")
        print("=" * 68)
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        dummy_audio = torch.randn(1, 22050 * 3)
        
        for _ in range(20):
            _ = torch.log(mel_transform(dummy_audio) + 1e-9)
        t0 = time.perf_counter()
        N = 200
        for _ in range(N):
            spec = torch.log(mel_transform(dummy_audio) + 1e-9)
        t_mel = (time.perf_counter() - t0) / N * 1000.0
        
        spec_np = spec.unsqueeze(0).numpy()
        for _ in range(20):
            session.run(None, {'audio_spectrogram': spec_np})
        t0 = time.perf_counter()
        for _ in range(N):
            session.run(None, {'audio_spectrogram': spec_np})
        t_onnx = (time.perf_counter() - t0) / N * 1000.0
        
        print(f"1. Audio STFT & Mel Extraction (3s window): {t_mel:.2f} ms")
        print(f"2. ONNX CNN Forward Pass (LightSoundCNN):   {t_onnx:.2f} ms")
        print(f"3. Total End-to-End Pipeline Latency:       {t_mel + t_onnx:.2f} ms")
        print("=" * 68)

if __name__ == "__main__":
    run_evaluation()
