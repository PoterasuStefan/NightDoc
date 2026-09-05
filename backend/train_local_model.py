import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 1. Clase medicale si de mediu pentru RespiSense AI (Health & Research)
SELECTED_CLASSES = ['coughing', 'breathing', 'snoring', 'sneezing', 'crying_baby', 'conversation', 'background']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}
print(f"RespiSense AI - Clase respiratorii selectate ({len(SELECTED_CLASSES)}): {CLASS_TO_IDX}")

# 2. Dataset personalizat cu segmentare audio de 3 secunde
class ChunkedSoundDataset(Dataset):
    def __init__(self, raw_samples, sample_rate=22050, duration=3.0):
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.samples = []
        
        print(f"Procesam si segmentam {len(raw_samples)} fisiere audio respiratorii/medicale...")
        for file_path, category in raw_samples:
            if category not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[category]
            try:
                data, sr = sf.read(file_path, dtype='float32')
            except Exception as e:
                print(f"Eroare la citire {file_path}: {e}")
                continue

            waveform = torch.from_numpy(data)
            
            # Gestionare stereo -> mono
            if waveform.ndim == 2:
                waveform = waveform.t()
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
                
            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
                
            L = waveform.shape[1]
            dur = L / self.sample_rate

            if category in ['coughing', 'sneezing']:
                hop_sec = 0.5   # Sunete scurte, importante
                max_chunks = 150
            elif category in ['breathing', 'snoring']:
                hop_sec = 0.75  # Pattern ciclic
                max_chunks = 120
            elif category == 'conversation':
                hop_sec = 4.0
                max_chunks = 80
            elif category == 'background':
                if dur > 120.0:
                    hop_sec = 8.0
                elif dur > 40.0:
                    hop_sec = 3.0
                else:
                    hop_sec = 1.5
                max_chunks = 60
            else:
                hop_sec = 1.0
                max_chunks = 100

            hop_samples = int(self.sample_rate * hop_sec)

            one_hot = torch.zeros(len(SELECTED_CLASSES), dtype=torch.float32)
            one_hot[label] = 1.0

            if L <= self.target_len:
                padded = torch.nn.functional.pad(waveform, (0, self.target_len - L))
                spec = self.mel_transform(padded)
                spec = torch.log(spec + 1e-9)
                self.samples.append((spec, one_hot))
            else:
                count = 0
                for start in range(0, L - self.target_len + 1, hop_samples):
                    chunk = waveform[:, start:start + self.target_len]
                    spec = self.mel_transform(chunk)
                    spec = torch.log(spec + 1e-9)
                    self.samples.append((spec, one_hot))
                    count += 1
                    if count >= max_chunks:
                        break
                    
        print(f"Total segmente respiratorii generate pentru antrenare: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec, target = self.samples[idx]
        return spec, target

# 3. Modelul CNN Usor (optimizat pentru viteza pe CPU/NPU)
class LightSoundCNN(nn.Module):
    def __init__(self, num_classes):
        super(LightSoundCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 4. Bucla de Antrenare
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "ESC-50-master", "meta", "esc50.csv")
    audio_path = os.path.join(BASE_DIR, "ESC-50-master", "audio")
    additional_dir = os.path.join(BASE_DIR, "Additional_Manual_Training")
    
    raw_samples = []
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        medical_categories = ['coughing', 'breathing', 'snoring', 'sneezing', 'crying_baby']
        df_medical = df[df['category'].isin(medical_categories)]
        for _, row in df_medical.iterrows():
            fpath = os.path.join(audio_path, row['filename'])
            if os.path.exists(fpath):
                raw_samples.append((fpath, row['category']))
                
        esc_ambient_categories = ['rain', 'wind', 'footsteps', 'clock_tick', 'keyboard_typing', 'insects']
        for cat in esc_ambient_categories:
            for f in df[df['category'] == cat]['filename'].head(15):
                fpath = os.path.join(audio_path, f)
                if os.path.exists(fpath):
                    raw_samples.append((fpath, 'background'))
                
    if os.path.exists(additional_dir):
        for root, _, files in os.walk(additional_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    category = os.path.basename(root).lower()
                    if category in ['conversatie', 'conversation', 'speech']:
                        category = 'conversation'
                    elif category in ['background', 'fundal']:
                        category = 'background'
                    if category in CLASS_TO_IDX:
                        raw_samples.append((os.path.join(root, f), category))
                        
    print(f"Total fisiere audio unice incarcate: {len(raw_samples)}")
    
    dataset = ChunkedSoundDataset(raw_samples)
    
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Antrenam RespiSense AI pe: {device}")
    
    model = LightSoundCNN(num_classes=len(SELECTED_CLASSES)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Incepe antrenarea modelului respirator (30 epoci)...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 29:
            print(f"Epoca {epoch+1}/30 - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    pt_file = os.path.join(BASE_DIR, "sound_radar_model.pt")
    torch.save(model.state_dict(), pt_file)
    print(f"\nPonderile PyTorch au fost salvate in: {pt_file}")

    dummy_input = torch.randn(1, 1, 64, 130).to(device)
    onnx_file = os.path.join(BASE_DIR, "sound_radar_model.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        input_names=['audio_spectrogram'],
        output_names=['class_probabilities'],
        dynamic_axes={'audio_spectrogram': {0: 'batch_size'}, 'class_probabilities': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"[SUCCES] Modelul RespiSense AI a fost exportat cu succes in: {onnx_file}")
