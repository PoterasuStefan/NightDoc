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

# 1. Alegem clasele de interes pentru aplicația noastră de accesibilitate
SELECTED_CLASSES = ['dog', 'siren', 'car_horn', 'crying_baby', 'door_wood_knock', 'clapping', 'conversation', 'background']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}
print(f"Clase selectate ({len(SELECTED_CLASSES)}): {CLASS_TO_IDX}")

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
        
        print(f"Procesăm și segmentăm {len(raw_samples)} fișiere audio de intrare...")
        for file_path, category in raw_samples:
            label = CLASS_TO_IDX[category]
            data, sr = sf.read(file_path, dtype='float32')
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

            # Reglăm pasul (hop) și limita de segmente per fișier
            if category == 'siren':
                hop_sec = 0.75
                max_chunks = 200
            elif category == 'conversation':
                hop_sec = 6.0
                max_chunks = 80
            elif category == 'background':
                if dur > 120.0:
                    hop_sec = 8.0
                elif dur > 40.0:
                    hop_sec = 3.0
                else:
                    hop_sec = 1.5
                max_chunks = 45
            else:
                hop_sec = 1.0
                max_chunks = 200

            hop_samples = int(self.sample_rate * hop_sec)

            # One-hot target vector pentru Multi-Label (Sigmoid)
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
                    
        print(f"Total segmente de 3 secunde generate: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec, target = self.samples[idx]
        return spec, target

# 3. Modelul CNN Ușor (optimizat pentru viteză pe CPU/NPU)
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
    
    # Încărcăm mostrele din ESC-50
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 1. Cele 6 clase de interes țintă
        df_selected = df[df['category'].isin(SELECTED_CLASSES[:6])]
        for _, row in df_selected.iterrows():
            fpath = os.path.join(audio_path, row['filename'])
            if os.path.exists(fpath):
                raw_samples.append((fpath, row['category']))
                
        # 2. Mostre ambientale din ESC-50 mapate la clasa 'background' (ploaie, vânt, pași)
        esc_ambient_categories = ['rain', 'wind', 'footsteps']
        for cat in esc_ambient_categories:
            for f in df[df['category'] == cat]['filename'].head(10):
                fpath = os.path.join(audio_path, f)
                if os.path.exists(fpath):
                    raw_samples.append((fpath, 'background'))
                
    # Încărcăm mostrele din Additional_Manual_Training/<categorie>/*.wav
    if os.path.exists(additional_dir):
        for root, _, files in os.walk(additional_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    category = os.path.basename(root).lower()
                    if category in ['conversatie', 'conversation', 'speech']:
                        category = 'conversation'
                    if category in CLASS_TO_IDX:
                        raw_samples.append((os.path.join(root, f), category))
                        
    print(f"Total fișiere audio unice încărcate: {len(raw_samples)}")
    
    dataset = ChunkedSoundDataset(raw_samples)
    
    # 85% train, 15% validare
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Antrenăm pe: {device}")
    
    model = LightSoundCNN(num_classes=len(SELECTED_CLASSES)).to(device)
    # Multi-Label Loss cu Sigmoid (BCEWithLogitsLoss)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Antrenăm pentru 30 epoci
    print("Începe antrenarea (30 epoci)...")
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

    # 5. Export în formatul ONNX și PyTorch
    model.eval()
    pt_file = os.path.join(BASE_DIR, "sound_radar_model.pt")
    torch.save(model.state_dict(), pt_file)
    print(f"\nPonderile PyTorch au fost salvate în: {pt_file}")

    dummy_input = torch.randn(1, 1, 64, 130).to(device) # Dimensiunea spectrogramei
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
    print(f"[SUCCES] Modelul a fost antrenat și exportat în: {onnx_file}")