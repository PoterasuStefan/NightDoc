import os
import onnxruntime as ort
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(BASE_DIR, "sound_radar_model.onnx")

# 1. Încărcăm modelul ONNX (folosind motorul Microsoft ONNX Runtime)
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

CLASSES = ['dog', 'siren', 'car_horn', 'crying_baby', 'door_wood_knock', 'clapping', 'conversation', 'background']

# 2. Simulăm o spectrogramă de test (sau folosești torchaudio să încarci un fișier .wav real)
dummy_spectrogram = np.random.randn(1, 1, 64, 130).astype(np.float32)

# 3. Rulăm inferența (durează ~5-10ms pe CPU)
outputs = session.run(None, {input_name: dummy_spectrogram})
probabilities = outputs[0][0]
predicted_class = CLASSES[np.argmax(probabilities)]

print(f"Sunet detectat local de ONNX Runtime: {predicted_class}")