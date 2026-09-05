import os
import onnxruntime as ort
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(BASE_DIR, "sound_radar_model.onnx")

# 1. Incarcam modelul ONNX (folosind motorul Microsoft ONNX Runtime)
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

CLASSES = ['coughing', 'breathing', 'snoring', 'sneezing', 'crying_baby', 'conversation', 'background']

# 2. Simulam o spectrograma de test
dummy_spectrogram = np.random.randn(1, 1, 64, 130).astype(np.float32)

# 3. Rulam inferenta pe CPU/NPU
outputs = session.run(None, {input_name: dummy_spectrogram})
probabilities = 1.0 / (1.0 + np.exp(-outputs[0][0]))
predicted_idx = int(np.argmax(probabilities))
predicted_class = CLASSES[predicted_idx]

print(f"RespiSense ONNX Runtime Test Reusit!")
print(f"Biomarker detectat local: {predicted_class} (Incredere: {round(probabilities[predicted_idx]*100, 1)}%)")
