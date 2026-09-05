<div align="center">

# 🌙 NightDoc / RespiSense AI
### Continuous Bedside Acoustic Biomarker Sentinel & Clinical Research Telemetry
#### *Developed for the Microsoft Hackathon (Health & Research Track)*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Edge_AI-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Microsoft Azure](https://img.shields.io/badge/Microsoft_Foundry-Azure_AI-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)](https://ai.azure.com/)
[![HL7 FHIR R4](https://img.shields.io/badge/HL7_FHIR-R4_Standards-E11D48?style=for-the-badge)](https://hl7.org/fhir/)

<p align="center">
  <b>NightDoc (RespiSense AI)</b> is a privacy-first, non-invasive acoustic biomarker sentinel for clinical research and nocturnal respiratory disease monitoring (Asthma, COPD, Sleep Apnea, Pediatric Distress). It combines ultra-fast on-device <b>Microsoft ONNX Runtime</b> inference with <b>HL7 FHIR R4</b> clinical standardization and <b>Azure AI Foundry</b> synthesis.
</p>

[✨ Key Features](#-key-features) •
[🏛️ System Architecture](#️-system-architecture) •
[🏥 Medical Standards & FHIR](#-medical-standards--fhir) •
[🚀 Quick Start](#-quick-start) •
[📱 Mobile Setup](#-mobile-setup) •
[🔐 Configuration](#-configuration)

---

</div>

## 🌟 Key Features

### 1. 🫁 Real-Time Acoustic Biomarker Detection (Edge ONNX Runtime)
- Runs lightweight multi-label convolutional neural network in **<10ms on CPU/NPU**.
- **100% Privacy-Preserving (HIPAA Compliant):** Audio waveforms are converted to Mel-spectrograms in-memory, classified locally, and immediately purged. Zero raw voice recordings are transmitted to the cloud.
- Classifies 7 distinct acoustic biomarker categories:
  - 🫁 **Paroxysmal Coughing** (Cough count & frequency)
  - 🌬️ **Breathing & Wheezing** (Respiratory cycle pattern)
  - 💤 **Snoring & Obstructive Sleep Apnea Risk**
  - 🤧 **Sneezing Reflex**
  - 👶 **Pediatric & Neonatal Distress** (Infant crying)
  - 💬 **Speech & Conversation** (Ambient dialogue)
  - 🍃 **Ambient Baseline** (Calibrated silence)

### 2. 🏥 Native HL7 FHIR R4 & Azure Health Data Services Ingestion
- Every detected biomarker event is transformed on-the-fly into an **HL7 FHIR R4 `Observation`** resource.
- Standardized medical coding:
  - **LOINC:** `8687-6` (Coughing), `9279-1` (Respiratory Rate), `93832-4` (Sleep disturbance).
  - **SNOMED-CT:** `263731006` (Coughing finding), `56018004` (Wheezing finding), `271600006` (Snoring finding), `271633008` (Infant distress).
- Exports transaction bundles ready for **Azure Health Data Services (FHIR Server)** and hospital EHR systems.

### 3. 🤖 Microsoft Azure AI Foundry Clinical Synthesis
- Evaluates nocturnal telemetry clusters (e.g., paroxysmal cough frequency per hour) via **Azure OpenAI (GPT-4o-mini)**.
- Computes an **Exacerbation Risk Index (0-100)** and generates clinical summaries with actionable pulmonology recommendations.
- Integrates **Azure Speech Services** for hands-free patient distress communication.

### 4. 🌙 Dual Mode Bedside & Clinical Portal
- **Bedside Patient Sentinel Mode:** Minimalist, dark-mode screen designed for nighttime with zero visual distraction, displaying real-time acoustic radial pulses and privacy verification.
- **Clinical Research Dashboard:** Comprehensive telemetry panel with cough chronologies, nocturnal disturbance charts, FHIR JSON viewer, and Azure report generator.

---

## 🏛️ System Architecture

```mermaid
graph TD
    A[Bedside Microphone / Smartphone] -->|Web Audio API PCM| B[WebSocket /ws/acoustic]
    
    subgraph Edge AI Sentinel (Local Machine / Mobile)
        B --> C[Audio Preprocessor & Mel-Spectrogram]
        C --> D[Microsoft ONNX Runtime Engine]
        D -->|~10ms Inference| E{Biomarker Detected?}
        E -->|Cough, Snore, Wheeze, Distress| F[HL7 FHIR R4 Observation Formatter]
        E -->|Normal Ambient Baseline| G[Live Telemetry Stream]
    end
    
    subgraph Microsoft Cloud & Clinical Ecosystem
        F --> H[Azure Health Data Services FHIR Server]
        F --> I[Azure AI Foundry / OpenAI GPT-4o-mini]
        I --> J[Physician Clinical Synthesis Report]
    end
    
    F --> K[Frontend Bedside Sentinel & Clinical Dashboard]
    J --> K
```

---

## 🏥 Medical Standards & FHIR

Each observation is structured as a valid FHIR R4 resource:
```json
{
  "resourceType": "Observation",
  "status": "final",
  "category": [{ "coding": [{ "code": "exam", "display": "Exam" }] }],
  "code": {
    "coding": [
      { "system": "http://snomed.info/sct", "code": "263731006", "display": "Coughing (finding)" },
      { "system": "http://loinc.org", "code": "8687-6", "display": "Coughing [PhenX]" }
    ]
  },
  "subject": { "reference": "Patient/PATIENT-RESPISENSE-001" },
  "device": { "reference": "Device/DEVICE-ONNX-EDGE-01" },
  "valueQuantity": { "value": 99.9, "unit": "%" },
  "interpretation": [{ "coding": [{ "code": "A", "display": "Abnormal / Clinical Alert" }] }]
}
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/PoterasuStefan/NightDoc.git
cd NightDoc
```

### 2. Install Python Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Launch the Application Server
```bash
python backend/main.py
```

### 4. Open in Browser
👉 **[http://localhost:8000](http://localhost:8000)**  
👉 **API Swagger Docs:** `http://localhost:8000/docs`

---

## 📱 Mobile Setup

You can run NightDoc at the bedside on any smartphone over local Wi-Fi:
1. Connect your phone to the same Wi-Fi network as your computer.
2. The server displays your LAN IP on startup (e.g. `http://192.168.1.xxx:8000`).
3. Open the link on Chrome / Safari and tap **Mic Live** to start nocturnal acoustic monitoring.

---

## 🔐 Configuration

Copy `.env.example` to `.env` to configure Microsoft Azure credentials:
```bash
cp .env.example .env
```
```env
AZURE_AI_API_KEY=your_actual_azure_api_key_here
AZURE_AI_ENDPOINT=https://stefanpoterasu-2164-resource.cognitiveservices.azure.com/
AZURE_AI_REGION=eastus
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_SPEECH_LANGUAGE=ro-RO
```

---

## 📄 License
This project is licensed under the **MIT License**.

<div align="center">
  <sub>RespiSense AI / NightDoc – Microsoft Hackathon Health & Research.</sub>
</div>
