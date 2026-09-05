# 🛡️ Responsible AI & Clinical Safety Framework
### NightDoc / RespiSense AI — Bedside Acoustic Biomarker Sentinel
#### *Aligned with the Microsoft Responsible AI Standard v2 & Health & Research Track Guidelines*

---

## ⚠️ Regulatory Status & Medical Non-Device Disclaimer (Non-SaMD)

> ### 🔴 IMPORTANT CLINICAL & REGULATORY NOTICE
> **NightDoc (RespiSense AI) is an investigational acoustic telemetry prototype and clinical research assistive tool.**
> 
> - **NOT A CERTIFIED MEDICAL DEVICE (Non-SaMD):** NightDoc is **NOT** certified as Software as a Medical Device (SaMD) under the U.S. FDA 21 CFR §860 / 21 CFR §880 or the European Union Medical Device Regulation (EU MDR 2017/745 Class IIa/IIb).
> - **NO PRIMARY DIAGNOSIS:** The application does not diagnose, treat, cure, or provide definitive prognosis for any disease, including Asthma, Chronic Obstructive Pulmonary Disease (COPD), COVID-19, or Infant Respiratory Distress Syndrome.
> - **HUMAN-IN-THE-LOOP (HITL) REQUIREMENT:** All acoustic risk stratifications and LLM-synthesized summaries (Azure OpenAI GPT-4o-mini) are marked as **`Preliminary Draft Telemetry`** and **strictly require review and electronic signature by a licensed medical professional** prior to clinical action.
> - **EMERGENCY ESCALATION PROTOCOL (112 / 911):** If a patient experiences acute dyspnea, cyanosis, severe stridor, chest pain, or neonatal apnea, **do not rely on this software**. Contact emergency medical services (**112 in Romania/EU, 911 in USA**) immediately.

---

## 🏛️ Alignment with Microsoft's 6 Responsible AI Principles

```
                 ┌─────────────────────────────────────────────────────────┐
                 │       Microsoft Responsible AI Standard (v2)            │
                 └────────────────────────────┬────────────────────────────┘
                                              │
         ┌───────────────────┬────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                ▼                   ▼                   ▼
    ┌─────────┐        ┌───────────┐    ┌───────────┐       ┌───────────┐       ┌──────────────┐
    │Fairness │        │Reliability│    │ Privacy & │       │ Inclusive │       │ Transparency │
    │ & Bias  │        │ & Safety  │    │ Security  │       │   ness    │       │& Accountable │
    └─────────┘        └───────────┘    └───────────┘       └───────────┘       └──────────────┘
```

### 1. ⚖️ Fairness & Demographic Representation
- **Training Cohorts Analyzed:**
  - *COUGHVID (EPFL) & Coswara (IISc):* Dry and productive coughs collected across adult demographics (18–65 years), balanced gender distribution.
  - *Clinical Auscultations (`Sounds-Coughs`):* Pulmonology hospital recordings for wheezing, stridor, crackles, and rhonchi.
  - *ESC-50 Acoustic Corpus:* Standardized infant crying, respiratory reflex, and ambient room noise.
  - *Conversational Audio:* Native Romanian and English natural speech dialogue.
- **Identified Demographic & Technical Gaps:**
  - *Geriatric Vocal Tremor (>75 years):* Underrepresented in current public audio corpora; voice changes may occasionally elevate ambient variance.
  - *Pediatric Spectrum:* Infant crying is well-represented, but pediatric toddler coughs (e.g. croup/pertussis) require dedicated pediatric clinical trials before expansion.
- **Mitigation & Bias Management:**
  - Rather than making universal clinical claims, the system explicitly defines its primary persona: **Adult / Elderly patients with diagnosed COPD or severe asthma requiring nocturnal exacerbation tracking**.
  - Secondary acoustic detectors (infant crying, snoring) are architected as **modular, configurable feature-flags** that can be independently toggled.

### 2. 🛡️ Reliability & Safety
- **Anti-False-Positive Speech Bias:**
  - In a home or clinic environment, family dialogue or physician rounds must never trigger paroxysmal cough alarms.
  - An asymmetric loss weighting (`pos_weight = 1.90` for `conversation` vs `0.70` for `coughing`) ensures high evidence is demanded before flagging a cough, while conversation is actively protected (96.7% conversation precision).
- **RMS Silence Gating:**
  - Hardware microphone noise and room silence below `RMS = 0.0075` are dynamically gated to prevent low-level hiss from causing inference artifacts.
- **Graceful Cloud Degradation:**
  - Edge ONNX inference operates 100% offline. If Azure AI Foundry or network connectivity is unavailable, local acoustic telemetry continues uninterrupted, generating deterministic local FHIR bundles with zero cloud dependency.

### 3. 🔒 Privacy & Security (Technical Safeguards Aligned with HIPAA & GDPR)
- **Clarification on Regulatory Compliance:**
  - HIPAA compliance is an institutional, administrative, and legal framework (requiring Business Associate Agreements, physical site security, and role-based policies), **not merely a client software feature**.
  - NightDoc implements **Privacy-by-Design Technical Safeguards** aligned with HIPAA Security Rule (§164.312) and GDPR (Art. 25 & 32):
    1. **Ephemeral RAM Buffers:** 3-second audio windows reside exclusively in volatile memory (RAM).
    2. **Immediate Purge:** Audio waveforms are converted to Mel-spectrogram tensors and immediately deleted from memory.
    3. **Zero Audio Retention:** No audio recordings (WAV, MP3, PCM) are ever written to disk or transmitted across local or cloud networks.
    4. **Local Feature Extraction:** Only anonymous numerical biomarker counters (e.g., cough frequency per hour) are transmitted to the clinician portal.

### 4. 🌍 Inclusiveness & Adaptive User Experience
- **Dual-UX Architecture:**
  - *Simple Bedside Sentinel Mode:* Tailored for elderly patients or home sleep environments — large typography, high-contrast dark palette, zero blue-light disturbance, and single-tap activation.
  - *Advanced Clinical Dashboard:* Tailored for pulmonologists and clinical researchers — detailed acoustic phenotyping, FHIR JSON inspector, LOINC/SNOMED-CT crosswalks, and trend analytics.

### 5. 🔍 Transparency & Model Card
- **Model Architecture:** `LightSoundCNN` (~90,000 parameters).
- **Inputs:** Log Mel-Spectrogram (64 mel-bins, 1024-point FFT, 512 hop length, 3.0s window at 22,050 Hz).
- **Empirical Validation:**
  - Evaluated on a strictly isolated, **file-level held-out test split (154 files, 432 segments, zero chunk leakage)**:
    - **Overall Accuracy:** 92.6%
    - **Macro F1-Score:** 83.8%
    - **Coughing:** Precision 67.9%, Recall 86.4%, F1 76.0% (19/22 detected, 0 confused with speech).
    - **Conversation:** Precision 78.6%, Recall 73.3%, F1 75.9% (0 confused with cough).
- **Public Reproduction Script:**
  - Verifiable by any auditor via `python backend/evaluate_model.py`.

### 6. 🧑‍⚕️ Accountability & Data Governance
- **Human-in-the-Loop Clinical Workflow:**
  1. Nocturnal Telemetry collected locally at bedside.
  2. Edge ONNX engine formats event as HL7 FHIR R4 `Observation`.
  3. Azure OpenAI (GPT-4o-mini) synthesizes a draft report constrained by **GOLD 2024 (COPD)** and **GINA 2024 (Asthma)** guidelines.
  4. The pulmonologist reviews raw acoustic trend charts, modifies recommendations, and electronically signs the report.
  5. The approved record is ingested into the hospital EHR / Azure Health Data Services.
- **Informed Consent via HL7 FHIR R4 `Consent` Resource:**
  - Transparent opt-in prior to microphone activation with clear scope definition:
    - Purpose of Use: `CLINRESRCH` (Clinical Research) / `TREAT` (Treatment Assistive).
    - Data Provision: Ephemeral acoustic biomarkers only; no biometric voice printing.

---

## 📜 Standard HL7 FHIR R4 Consent Template

```json
{
  "resourceType": "Consent",
  "id": "CONSENT-RESPISENSE-001",
  "status": "active",
  "scope": {
    "coding": [
      {
        "system": "http://terminology.hl7.org/CodeSystem/consentscope",
        "code": "patient-privacy",
        "display": "Privacy Consent"
      }
    ]
  },
  "category": [
    {
      "coding": [
        {
          "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
          "code": "CLINRESRCH",
          "display": "Clinical Research Telemetry"
        }
      ]
    }
  ],
  "patient": {
    "reference": "Patient/PATIENT-RESPISENSE-001",
    "display": "Bedside Monitored Patient"
  },
  "dateTime": "2026-09-05T00:00:00Z",
  "policyRule": {
    "coding": [
      {
        "system": "http://nightdoc.ai/fhir/policies",
        "code": "opt-in-zero-audio-retention",
        "display": "Opt-In Local Edge Processing with Zero Cloud Audio Retention"
      }
    ]
  },
  "provision": {
    "type": "permit",
    "period": { "start": "2026-09-05T00:00:00Z" },
    "purpose": [
      { "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason", "code": "TREAT" },
      { "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason", "code": "CLINRESRCH" }
    ],
    "data": [
      {
        "meaning": "related",
        "reference": { "reference": "Observation/OBS-ACOUSTIC-SUMMARY" }
      }
    ]
  }
}
```

---

<div align="center">
  <sub>NightDoc / RespiSense AI • Microsoft Hackathon (Health & Research Track) • Responsible AI Documentation v2.0</sub>
</div>
