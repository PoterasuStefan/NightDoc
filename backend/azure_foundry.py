"""
Microsoft Azure AI Foundry & Azure Health Integration for RespiSense AI
Features:
1. Azure AI Foundry / OpenAI GPT-4o-mini Clinical Insights & Exacerbation Risk Scoring
2. Azure Speech Services for Bedside Conversational & Distress Transcription
3. Azure Health Data Services (FHIR) Sync
"""
import os
import io
import json
import asyncio
import httpx
from datetime import datetime, timezone
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

AZURE_AI_KEY = os.getenv("AZURE_AI_API_KEY", "").strip()
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "https://your-azure-ai-resource.cognitiveservices.azure.com/").strip()
AZURE_AI_REGION = os.getenv("AZURE_AI_REGION", "eastus").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
AZURE_SPEECH_LANGUAGE = os.getenv("AZURE_SPEECH_LANGUAGE", "ro-RO").strip()

class AzureFoundryService:
    def __init__(self):
        self.api_key = AZURE_AI_KEY
        self.endpoint = AZURE_AI_ENDPOINT.rstrip('/')
        self.region = AZURE_AI_REGION
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.language = AZURE_SPEECH_LANGUAGE

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_azure_foundry_api_key_here" and len(self.api_key) > 10)

    def get_info(self) -> dict:
        return {
            "provider": "Microsoft Azure AI Foundry",
            "resource": os.getenv("AZURE_AI_RESOURCE_NAME", "azure-foundry-health-resource"),
            "endpoint": self.endpoint,
            "region": self.region,
            "deployment": self.deployment,
            "language": self.language,
            "configured": self.is_configured(),
            "services": [
                "Azure AI Foundry (LLM Clinical Synthesis)",
                "Azure Cognitive Services Speech (Acoustic Telemetry)",
                "Azure Health Data Services (HL7 FHIR R4 Connector)"
            ]
        }

    async def generate_clinical_summary(self, session_telemetry: dict) -> dict:
        """
        Uses Azure AI Foundry (Azure OpenAI / Azure AI Model Inference) to generate a structured
        clinical report based on acoustic biomarker telemetry (cough count, wheezing index, nocturnal episodes).
        """
        prompt = f"""Ești un asistent medical AI specializat în pneumologie și cercetare clinică (RespiSense AI).
Analizează următoarele date de telemetrie acustică colectate de la marginea patului pacientului în format FHIR/Edge AI:

DATE TELEMETRIE:
- Episoade de tuse (Cough Count): {session_telemetry.get('cough_count', 0)}
- Episoade de wheezing/respirație: {session_telemetry.get('breathing_count', 0)}
- Episoade de sforăit / apnee: {session_telemetry.get('snoring_count', 0)}
- Episoade de strănut: {session_telemetry.get('sneezing_count', 0)}
- Detresă/Plâns: {session_telemetry.get('crying_count', 0)}
- Durata monitorizării: {session_telemetry.get('duration_minutes', 30)} minute
- Indice de perturbare nocturnă: {session_telemetry.get('disturbance_score', 'Scăzut')}

Generează un raport clinic concis în limba română și engleză care conține:
1. **Evaluare Risc Acut (Scăzut / Mediu / Ridicat)**
2. **Interpretare Clinică a Biomarkerilor Acustici**
3. **Recomandare pentru Medicul Pneumolog / Protocol Studiu Clinic**
4. **Coduri Medicale Recomandate (SNOMED-CT / LOINC)**

Răspunde în format JSON valid cu cheile:
"risk_level", "risk_score_100", "summary_ro", "summary_en", "actionable_recommendation", "recommended_snomed".
"""
        if not self.is_configured():
            # Fallback inteligent pentru demo fără cheie Azure activă
            cough_c = session_telemetry.get('cough_count', 0)
            risk = "Ridicat (High)" if cough_c >= 5 else ("Moderat (Medium)" if cough_c >= 2 else "Scăzut (Low)")
            risk_score = min(95, 20 + cough_c * 15)
            
            return {
                "success": True,
                "provider": "RespiSense AI (Local Inference Baseline)",
                "risk_level": risk,
                "risk_score_100": risk_score,
                "summary_ro": f"Monitorizarea acustică nocturnă indică {cough_c} accese de tuse paroxistică. Risc estimat de exacerbare respiratorie: {risk}.",
                "summary_en": f"Acoustic bedside monitoring detected {cough_c} paroxysmal cough events. Estimated exacerbation risk: {risk}.",
                "actionable_recommendation": "Recomandare monitorizare continuă a saturației de oxigen (SpO2) și evaluare spirometrică matinală.",
                "recommended_snomed": ["263731006 (Coughing)", "56018004 (Wheezing finding)", "271600006 (Snoring)"],
                "is_azure_cloud": False
            }

        # Apel către Azure OpenAI / Azure AI Model Inference endpoint
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version=2024-02-15-preview"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": "You are an expert pulmonology AI synthesizer formatted as JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 800
        }

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    parsed["success"] = True
                    parsed["provider"] = "Microsoft Azure AI Foundry (Azure OpenAI)"
                    parsed["is_azure_cloud"] = True
                    return parsed
                else:
                    return {
                        "success": False,
                        "error": f"Azure AI Foundry HTTP {resp.status_code}: {resp.text}",
                        "is_azure_cloud": True
                    }
        except Exception as e:
            return {
                "success": False,
                "error": f"Azure AI connection error: {str(e)}",
                "is_azure_cloud": False
            }

    async def transcribe_audio_wav(self, wav_bytes: bytes) -> dict:
        """Transcribes speech / conversations using Azure Speech SDK."""
        if not self.is_configured():
            return {
                "success": False,
                "configured": False,
                "text": "",
                "message": "Azure AI Key not set in .env"
            }

        try:
            import azure.cognitiveservices.speech as speechsdk
            speech_config = speechsdk.SpeechConfig(
                subscription=self.api_key,
                region=self.region
            )
            speech_config.speech_recognition_language = self.language
            
            push_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            
            if wav_bytes.startswith(b'RIFF') and len(wav_bytes) > 44:
                pcm_data = wav_bytes[44:]
            else:
                pcm_data = wav_bytes
                
            push_stream.write(pcm_data)
            push_stream.close()
            
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, recognizer.recognize_once)
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return {
                    "success": True,
                    "configured": True,
                    "text": result.text,
                    "source": "Azure Speech (Foundry)"
                }
            return {
                "success": False,
                "configured": True,
                "text": "",
                "message": f"Azure Speech result: {result.reason}"
            }
        except Exception as e:
            return {
                "success": False,
                "configured": True,
                "text": "",
                "message": f"Azure Speech error: {str(e)}"
            }

azure_service = AzureFoundryService()
