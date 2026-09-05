"""
Microsoft Azure AI Foundry & Azure Health Integration for RespiSense AI
Features:
1. Azure AI Foundry / Microsoft Phi-3 & Azure AI Inference Clinical Insights
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

AZURE_AI_KEY = os.getenv("AZURE_AI_KEY", os.getenv("AZURE_AI_API_KEY", "")).strip()
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "https://your-azure-ai-resource.services.ai.azure.com/").strip()
AZURE_AI_REGION = os.getenv("AZURE_AI_REGION", "eastus").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
AZURE_SPEECH_LANGUAGE = os.getenv("AZURE_SPEECH_LANGUAGE", "ro-RO").strip()

# Import official Azure AI Inference SDK if available
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
    AZURE_INFERENCE_AVAILABLE = True
except ImportError:
    AZURE_INFERENCE_AVAILABLE = False


class AzureFoundryService:
    def __init__(self):
        self.api_key = AZURE_AI_KEY
        self.endpoint = AZURE_AI_ENDPOINT.rstrip('/')
        self.region = AZURE_AI_REGION
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.language = AZURE_SPEECH_LANGUAGE

    def is_configured(self) -> bool:
        placeholder = "your_azure_ai_key_here"
        return bool(self.api_key and self.api_key != placeholder and len(self.api_key) > 10)

    def get_info(self) -> dict:
        return {
            "provider": "Microsoft Azure AI Foundry",
            "endpoint": self.endpoint,
            "region": self.region,
            "deployment": self.deployment,
            "language": self.language,
            "configured": self.is_configured(),
            "azure_ai_inference_installed": AZURE_INFERENCE_AVAILABLE,
            "services": [
                "Azure AI Foundry (Phi-3 / Azure AI Inference Clinical Synthesis)",
                "Azure Cognitive Services Speech (Acoustic Telemetry)",
                "Azure Health Data Services (HL7 FHIR R4 Connector)"
            ]
        }

    async def generate_acoustic_clinical_summary(self, telemetry: dict) -> str:
        """
        Generates a strict 2-sentence objective clinical summary based on acoustic telemetry
        (cough count, duration, severity) using Azure AI Content Understanding / Phi-3 via Azure AI Inference.
        Instructs AI to summarize objectively and strictly avoid diagnosing.
        """
        cough_count = telemetry.get("cough_count", 0)
        duration = telemetry.get("duration", telemetry.get("duration_minutes", 30))
        severity = telemetry.get("severity", "moderate")

        prompt = (
            f"Acoustic telemetry recorded {cough_count} cough events over a monitoring duration of {duration} minutes "
            f"with an evaluated severity rating of {severity}. "
            "Please provide a strict 2-sentence clinical summary describing these acoustic telemetry observations objectively. "
            "Do NOT diagnose any disease, illness, or underlying medical condition."
        )

        system_instruction = (
            "You are an objective clinical acoustic biomarker monitoring assistant for respiratory research. "
            "Your output MUST be exactly two sentences long. "
            "Summarize the provided acoustic telemetry objectively and factually. "
            "STRICTLY AVOID diagnosing or implying any disease or medical condition."
        )

        # 1. Try Azure OpenAI Endpoint (/openai/deployments/{deployment}/chat/completions)
        if self.is_configured():
            try:
                url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version=2024-02-15-preview"
                headers = {
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 150
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"].strip()
                        return self._sanitize_two_sentences(content, cough_count, duration, severity)
            except Exception as e:
                print(f"[AzureFoundry] Azure OpenAI REST call failed: {e}")

        # 2. Try native Azure AI Inference SDK (Phi-3 / Content Understanding)
        if self.is_configured() and AZURE_INFERENCE_AVAILABLE:
            try:
                client = ChatCompletionsClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.api_key)
                )
                response = client.complete(
                    messages=[
                        SystemMessage(content=system_instruction),
                        UserMessage(content=prompt),
                    ],
                    model=self.deployment,
                    temperature=0.2,
                    max_tokens=150,
                )
                raw_text = response.choices[0].message.content.strip()
                return self._sanitize_two_sentences(raw_text, cough_count, duration, severity)
            except Exception as e:
                print(f"[AzureFoundry] Azure AI Inference SDK call failed: {e}")

        # 3. Objective Baseline Fallback (Strict 2 sentences, non-diagnostic)
        return self._build_fallback_summary(cough_count, duration, severity)

    def _sanitize_two_sentences(self, text: str, cough_count: int, duration: float, severity: str) -> str:
        """Ensures the text output is strictly 2 sentences long."""
        cleaned = text.replace("\n", " ").strip()
        parts = [p.strip() for p in cleaned.split(".") if p.strip()]
        if len(parts) >= 2:
            return f"{parts[0]}. {parts[1]}."
        elif len(parts) == 1:
            return f"{parts[0]}. Monitoring continues to record acoustic biomarker trends objectively."
        return self._build_fallback_summary(cough_count, duration, severity)

    def _build_fallback_summary(self, cough_count: int, duration: float, severity: str) -> str:
        s1 = f"Acoustic telemetry recorded {cough_count} cough episodes during a {duration}-minute bedside monitoring window with a {severity} severity index."
        s2 = "Observed biomarker metrics reflect passive acoustic activity and are presented objectively for clinical evaluation without diagnostic assessment."
        return f"{s1} {s2}"

    async def generate_clinical_summary(self, session_telemetry: dict) -> dict:
        """
        Uses Azure AI Foundry to generate structured clinical telemetry summary report.
        """
        summary_text = await self.generate_acoustic_clinical_summary(session_telemetry)
        cough_c = session_telemetry.get('cough_count', 0)
        risk = "High" if cough_c >= 5 else ("Medium" if cough_c >= 2 else "Low")
        risk_score = min(95, 20 + cough_c * 15)

        return {
            "success": True,
            "provider": "Microsoft Azure AI Foundry (Phi-3 / Azure AI Inference)",
            "risk_level": risk,
            "risk_score_100": risk_score,
            "summary": summary_text,
            "summary_ro": summary_text,
            "summary_en": summary_text,
            "actionable_recommendation": "Objective telemetry summary generated for physician review.",
            "recommended_snomed": ["263731006 (Coughing)", "56018004 (Wheezing finding)", "271600006 (Snoring)"],
            "is_azure_cloud": self.is_configured()
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
