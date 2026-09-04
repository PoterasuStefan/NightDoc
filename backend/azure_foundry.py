"""
Microsoft Foundry & Azure AI Content Understanding / Speech Integration
Connects to Azure AI Services hosted under resource: stefanpoterasu-2164-resource
"""
import os
import io
import wave
import json
import asyncio
import httpx
from dotenv import load_dotenv

# Load environment variables from project root .env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

AZURE_AI_API_KEY = os.getenv("AZURE_AI_API_KEY", "").strip()
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "https://stefanpoterasu-2164-resource.cognitiveservices.azure.com/").strip()
AZURE_AI_REGION = os.getenv("AZURE_AI_REGION", "eastus").strip()
AZURE_CONTENT_UNDERSTANDING_ANALYZER = os.getenv("AZURE_CONTENT_UNDERSTANDING_ANALYZER", "conversation-analyzer").strip()
AZURE_SPEECH_LANGUAGE = os.getenv("AZURE_SPEECH_LANGUAGE", "ro-RO").strip()

class AzureFoundryService:
    def __init__(self):
        self.api_key = AZURE_AI_API_KEY
        self.endpoint = AZURE_AI_ENDPOINT.rstrip('/')
        self.region = AZURE_AI_REGION
        self.analyzer_id = AZURE_CONTENT_UNDERSTANDING_ANALYZER
        self.language = AZURE_SPEECH_LANGUAGE

    def is_configured(self) -> bool:
        """Returns True if user has configured an actual Azure API key."""
        return bool(self.api_key and self.api_key != "your_azure_foundry_api_key_here" and len(self.api_key) > 10)

    def get_info(self) -> dict:
        return {
            "provider": "Microsoft Azure AI Foundry",
            "resource": "stefanpoterasu-2164-resource",
            "endpoint": self.endpoint,
            "region": self.region,
            "language": self.language,
            "analyzer_id": self.analyzer_id,
            "configured": self.is_configured()
        }

    async def transcribe_audio_wav(self, wav_bytes: bytes) -> dict:
        """
        Transcribes audio using Azure AI Speech SDK or Content Understanding REST API.
        Falls back cleanly if API key is not yet configured.
        """
        if not self.is_configured():
            return {
                "success": False,
                "configured": False,
                "text": "",
                "message": "Azure AI Key not set in .env. Enter key to enable cloud subtitles."
            }

        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Setup speech config
            speech_config = speechsdk.SpeechConfig(
                subscription=self.api_key,
                region=self.region
            )
            speech_config.speech_recognition_language = self.language
            
            # Setup push audio stream from WAV bytes
            push_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            
            # Feed WAV raw payload (skip 44-byte header if standard WAV)
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
            
            # Run async recognition in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, recognizer.recognize_once)
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return {
                    "success": True,
                    "configured": True,
                    "text": result.text,
                    "source": "azure_speech_foundry"
                }
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return {
                    "success": False,
                    "configured": True,
                    "text": "",
                    "message": "No speech recognized in audio buffer"
                }
            else:
                return {
                    "success": False,
                    "configured": True,
                    "text": "",
                    "message": f"Azure Speech status: {result.reason}"
                }
        except Exception as e:
            # Fallback to Azure Content Understanding REST API call if SpeechSDK fails
            return await self._call_content_understanding_api(wav_bytes, str(e))

    async def _call_content_understanding_api(self, wav_bytes: bytes, sdk_error: str) -> dict:
        """Call Microsoft Foundry Content Understanding REST Endpoint directly."""
        url = f"{self.endpoint}/contentunderstanding/analyzers/{self.analyzer_id}:analyze?api-version=2024-12-01-preview"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "audio/wav"
        }
        
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(url, headers=headers, content=wav_bytes)
                if resp.status_code in [200, 202]:
                    data = resp.json()
                    return {
                        "success": True,
                        "configured": True,
                        "text": data.get("result", {}).get("transcript", ""),
                        "raw": data,
                        "source": "azure_content_understanding"
                    }
                else:
                    return {
                        "success": False,
                        "configured": True,
                        "text": "",
                        "message": f"HTTP {resp.status_code}: {resp.text}"
                    }
        except Exception as err:
            return {
                "success": False,
                "configured": True,
                "text": "",
                "message": f"Azure Foundry call error: {err} (SDK Error: {sdk_error})"
            }

azure_service = AzureFoundryService()
