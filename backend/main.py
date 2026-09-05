import os
import sys
import json
import asyncio
import io
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from classifier import RespiSenseClassifier, CLASSES, CLASS_METADATA
from azure_foundry import azure_service
from fhir_formatter import fhir_formatter, MEDICAL_CODES

# Initialize FastAPI App
app = FastAPI(
    title="RespiSense AI – Acoustic Biomarker & Clinical Telemetry Backend",
    description="Edge AI (Microsoft ONNX Runtime) + Cloud Intelligence (Azure AI Foundry & HL7 FHIR R4)",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RespiSense ONNX Runtime Model
print("==================================================================")
print("  [*] Loading RespiSense AI ONNX Model (sound_radar_model.onnx)...")
classifier = RespiSenseClassifier()
print(f"  [*] ONNX Model loaded successfully! Supported Health Classes: {CLASSES}")
print("==================================================================")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket Client connected. Active clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket Client disconnected. Active clients: {len(self.active_connections)}")

    async def broadcast_json(self, data: dict, exclude: WebSocket = None):
        for connection in self.active_connections:
            if connection != exclude:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    print(f"Error broadcasting message: {e}")

manager = ConnectionManager()

# ----------------- REST API Endpoints ----------------- #

@app.get("/api/model-info")
async def get_model_info():
    """Returns local ONNX model configuration and medical coding metadata."""
    return {
        "status": "ready",
        "project": "RespiSense AI",
        "model_file": "sound_radar_model.onnx",
        "engine": "Microsoft ONNX Runtime",
        "classes": CLASSES,
        "class_metadata": CLASS_METADATA,
        "medical_standards": {
            "snomed_ct": True,
            "loinc": True,
            "hl7_fhir_r4": True
        }
    }

@app.get("/api/azure-info")
async def get_azure_info():
    """Returns Azure AI Foundry & Azure Health Data Services configuration."""
    return azure_service.get_info()

@app.get("/api/telemetry/stats")
async def get_telemetry_stats():
    """Returns active session clinical telemetry (cough frequency, nocturnal disturbance score)."""
    return classifier.get_telemetry_summary()

@app.post("/api/telemetry/reset")
async def reset_telemetry():
    """Resets the current monitoring session."""
    classifier.reset_telemetry()
    return {"success": True, "message": "Telemetry session reset successfully."}

@app.get("/api/fhir/observations")
async def get_fhir_observations(limit: int = 20):
    """Returns latest FHIR Observation resources."""
    all_obs = list(classifier.fhir_observations)
    return {
        "resourceType": "List",
        "total": len(all_obs),
        "entry": all_obs[-limit:]
    }

@app.post("/api/fhir/export-bundle")
async def export_fhir_bundle():
    """Exports all session observations packaged as an HL7 FHIR R4 Bundle."""
    all_obs = list(classifier.fhir_observations)
    bundle = fhir_formatter.create_bundle(all_obs)
    return bundle

@app.post("/api/clinical-summary")
async def generate_clinical_summary():
    """Triggers Azure AI Foundry (Azure OpenAI) to analyze session telemetry and generate clinical insights."""
    summary_data = classifier.get_telemetry_summary()
    analysis = await azure_service.generate_clinical_summary(summary_data)
    return analysis

@app.post("/api/classify-audio")
async def classify_audio(file: UploadFile = File(...)):
    """Classifies an uploaded WAV file and broadcasts the clinical biomarker event."""
    contents = await file.read()
    result = classifier.classify_wav_bytes(contents)
    
    if result.get("success"):
        # Broadcast biomarker event to all connected dashboards
        await manager.broadcast_json({
            "type": "biomarker_event",
            "payload": result
        })
        
        # If speech is detected, trigger Azure Speech transcription
        if result.get("predicted_class") == "conversation" and azure_service.is_configured():
            asyncio.create_task(transcribe_and_broadcast(contents, result.get("direction_angle", 0.0)))
            
    return result

async def transcribe_and_broadcast(audio_bytes: bytes, angle: float):
    try:
        tx = await azure_service.transcribe_audio_wav(audio_bytes)
        if tx.get("success") and tx.get("text"):
            await manager.broadcast_json({
                "type": "transcription",
                "payload": {
                    "text": tx["text"],
                    "is_final": True,
                    "angle": angle,
                    "source": "Azure AI Foundry (Speech)"
                }
            })
    except Exception as e:
        print(f"Azure Speech Transcription Error: {e}")

# ----------------- WebSocket Live Stream ----------------- #

@app.websocket("/ws/acoustic")
async def acoustic_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status & model info upon connection
        await websocket.send_json({
            "type": "status",
            "payload": {
                "connected": True,
                "label": "RespiSense AI Edge ONNX + Azure Foundry Connected",
                "classes": CLASSES,
                "azure_configured": azure_service.is_configured(),
                "telemetry": classifier.get_telemetry_summary()
            }
        })
        
        while True:
            message = await websocket.receive()
            
            # 1. Handle binary audio chunks (WAV or raw PCM)
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                result = classifier.classify_wav_bytes(audio_bytes)
                
                if result.get("success"):
                    # Broadcast biomarker to all listening dashboards
                    await manager.broadcast_json({
                        "type": "biomarker_event",
                        "payload": result
                    })

                    # If speech detected, trigger Azure transcription
                    if result.get("predicted_class") == "conversation" and azure_service.is_configured():
                        asyncio.create_task(transcribe_and_broadcast(audio_bytes, result.get("direction_angle", 0.0)))
            
            # 2. Handle JSON commands (e.g. simulation triggers or test commands)
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "test_sound":
                        sound_class = data.get("class", "coughing")
                        angle = data.get("angle", 45)
                        
                        meta = CLASS_METADATA.get(sound_class, CLASS_METADATA["coughing"])
                        probs = {c: 1.0 for c in CLASSES}
                        probs[sound_class] = 94.5
                        
                        # Generate FHIR observation for simulation
                        now_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                        fhir_obs = fhir_formatter.create_observation(
                            predicted_class=sound_class,
                            confidence=94.5,
                            probabilities=probs,
                            rms_energy=0.082,
                            direction_angle=angle,
                            timestamp=now_iso
                        )
                        classifier.fhir_observations.append(fhir_obs)
                        if sound_class in classifier.counts:
                            classifier.counts[sound_class] += 1

                        sim_result = {
                            "success": True,
                            "is_alert": meta["is_alert"],
                            "predicted_class": sound_class,
                            "title": meta["title"],
                            "icon": meta["icon"],
                            "color": meta["color"],
                            "confidence": 94.5,
                            "criticality": meta["criticality"],
                            "direction_angle": angle,
                            "probabilities": probs,
                            "rms_energy": 0.082,
                            "inference_ms": 7.4,
                            "fhir_id": fhir_obs["id"],
                            "fhir_resource": fhir_obs,
                            "snomed": meta.get("snomed", ""),
                            "loinc": meta.get("loinc", "")
                        }
                        
                        await manager.broadcast_json({
                            "type": "biomarker_event",
                            "payload": sim_result
                        })

                    elif msg_type == "request_telemetry":
                        await websocket.send_json({
                            "type": "telemetry_update",
                            "payload": classifier.get_telemetry_summary()
                        })

                except Exception as e:
                    print(f"Error processing JSON WebSocket message: {e}")
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        manager.disconnect(websocket)

# Serve Frontend static assets if available
FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

def get_local_ip():
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    local_ip = get_local_ip()
    print("==================================================================")
    print("  [*] RespiSense AI Server Active (Edge ONNX + Azure Foundry)")
    print("  [*] Local PC:    http://localhost:8000")
    print(f"  [*] Phone (LAN): http://{local_ip}:8000")
    print("  [*] Swagger Docs: http://localhost:8000/docs")
    print("==================================================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)
