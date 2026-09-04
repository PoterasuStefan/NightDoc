import os
import sys
import json
import asyncio
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from classifier import SoundRadarClassifier, CLASSES, CLASS_METADATA
from azure_foundry import azure_service

# Initialize FastAPI app
app = FastAPI(title="SoundSense Acoustic Radar Backend")

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Local ONNX Model
print("Loading Local ONNX Acoustic Model (sound_radar_model.onnx)...")
classifier = SoundRadarClassifier()
print(f"ONNX Model loaded successfully! Supported classes: {CLASSES}")

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

# REST Endpoints
@app.get("/api/model-info")
async def get_model_info():
    return {
        "status": "ready",
        "model_file": "sound_radar_model.onnx",
        "engine": "ONNX Runtime",
        "classes": CLASSES,
        "class_metadata": CLASS_METADATA
    }

@app.get("/api/azure-info")
async def get_azure_info():
    """Returns Microsoft Foundry & Azure AI configuration status."""
    return azure_service.get_info()

async def transcribe_and_broadcast(audio_bytes: bytes, angle: float):
    """Processes speech audio via Azure Foundry / Azure Speech and broadcasts subtitles."""
    try:
        tx = await azure_service.transcribe_audio_wav(audio_bytes)
        if tx.get("success") and tx.get("text"):
            await manager.broadcast_json({
                "type": "transcription",
                "payload": {
                    "text": tx["text"],
                    "is_final": True,
                    "angle": angle,
                    "source": "Microsoft Foundry (Azure AI)"
                }
            })
    except Exception as e:
        print(f"Azure Speech Transcription Error: {e}")

@app.post("/api/classify-audio")
async def classify_audio(file: UploadFile = File(...)):
    """
    Classifies an uploaded WAV/MP3/audio file using the local ONNX model
    and broadcasts the result to all connected radar clients.
    """
    contents = await file.read()
    result = classifier.classify_wav_bytes(contents)
    
    if result.get("success"):
        is_speech = (result.get("predicted_class") == "conversation") or (result.get("probabilities", {}).get("conversation", 0) > 35.0)

        if result.get("is_alert"):
            # Alert confirmat: difuzăm alerta completă către radar
            await manager.broadcast_json({
                "type": "alert",
                "payload": {
                    "is_background": False,
                    "is_speech": is_speech,
                    "title": result["title"],
                    "direction": result["direction_label"],
                    "angle": result["direction_angle"],
                    "icon": result["icon"],
                    "color": result["color"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "intensity": min(1.4, max(0.8, result["confidence"] / 70.0)),
                    "alert_status": result.get("alert_status", "confirmed")
                }
            })
        else:
            # Zgomot ambiental / chatter / liniște: mod de ascultare fără alertă intruzivă
            await manager.broadcast_json({
                "type": "ambient",
                "payload": {
                    "is_background": True,
                    "is_speech": is_speech,
                    "title": result["title"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "rms_energy": result.get("rms_energy", 0.0),
                    "alert_status": result.get("alert_status", "ambient")
                }
            })

        if is_speech and azure_service.is_configured():
            asyncio.create_task(transcribe_and_broadcast(contents, result.get("direction_angle", 0.0)))
    
    return result

@app.websocket("/ws/acoustic")
async def acoustic_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status & model info upon connection
        await websocket.send_json({
            "type": "status",
            "payload": {
                "connected": True,
                "label": "Local ONNX + Azure Foundry Ready",
                "classes": CLASSES,
                "azure_configured": azure_service.is_configured()
            }
        })
        
        while True:
            message = await websocket.receive()
            
            # 1. Handle binary audio chunks (WAV or raw PCM)
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                
                # Run ONNX inference
                result = classifier.classify_wav_bytes(audio_bytes)
                
                if result.get("success"):
                    is_speech = (result.get("predicted_class") == "conversation") or (result.get("probabilities", {}).get("conversation", 0) > 35.0)

                    if result.get("is_alert"):
                        # Alert confirmat: difuzăm alerta completă către radar
                        await manager.broadcast_json({
                            "type": "alert",
                            "payload": {
                                "is_background": False,
                                "is_speech": is_speech,
                                "title": result["title"],
                                "direction": result["direction_label"],
                                "angle": result["direction_angle"],
                                "icon": result["icon"],
                                "color": result["color"],
                                "confidence": result["confidence"],
                                "probabilities": result["probabilities"],
                                "intensity": min(1.4, max(0.8, result["confidence"] / 70.0)),
                                "alert_status": result.get("alert_status", "confirmed")
                            }
                        })
                    else:
                        # Zgomot ambiental / chatter / liniște / în curs de confirmare
                        await manager.broadcast_json({
                            "type": "ambient",
                            "payload": {
                                "is_background": True,
                                "is_speech": is_speech,
                                "title": result["title"],
                                "confidence": result["confidence"],
                                "probabilities": result["probabilities"],
                                "rms_energy": result.get("rms_energy", 0.0),
                                "alert_status": result.get("alert_status", "ambient")
                            }
                        })

                    # If speech is detected, trigger Azure Foundry speech transcription
                    if is_speech and azure_service.is_configured():
                        asyncio.create_task(transcribe_and_broadcast(audio_bytes, result.get("direction_angle", 0.0)))
            
            # 2. Handle text/JSON commands from frontend (e.g. simulation triggers or speech events)
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "test_sound":
                        # Test a specific sound class with sample audio if available
                        sound_class = data.get("class", "crying_baby")
                        angle = data.get("angle", 60)
                        
                        meta = CLASS_METADATA.get(sound_class, {
                            "title": sound_class.replace('_', ' ').title(),
                            "icon": "🔊",
                            "color": "#FACC15"
                        })
                        
                        # Generate mock probabilities focusing on this class
                        probs = {c: 1.0 for c in CLASSES}
                        probs[sound_class] = 92.4
                        
                        await manager.broadcast_json({
                            "type": "alert",
                            "payload": {
                                "title": meta["title"],
                                "direction": f"{angle}° detected",
                                "angle": angle,
                                "icon": meta["icon"],
                                "color": meta["color"],
                                "confidence": 92,
                                "probabilities": probs,
                                "intensity": 1.0
                            }
                        })
                        
                    elif msg_type == "live_speech":
                        # Forward speech transcript to other connected clients (exclude sender)
                        await manager.broadcast_json({
                            "type": "transcription",
                            "payload": {
                                "text": data.get("text", ""),
                                "is_final": data.get("is_final", False),
                                "angle": data.get("angle", 0),
                                "source": data.get("source", "Connected Peer")
                            }
                        }, exclude=websocket)
                except Exception as e:
                    print(f"Error processing JSON WebSocket message: {e}")
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        manager.disconnect(websocket)

# Mount frontend files at root
FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    print("==================================================================")
    print("  [*] SoundSense Radar Server Active (PC & Mobile)")
    print("  [*] Local PC:   http://localhost:8000")
    print("  [*] Phone (LAN): http://192.168.1.208:8000")
    print("==================================================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)
