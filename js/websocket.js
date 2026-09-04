/**
 * WebSocket Client for communicating with the Python / FastAPI backend.
 * Receives sound classification alerts, directional calculations, and live subtitles.
 */
class BackendSocket {
  constructor(url = null) {
    if (!url) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname || 'localhost';
      const port = window.location.port || '8000';
      this.url = `${protocol}//${host}:${port}/ws/acoustic`;
    } else {
      this.url = url;
    }
    this.socket = null;
    this.isConnected = false;
    this.listeners = {
      alert: [],
      transcription: [],
      status: []
    };
  }

  connect() {
    try {
      this.socket = new WebSocket(this.url);

      this.socket.onopen = () => {
        this.isConnected = true;
        this.emit('status', { connected: true, label: 'FastAPI Backend Online' });
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'alert') {
            this.emit('alert', data.payload);
          } else if (data.type === 'transcription') {
            this.emit('transcription', data.payload);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket JSON payload:', e);
        }
      };

      this.socket.onclose = () => {
        this.isConnected = false;
        this.emit('status', { connected: false, label: 'Standalone Demo' });
      };

      this.socket.onerror = () => {
        this.isConnected = false;
        this.emit('status', { connected: false, label: 'Standalone Demo' });
      };
    } catch (err) {
      this.isConnected = false;
      this.emit('status', { connected: false, label: 'Standalone Demo' });
    }
  }

  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    }
  }

  emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(cb => cb(data));
    }
  }

  sendAudioChunk(chunkBuffer) {
    if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(chunkBuffer);
    }
  }

  sendText(text) {
    if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(typeof text === 'string' ? text : JSON.stringify(text));
    }
  }
}

window.BackendSocket = BackendSocket;
