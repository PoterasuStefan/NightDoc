/**
 * Audio Engine - Handles Web Audio API capture, volume metering,
 * frequency analysis, and real-time audio chunking for ONNX classification.
 */
class AudioEngine {
  constructor() {
    this.audioCtx = null;
    this.stream = null;
    this.analyser = null;
    this.source = null;
    this.processor = null;
    this.isListening = false;
    this.energyCallback = null;
    this.audioChunkCallback = null;
    this.bufferData = null;
    
    // 2-3 second rolling audio buffer (at 22050 Hz)
    this.recordedSamplesL = [];
    this.recordedSamplesR = [];
    this.maxSamples = 22050 * 2.5; // ~2.5 seconds
    this.lastTriggerTime = 0;
  }

  async startMicrophone() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 2, // Stereo for L/R phase & direction calculation
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });

      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 22050
      });

      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;

      this.source = this.audioCtx.createMediaStreamSource(this.stream);
      this.source.connect(this.analyser);

      // ScriptProcessorNode for streaming PCM audio chunks
      const bufferSize = 4096;
      this.processor = this.audioCtx.createScriptProcessor(bufferSize, 2, 2);
      this.source.connect(this.processor);
      this.processor.connect(this.audioCtx.destination);

      this.processor.onaudioprocess = (e) => {
        if (!this.isListening) return;

        const left = e.inputBuffer.getChannelData(0);
        const right = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : left;

        // Append to rolling buffers
        for (let i = 0; i < left.length; i++) {
          this.recordedSamplesL.push(left[i]);
          this.recordedSamplesR.push(right[i]);
        }

        // Limit buffer size to last 2.5 seconds
        if (this.recordedSamplesL.length > this.maxSamples) {
          this.recordedSamplesL = this.recordedSamplesL.slice(-this.maxSamples);
          this.recordedSamplesR = this.recordedSamplesR.slice(-this.maxSamples);
        }

        // Check if loud sound occurred (energy peak) to send buffer to ONNX model
        const now = performance.now();
        if (now - this.lastTriggerTime > 1800 && this.recordedSamplesL.length >= 22050 * 1.5) {
          // Calculate RMS of recent slice
          let sum = 0;
          for (let i = left.length - 1024; i < left.length; i++) {
            sum += left[i] * left[i];
          }
          const rms = Math.sqrt(sum / 1024);

          if (rms > 0.04) {
            this.lastTriggerTime = now;
            this.dispatchWavBuffer();
          }
        }
      };

      this.bufferData = new Uint8Array(this.analyser.frequencyBinCount);
      this.isListening = true;
      this.monitorLoop();
      return true;
    } catch (err) {
      console.warn('Microphone permission not granted or not available:', err);
      return false;
    }
  }

  stopMicrophone() {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close();
      this.audioCtx = null;
    }
    this.isListening = false;
    this.recordedSamplesL = [];
    this.recordedSamplesR = [];
  }

  monitorLoop() {
    if (!this.isListening || !this.analyser) return;

    this.analyser.getByteFrequencyData(this.bufferData);

    // Calculate overall audio energy / volume level (0.0 to 1.0)
    let sum = 0;
    for (let i = 0; i < this.bufferData.length; i++) {
      sum += this.bufferData[i];
    }
    const average = sum / this.bufferData.length;
    const energy = Math.min(1.0, average / 128);

    if (this.energyCallback) {
      this.energyCallback(energy, this.bufferData);
    }

    requestAnimationFrame(() => this.monitorLoop());
  }

  onEnergy(cb) {
    this.energyCallback = cb;
  }

  onAudioChunk(cb) {
    this.audioChunkCallback = cb;
  }

  dispatchWavBuffer() {
    if (!this.audioChunkCallback || this.recordedSamplesL.length === 0) return;

    const wavBlob = this.encodeWAV(this.recordedSamplesL, this.recordedSamplesR, 22050);
    this.audioChunkCallback(wavBlob);
  }

  /**
   * Helper: Encodes stereo Float32 audio samples into a standard 16-bit PCM WAV Blob
   */
  encodeWAV(samplesL, samplesR, sampleRate) {
    const numChannels = 2;
    const numSamples = samplesL.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2 * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + numSamples * 2 * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw PCM) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, numChannels, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * numChannels * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, numChannels * 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, numSamples * 2 * 2, true);

    // Interleave left and right channels as 16-bit PCM
    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      let sL = Math.max(-1, Math.min(1, samplesL[i]));
      let sR = Math.max(-1, Math.min(1, samplesR[i]));

      view.setInt16(offset, sL < 0 ? sL * 0x8000 : sL * 0x7FFF, true);
      offset += 2;
      view.setInt16(offset, sR < 0 ? sR * 0x8000 : sR * 0x7FFF, true);
      offset += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  }
}

window.AudioEngine = AudioEngine;
