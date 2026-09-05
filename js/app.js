/**
 * NightDoc / RespiSense AI – Main Application State & View Controller
 * Integrates ONNX Edge AI, HL7 FHIR Formatter, and Azure AI Foundry
 */

class NightDocApp {
  constructor() {
    this.wsClient = null;
    this.audioEngine = null;
    this.radar = null;

    this.activeView = 'bedside'; // 'bedside' | 'clinical'
    this.latestFhirObservation = null;
    this.telemetryData = {
      cough_count: 0,
      coughs_per_hour: 0.0,
      breathing_count: 0,
      snoring_count: 0,
      sneezing_count: 0,
      crying_count: 0,
      disturbance_score: 0,
      total_events: 0,
      recent_events: []
    };

    this.classes = ['coughing', 'breathing', 'snoring', 'sneezing', 'crying_baby', 'conversation', 'background'];
    this.classMeta = {
      coughing: { title: 'Coughing (Tuse)', icon: '🫁', color: '#EF4444', snomed: '263731006', loinc: '8687-6' },
      breathing: { title: 'Breathing (Respirație)', icon: '🌬️', color: '#38BDF8', snomed: '56018004', loinc: '9279-1' },
      snoring: { title: 'Snoring (Sforăit)', icon: '💤', color: '#F59E0B', snomed: '271600006', loinc: '93832-4' },
      sneezing: { title: 'Sneezing (Strănut)', icon: '🤧', color: '#A855F7', snomed: '16962002', loinc: '8688-4' },
      crying_baby: { title: 'Pediatric Distress (Plâns)', icon: '👶', color: '#EC4899', snomed: '271633008', loinc: '72144-9' },
      conversation: { title: 'Speech / Voice', icon: '💬', color: '#10B981', snomed: '286369001', loinc: 'LA11874-7' },
      background: { title: 'Ambient Baseline', icon: '🍃', color: '#64748B', snomed: '162076009', loinc: 'LA28669-9' }
    };

    this.initElements();
    this.initModules();
    this.attachEventListeners();
  }

  initElements() {
    this.viewBedsideBtn = document.getElementById('viewBedsideBtn');
    this.viewClinicalBtn = document.getElementById('viewClinicalBtn');
    this.bedsideView = document.getElementById('bedsideView');
    this.clinicalView = document.getElementById('clinicalView');

    this.micToggleBtn = document.getElementById('micToggleBtn');
    this.simToggleBtn = document.getElementById('simToggleBtn');
    this.simulationDrawer = document.getElementById('simulationDrawer');
    this.closeSimBtn = document.getElementById('closeSimBtn');
    this.backdrop = document.getElementById('backdrop');

    // Bedside UI
    this.alertBadge = document.getElementById('alertBadge');
    this.centerIcon = document.getElementById('centerIcon');
    this.centerTitle = document.getElementById('centerTitle');
    this.centerDirection = document.getElementById('centerDirection');
    this.confidencePill = document.getElementById('confidencePill');
    this.idleState = document.getElementById('idleState');
    this.top3ProbList = document.getElementById('top3ProbList');

    this.quickCoughCount = document.getElementById('quickCoughCount');
    this.quickCoughRate = document.getElementById('quickCoughRate');
    this.quickDisturbanceScore = document.getElementById('quickDisturbanceScore');

    // Clinical UI
    this.clinCoughCount = document.getElementById('clinCoughCount');
    this.clinCoughRate = document.getElementById('clinCoughRate');
    this.clinSnoreCount = document.getElementById('clinSnoreCount');
    this.clinBreatheCount = document.getElementById('clinBreatheCount');
    this.clinDisturbanceScore = document.getElementById('clinDisturbanceScore');
    this.clinRiskLevel = document.getElementById('clinRiskLevel');

    this.generateAiReportBtn = document.getElementById('generateAiReportBtn');
    this.aiReportBox = document.getElementById('aiReportBox');
    this.aiReportText = document.getElementById('aiReportText');
    this.aiReportBadge = document.getElementById('aiReportBadge');
    this.aiReportTime = document.getElementById('aiReportTime');
    this.aiSnomedTags = document.getElementById('aiSnomedTags');

    this.viewFhirModalBtn = document.getElementById('viewFhirModalBtn');
    this.exportFhirBundleBtn = document.getElementById('exportFhirBundleBtn');
    this.fhirCodeSnippet = document.getElementById('fhirCodeSnippet');
    this.clinicalEventsTableBody = document.getElementById('clinicalEventsTableBody');
    this.resetSessionBtn = document.getElementById('resetSessionBtn');

    // FHIR Modal
    this.fhirModal = document.getElementById('fhirModal');
    this.fhirModalCode = document.getElementById('fhirModalCode');
    this.closeFhirModalBtn = document.getElementById('closeFhirModalBtn');
    this.closeFhirModalFooterBtn = document.getElementById('closeFhirModalFooterBtn');
    this.copyFhirJsonBtn = document.getElementById('copyFhirJsonBtn');

    // Audio file upload
    this.audioFileInput = document.getElementById('audioFileInput');
    this.uploadAudioBtn = document.getElementById('uploadAudioBtn');
    this.probList = document.getElementById('probList');
  }

  initModules() {
    this.radar = new AcousticRadar('radarCanvas');
    this.audioEngine = new AudioEngine((wavBlob) => this.onAudioChunkCaptured(wavBlob));
    this.wsClient = new WebSocketClient((msg) => this.onWebSocketMessage(msg));
    this.wsClient.connect();
  }

  attachEventListeners() {
    // Mode Switcher
    this.viewBedsideBtn.addEventListener('click', () => this.switchView('bedside'));
    this.viewClinicalBtn.addEventListener('click', () => this.switchView('clinical'));

    // Mic toggle
    this.micToggleBtn.addEventListener('click', () => this.toggleMicrophone());

    // Simulator drawer
    this.simToggleBtn.addEventListener('click', () => this.openSimulationDrawer());
    this.closeSimBtn.addEventListener('click', () => this.closeSimulationDrawer());
    this.backdrop.addEventListener('click', () => this.closeSimulationDrawer());

    // Preset biomarker simulator triggers
    document.querySelectorAll('.sim-card').forEach((card) => {
      card.addEventListener('click', (e) => {
        const soundClass = card.getAttribute('data-sound');
        const angle = parseFloat(card.getAttribute('data-angle')) || 0;
        this.triggerSimulation(soundClass, angle);
      });
    });

    // File upload
    this.uploadAudioBtn.addEventListener('click', () => this.audioFileInput.click());
    this.audioFileInput.addEventListener('change', (e) => this.handleFileUpload(e));

    // Azure AI Report
    this.generateAiReportBtn.addEventListener('click', () => this.generateAzureClinicalReport());

    // FHIR Modal
    this.viewFhirModalBtn.addEventListener('click', () => this.openFhirModal());
    this.closeFhirModalBtn.addEventListener('click', () => this.closeFhirModal());
    this.closeFhirModalFooterBtn.addEventListener('click', () => this.closeFhirModal());
    this.copyFhirJsonBtn.addEventListener('click', () => this.copyFhirJson());

    // Export FHIR Bundle
    this.exportFhirBundleBtn.addEventListener('click', () => this.exportFhirBundle());

    // Reset session
    this.resetSessionBtn.addEventListener('click', () => this.resetSession());
  }

  switchView(mode) {
    this.activeView = mode;
    if (mode === 'bedside') {
      this.viewBedsideBtn.classList.add('active');
      this.viewClinicalBtn.classList.remove('active');
      this.bedsideView.classList.add('active-view');
      this.clinicalView.classList.remove('active-view');
    } else {
      this.viewClinicalBtn.classList.add('active');
      this.viewBedsideBtn.classList.remove('active');
      this.clinicalView.classList.add('active-view');
      this.bedsideView.classList.remove('active-view');
    }
  }

  openSimulationDrawer() {
    this.simulationDrawer.classList.add('open');
    this.backdrop.classList.add('visible');
  }

  closeSimulationDrawer() {
    this.simulationDrawer.classList.remove('open');
    this.backdrop.classList.remove('visible');
  }

  async toggleMicrophone() {
    if (this.audioEngine.isRecording) {
      await this.audioEngine.stop();
      this.micToggleBtn.classList.remove('active');
      this.micToggleBtn.querySelector('.btn-label').textContent = 'Mic Live';
    } else {
      const ok = await this.audioEngine.start();
      if (ok) {
        this.micToggleBtn.classList.add('active');
        this.micToggleBtn.querySelector('.btn-label').textContent = 'Listening...';
      }
    }
  }

  onAudioChunkCaptured(wavBlob) {
    if (this.wsClient && this.wsClient.isConnected) {
      wavBlob.arrayBuffer().then((buf) => {
        this.wsClient.sendBinary(buf);
      });
    }
  }

  onWebSocketMessage(msg) {
    const type = msg.type;
    const payload = msg.payload || {};

    if (type === 'status') {
      document.getElementById('connectionStatus').textContent = payload.label || 'Connected';
      if (payload.telemetry) {
        this.updateTelemetry(payload.telemetry);
      }
    } else if (type === 'biomarker_event') {
      this.handleBiomarkerEvent(payload);
    } else if (type === 'transcription') {
      console.log('Azure Speech Transcription:', payload.text);
    }
  }

  handleBiomarkerEvent(data) {
    const predicted = data.predicted_class || 'background';
    const meta = this.classMeta[predicted] || this.classMeta.background;
    const conf = data.confidence || 0;
    const isAlert = data.is_alert;
    const angle = data.direction_angle || 0;

    // Trigger radar ripple
    if (this.radar) {
      this.radar.triggerBurst({
        angle: angle,
        color: meta.color,
        intensity: Math.min(1.5, Math.max(0.6, conf / 60.0)),
        isAlert: isAlert
      });
    }

    // Update Bedside Overlay
    if (isAlert || conf > 40.0) {
      this.alertBadge.classList.remove('hidden');
      this.idleState.classList.add('hidden');
      this.centerIcon.textContent = meta.icon;
      this.centerTitle.textContent = meta.title;
      this.centerDirection.textContent = `Spatial Angle: ${Math.round(angle)}° | SNOMED: ${meta.snomed}`;
      this.confidencePill.textContent = `${conf.toFixed(1)}% ONNX Confidence`;
      this.confidencePill.style.backgroundColor = meta.color;
    } else {
      setTimeout(() => {
        this.alertBadge.classList.add('hidden');
        this.idleState.classList.remove('hidden');
      }, 2500);
    }

    // Update Probabilities HUD
    if (data.probabilities) {
      this.renderProbabilities(data.probabilities);
    }

    // Save FHIR Observation
    if (data.fhir_resource) {
      this.latestFhirObservation = data.fhir_resource;
      this.fhirCodeSnippet.textContent = JSON.stringify(data.fhir_resource, null, 2);
    }

    // Refresh telemetry stats from backend
    this.fetchTelemetryStats();
  }

  renderProbabilities(probs) {
    const sorted = Object.entries(probs)
      .sort((a, b) => b[1] - a[1]);

    const top3 = sorted.slice(0, 3);
    this.top3ProbList.innerHTML = top3.map(([cls, val]) => {
      const m = this.classMeta[cls] || { title: cls, icon: '•', color: '#64748B' };
      return `
        <div class="prob-row">
          <div class="prob-row-header">
            <span class="prob-name">${m.icon} ${m.title}</span>
            <span class="prob-val">${val}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: ${Math.min(100, Math.max(4, val))}%; background-color: ${m.color}"></div>
          </div>
        </div>
      `;
    }).join('');

    // All probabilities list in drawer
    this.probList.innerHTML = sorted.map(([cls, val]) => {
      const m = this.classMeta[cls] || { title: cls, icon: '•', color: '#64748B' };
      return `
        <div class="prob-row">
          <div class="prob-row-header">
            <span class="prob-name">${m.icon} ${m.title}</span>
            <span class="prob-val">${val}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: ${Math.min(100, Math.max(2, val))}%; background-color: ${m.color}"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  async fetchTelemetryStats() {
    try {
      const res = await fetch('/api/telemetry/stats');
      if (res.ok) {
        const stats = await res.json();
        this.updateTelemetry(stats);
      }
    } catch (e) {
      console.warn('Could not fetch telemetry stats:', e);
    }
  }

  updateTelemetry(stats) {
    this.telemetryData = stats;

    // Bedside Footer
    this.quickCoughCount.textContent = stats.cough_count || 0;
    this.quickCoughRate.textContent = `${stats.coughs_per_hour || 0.0} / hr`;
    const dist = stats.disturbance_score || 0;
    this.quickDisturbanceScore.textContent = `${dist}% (${dist > 50 ? 'High' : dist > 20 ? 'Medium' : 'Low'})`;
    this.quickDisturbanceScore.className = `stat-value ${dist > 50 ? 'status-high' : dist > 20 ? 'status-med' : 'status-low'}`;

    // Clinical Dashboard
    this.clinCoughCount.textContent = stats.cough_count || 0;
    this.clinCoughRate.textContent = `${stats.coughs_per_hour || 0.0} episodes / hr`;
    this.clinSnoreCount.textContent = stats.snoring_count || 0;
    this.clinBreatheCount.textContent = stats.breathing_count || 0;
    this.clinDisturbanceScore.textContent = `${dist}%`;
    this.clinRiskLevel.textContent = dist > 50 ? 'Critical Exacerbation Alert' : dist > 20 ? 'Moderate Airway Obstruction' : 'Normal Baseline';

    // Update Events Table
    if (stats.recent_events && stats.recent_events.length > 0) {
      this.renderEventsTable(stats.recent_events);
    }
  }

  renderEventsTable(events) {
    const reversed = [...events].reverse();
    this.clinicalEventsTableBody.innerHTML = reversed.map((ev) => {
      const meta = this.classMeta[ev.predicted_class] || this.classMeta.background;
      const timeStr = new Date(ev.timestamp).toLocaleTimeString();
      const critClass = ev.criticality === 'high' ? 'badge-crit-high' : ev.criticality === 'medium' ? 'badge-crit-med' : 'badge-crit-low';
      
      return `
        <tr>
          <td><span class="font-mono text-muted">${timeStr}</span></td>
          <td><b>${meta.icon} ${meta.title}</b></td>
          <td><span class="badge" style="background:${meta.color}22; color:${meta.color}">${ev.predicted_class}</span></td>
          <td><b>${ev.confidence}%</b></td>
          <td><span class="font-mono text-xs">LOINC ${ev.loinc} | SNOMED ${ev.snomed}</span></td>
          <td><span class="badge ${critClass}">${ev.criticality.toUpperCase()}</span></td>
          <td><span class="font-mono text-muted">${ev.inference_ms || 10}ms</span></td>
        </tr>
      `;
    }).join('');
  }

  triggerSimulation(soundClass, angle = 0) {
    if (this.wsClient && this.wsClient.isConnected) {
      this.wsClient.sendJson({
        type: 'test_sound',
        class: soundClass,
        angle: angle
      });
    }
  }

  async handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      this.uploadAudioBtn.innerHTML = `<span>Classifying ONNX...</span>`;
      const res = await fetch('/api/classify-audio', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      this.uploadAudioBtn.innerHTML = `<span>Choose ESC-50 / Real Audio WAV</span>`;

      if (data.success) {
        this.handleBiomarkerEvent(data);
      } else {
        alert('Eroare la clasificare audio: ' + (data.error || 'Necunoscut'));
      }
    } catch (err) {
      this.uploadAudioBtn.innerHTML = `<span>Choose ESC-50 / Real Audio WAV</span>`;
      alert('Eroare la comunicarea cu serverul: ' + err.message);
    }
  }

  async generateAzureClinicalReport() {
    this.aiReportBox.classList.remove('hidden');
    this.aiReportText.innerHTML = `
      <div class="ai-loading">
        <span class="spinner"></span>
        <span>Calling <b>Microsoft Azure AI Foundry</b> (Azure OpenAI GPT-4o-mini)...</span>
      </div>
    `;

    try {
      const res = await fetch('/api/clinical-summary', { method: 'POST' });
      const data = await res.json();

      if (data.success) {
        this.aiReportBadge.textContent = data.provider || 'Azure AI Foundry';
        this.aiReportTime.textContent = new Date().toLocaleTimeString();
        this.aiReportText.innerHTML = `
          <div class="report-section">
            <h4>📋 Sinteză Clinică:</h4>
            <p>${data.summary_ro || data.summary_en}</p>
          </div>
          <div class="report-section">
            <h4>💡 Recomandare Pneumologie:</h4>
            <p><b>${data.actionable_recommendation || 'Continuare monitorizare neinvazivă nocturnă.'}</b></p>
          </div>
          <div class="report-section">
            <h4>🚨 Evaluare Risc: <span class="badge ${data.risk_level?.includes('High') || data.risk_level?.includes('Ridicat') ? 'badge-crit-high' : 'badge-crit-low'}">${data.risk_level || 'Scăzut'}</span></h4>
          </div>
        `;

        if (data.recommended_snomed) {
          this.aiSnomedTags.innerHTML = data.recommended_snomed.map(code => `<span class="snomed-pill">🔖 ${code}</span>`).join('');
        }
      } else {
        this.aiReportText.textContent = 'Eroare raport Azure AI: ' + (data.error || 'Verificați configurarea Azure');
      }
    } catch (e) {
      this.aiReportText.textContent = 'Eroare la conexiunea cu serverul: ' + e.message;
    }
  }

  openFhirModal() {
    this.fhirModal.classList.remove('hidden');
    if (this.latestFhirObservation) {
      this.fhirModalCode.textContent = JSON.stringify(this.latestFhirObservation, null, 2);
    } else {
      this.fhirModalCode.textContent = '// Nicio observație generată încă. Declanșați un eveniment sau porniți microfonul.';
    }
  }

  closeFhirModal() {
    this.fhirModal.classList.add('hidden');
  }

  copyFhirJson() {
    const text = this.fhirModalCode.textContent;
    navigator.clipboard.writeText(text);
    this.copyFhirJsonBtn.textContent = 'Copied!';
    setTimeout(() => { this.copyFhirJsonBtn.textContent = 'Copy JSON'; }, 2000);
  }

  async exportFhirBundle() {
    try {
      const res = await fetch('/api/fhir/export-bundle', { method: 'POST' });
      const bundle = await res.json();
      
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `FHIR_Bundle_RespiSense_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert('Eroare la exportul FHIR Bundle: ' + e.message);
    }
  }

  async resetSession() {
    if (confirm('Sigur doriți să resetați telemetria sesiunii curente?')) {
      await fetch('/api/telemetry/reset', { method: 'POST' });
      this.fetchTelemetryStats();
      this.clinicalEventsTableBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Sesiune resetată. Așteptăm noi detecții...</td></tr>`;
      this.aiReportBox.classList.add('hidden');
    }
  }
}

// Bootstrap
window.addEventListener('DOMContentLoaded', () => {
  window.app = new NightDocApp();
});
