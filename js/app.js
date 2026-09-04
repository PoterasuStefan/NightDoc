/**
 * SoundSense Main Application Controller
 * Integrates:
 * - Local ONNX Acoustic Model (8 classes with Background & Conversation)
 * - Microsoft Foundry & Azure AI Content Understanding / Speech SDK
 * - Live Conversation Mode & Subtitles
 * - Dynamic Attention-Grabbing Green Bounce on Mode Switcher upon Speech Detection
 * - 360° Directional Radial Radar & TDoA Cross-Correlation
 */
document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Radars
  const mainRadar = new AcousticRadar('radarCanvas', {
    numPoints: 260,
    baseRadiusRatio: 0.58
  });

  const miniRadar = new AcousticRadar('miniRadarCanvas', {
    numPoints: 80,
    baseRadiusRatio: 0.65,
    isMini: true
  });

  // 2. Initialize Audio & WebSocket
  const audio = new AudioEngine();
  const socket = new BackendSocket();

  // DOM Elements - Views
  const standbyView = document.getElementById('standbyView');
  const conversationView = document.getElementById('conversationView');
  const goToConversationBtn = document.getElementById('goToConversationBtn');
  const backToStandbyBtn = document.getElementById('backToStandbyBtn');
  
  // DOM Elements - Center Radar Overlay
  const alertBadge = document.getElementById('alertBadge');
  const idleState = document.getElementById('idleState');
  const idleLabel = idleState ? idleState.querySelector('.idle-label') : null;
  const centerIcon = document.getElementById('centerIcon');
  const centerTitle = document.getElementById('centerTitle');
  const centerDirection = document.getElementById('centerDirection');
  const confidencePill = document.getElementById('confidencePill');

  // DOM Elements - Simulation Drawer
  const simToggleBtn = document.getElementById('simToggleBtn');
  const simulationDrawer = document.getElementById('simulationDrawer');
  const closeSimBtn = document.getElementById('closeSimBtn');
  const backdrop = document.getElementById('backdrop');
  
  const manualAngleSlider = document.getElementById('manualAngleSlider');
  const angleDisplay = document.getElementById('angleDisplay');
  const triggerCustomSoundBtn = document.getElementById('triggerCustomSoundBtn');
  const toggleAmbientNoiseBtn = document.getElementById('toggleAmbientNoiseBtn');
  const micToggleBtn = document.getElementById('micToggleBtn');
  const conversationEnergyBar = document.getElementById('conversationEnergyBar');

  // DOM Elements - HUD & File Upload
  const uploadAudioBtn = document.getElementById('uploadAudioBtn');
  const audioFileInput = document.getElementById('audioFileInput');
  const probList = document.getElementById('probList');
  const top3ProbList = document.getElementById('top3ProbList');

  // DOM Elements - Conversation Mode & Captions
  const captionsContainer = document.getElementById('captionsContainer');
  const activeCaptionBubble = document.getElementById('activeCaptionBubble');
  const liveSubtitleText = document.getElementById('liveSubtitleText');
  const clearCaptionsBtn = document.getElementById('clearCaptionsBtn');
  const speakerDirectionText = document.getElementById('speakerDirectionText');

  let alertResetTimer = null;
  let speechHighlightTimer = null;
  let speechRecognizer = null;

  // Metadata for all 8 ONNX model classes
  const CLASS_COLORS = {
    door_wood_knock: '#38BDF8', // Sky Blue
    car_horn: '#F97316',        // Orange
    crying_baby: '#EC4899',     // Pink
    dog: '#FACC15',             // Amber Yellow
    background: '#64748B',      // Slate Grey
    siren: '#EF4444',           // Red
    clapping: '#A855F7',        // Purple
    conversation: '#10B981'     // Emerald Green
  };

  const CLASS_NAMES = {
    door_wood_knock: 'Door Knock',
    car_horn: 'Car Horn',
    crying_baby: 'Crying Baby',
    dog: 'Dog Bark',
    background: 'Ambient Noise',
    siren: 'Emergency Siren',
    clapping: 'Clapping',
    conversation: 'Speech / Conversation'
  };

  const CLASS_ICONS = {
    door_wood_knock: '🚪',
    car_horn: '🚗',
    crying_baby: '👶',
    dog: '🐕',
    background: '🍃',
    siren: '🚑',
    clapping: '👏',
    conversation: '💬'
  };

  // Helper to Highlight "Switch to Conversation Mode" with Green Glow & Bounce
  function highlightConversationButton(highlight = true) {
    if (!goToConversationBtn) return;
    const btnText = goToConversationBtn.querySelector('.btn-text');

    if (highlight) {
      goToConversationBtn.classList.add('speech-detected');
      if (btnText) btnText.textContent = 'Speech Detected! Go to Conversation';

      if (speechHighlightTimer) clearTimeout(speechHighlightTimer);
      // Keep highlighted for 6 seconds unless user switches
      speechHighlightTimer = setTimeout(() => {
        highlightConversationButton(false);
      }, 6000);
    } else {
      goToConversationBtn.classList.remove('speech-detected');
      if (btnText) btnText.textContent = 'Conversation mode';
      if (speechHighlightTimer) clearTimeout(speechHighlightTimer);
    }
  }

  // Render Top 3 Live Confidence Score HUD
  function updateTop3HUD(probs = {}) {
    if (!top3ProbList) return;
    top3ProbList.innerHTML = '';

    const sorted = Object.entries(probs).filter(([_, v]) => v > 0).sort((a, b) => b[1] - a[1]);
    const top3 = sorted.slice(0, 3);

    if (top3.length === 0) {
      top3ProbList.innerHTML = `
        <div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 8px 0; font-weight: 500;">
          Awaiting sound detection...
        </div>
      `;
      return;
    }

    top3.forEach(([cls, val]) => {
      const color = CLASS_COLORS[cls] || '#FACC15';
      const name = CLASS_NAMES[cls] || cls.replace('_', ' ');
      const pct = typeof val === 'number' ? val.toFixed(1) : parseFloat(val || 0).toFixed(1);

      const row = document.createElement('div');
      row.className = 'top3-prob-row';
      row.innerHTML = `
        <div class="top3-prob-meta">
          <span class="top3-prob-name" style="color: ${color}">${name}</span>
          <span class="top3-prob-value">${pct}%</span>
        </div>
        <div class="top3-prob-track">
          <div class="top3-prob-fill" style="width: ${Math.min(100, Math.max(2, pct))}%; background: ${color}; box-shadow: 0 0 8px ${color}80;"></div>
        </div>
      `;
      top3ProbList.appendChild(row);
    });
  }

  // Render full probability bars in Drawer
  function updateProbabilitiesHUD(probs = {}) {
    updateTop3HUD(probs);

    if (!probList) return;
    probList.innerHTML = '';

    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const classes = sorted.length > 0 ? sorted : Object.keys(CLASS_COLORS).map(k => [k, 0]);

    classes.forEach(([cls, val]) => {
      const color = CLASS_COLORS[cls] || '#FACC15';
      const name = CLASS_NAMES[cls] || cls.replace('_', ' ');
      const pct = typeof val === 'number' ? val.toFixed(1) : parseFloat(val || 0).toFixed(1);

      const item = document.createElement('div');
      item.className = 'prob-item';
      item.innerHTML = `
        <div class="prob-meta">
          <span style="color: ${color}">${name}</span>
          <span>${pct}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${pct}%; background: ${color}"></div>
        </div>
      `;
      probList.appendChild(item);
    });
  }

  // Start with clean empty HUD on startup
  updateProbabilitiesHUD({});

  // Function to handle Ambient Noise / Conversation / Chatter
  function displayAmbient({ title, is_speech = false, probabilities = null, rms_energy = 0.08 }) {
    mainRadar.clearAlert();
    miniRadar.clearAlert();

    const isSpeechDetected = is_speech || 
      (title && (title.includes('Speech') || title.includes('Conversation'))) ||
      (probabilities && probabilities.conversation > 30.0);

    // If speech is detected, highlight the Switch to Conversation button with green bounce!
    if (isSpeechDetected) {
      highlightConversationButton(true);
    }

    // Activate subtle grey waves around perimeter matching sound energy
    const waveAmp = Math.min(2.5, Math.max(1.0, (rms_energy || 0.08) * 20));
    mainRadar.setAmbientNoise(true, waveAmp);

    alertBadge.classList.add('hidden');
    idleState.classList.remove('hidden');
    
    if (idleLabel) {
      if (isSpeechDetected) {
        idleLabel.textContent = '💬 Speech / Conversation (Listening...)';
      } else {
        idleLabel.textContent = '🍃 Ambient chatter (Listening...)';
      }
    }

    if (probabilities) {
      updateProbabilitiesHUD(probabilities);
    }

    if (alertResetTimer) clearTimeout(alertResetTimer);
    alertResetTimer = setTimeout(() => {
      if (!audio.isListening) {
        mainRadar.setAmbientNoise(false, 0.0);
      }
      if (idleLabel) {
        idleLabel.textContent = audio.isListening ? 'Listening for sounds...' : 'Microphone idle';
      }
    }, 4000);
  }

  // Function to show dynamic special sound alert
  function displayAlert({ title, direction, angle, icon, color = '#FACC15', confidence = 94, intensity = 1.0, is_speech = false, probabilities = null }) {
    mainRadar.triggerAlert(angle, intensity, color);
    miniRadar.triggerAlert(angle, intensity, color);

    const isSpeechDetected = is_speech || (title && (title.includes('Speech') || title.includes('Conversation')));
    if (isSpeechDetected) {
      highlightConversationButton(true);
    }

    centerIcon.textContent = icon || '🔊';
    centerTitle.textContent = title;
    centerDirection.textContent = direction;
    centerDirection.style.color = color;
    confidencePill.textContent = `${confidence}% match`;
    confidencePill.style.color = color;
    confidencePill.style.borderColor = `${color}4D`;
    confidencePill.style.background = `${color}26`;

    alertBadge.classList.remove('hidden');
    idleState.classList.add('hidden');

    if (probabilities) {
      updateProbabilitiesHUD(probabilities);
    }

    if (speakerDirectionText) {
      speakerDirectionText.textContent = `Speaker at ${Math.round(angle)}° (${direction})`;
    }

    if (alertResetTimer) clearTimeout(alertResetTimer);
    alertResetTimer = setTimeout(() => {
      alertBadge.classList.add('hidden');
      idleState.classList.remove('hidden');
      if (idleLabel) {
        idleLabel.textContent = audio.isListening ? 'Listening for sounds...' : 'Microphone idle';
      }
      mainRadar.clearAlert();
      miniRadar.clearAlert();
      if (!audio.isListening) {
        mainRadar.setAmbientNoise(false, 0.0);
      }
    }, 4500);
  }

  // ==========================================================================
  // CONVERSATION MODE: Captions & Subtitles Engine
  // ==========================================================================
  let lastAddedCaption = { text: '', time: 0 };

  function appendFinalCaption(text, angle = 0, source = 'Live Voice') {
    if (!text || !text.trim() || !captionsContainer) return;
    const trimmed = text.trim();

    const now = new Date();
    const nowMs = now.getTime();

    // Prevent duplicate bubbles if the exact same text was added within 2.5 seconds
    if (lastAddedCaption.text.toLowerCase() === trimmed.toLowerCase() && (nowMs - lastAddedCaption.time) < 2500) {
      return;
    }
    lastAddedCaption = { text: trimmed, time: nowMs };

    const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

    const bubble = document.createElement('div');
    bubble.className = 'caption-bubble past';
    bubble.innerHTML = `
      <span class="caption-timestamp">${timeStr} &bull; ${source}</span>
      <p class="caption-text">"${trimmed}"</p>
    `;
    
    if (activeCaptionBubble) {
      captionsContainer.insertBefore(bubble, activeCaptionBubble);
    } else {
      captionsContainer.appendChild(bubble);
    }

    // Reset active subtitle placeholder
    if (liveSubtitleText) {
      liveSubtitleText.innerHTML = `
        <span class="confirmed-text" style="color: var(--text-muted); font-size: 1.05rem;">Listening for speech / Vorbiți la microfon...</span>
        <span class="cursor-blink">|</span>
      `;
    }

    captionsContainer.scrollTop = captionsContainer.scrollHeight;

    if (speakerDirectionText && angle !== undefined) {
      speakerDirectionText.textContent = `Speaker at ${Math.round(angle)}°`;
    }
  }

  function updateInterimCaption(text) {
    if (!liveSubtitleText) return;
    liveSubtitleText.innerHTML = `
      <span class="confirmed-text">${text}</span>
      <span class="cursor-blink">|</span>
    `;
    if (captionsContainer) {
      captionsContainer.scrollTop = captionsContainer.scrollHeight;
    }
  }

  if (clearCaptionsBtn && captionsContainer) {
    clearCaptionsBtn.addEventListener('click', () => {
      const pastBubbles = captionsContainer.querySelectorAll('.caption-bubble.past');
      pastBubbles.forEach(b => b.remove());
      if (liveSubtitleText) {
        liveSubtitleText.innerHTML = `
          <span class="confirmed-text" style="color: var(--text-muted); font-size: 1.05rem;">Listening for speech / Vorbiți la microfon...</span>
          <span class="cursor-blink">|</span>
        `;
      }
    });
  }

  // Quick Speech Test Input Form
  const speechTestForm = document.getElementById('speechTestForm');
  const speechTextInput = document.getElementById('speechTextInput');
  if (speechTestForm && speechTextInput) {
    speechTestForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const val = speechTextInput.value.trim();
      if (!val) return;

      const angle = mainRadar.activeAlert ? mainRadar.activeAlert.angle : 0;
      appendFinalCaption(val, angle, 'Speech Test');
      
      socket.sendText(JSON.stringify({
        type: "live_speech",
        text: val,
        is_final: true,
        angle: angle
      }));

      speechTextInput.value = '';
    });
  }

  // Robust In-Browser Speech Recognition Engine (Web Speech API with Auto-Restart)
  let speechLang = 'ro-RO';
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

  function initSpeechEngine() {
    if (!SpeechRec) {
      console.warn('Web Speech API not supported in this browser.');
      return null;
    }

    try {
      const recognizer = new SpeechRec();
      recognizer.continuous = true;
      recognizer.interimResults = true;
      recognizer.lang = speechLang;

      recognizer.onstart = () => {
        const badge = document.getElementById('liveStatusBadge');
        if (badge) badge.textContent = `🎙️ Live (${speechLang === 'ro-RO' ? 'Română' : 'English'})`;
      };

      recognizer.onresult = (event) => {
        let interim = '';
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interim += event.results[i][0].transcript;
          }
        }

        if (interim) {
          updateInterimCaption(interim);
        }
        if (finalTranscript) {
          const angle = mainRadar.activeAlert ? mainRadar.activeAlert.angle : 0;
          appendFinalCaption(finalTranscript, angle, 'Live Voice');
          socket.sendText(JSON.stringify({
            type: "live_speech",
            text: finalTranscript,
            is_final: true,
            angle: angle
          }));
        }
      };

      recognizer.onerror = (e) => {
        console.warn('Speech Recognition warning:', e.error);
        if (e.error === 'language-not-supported' || e.error === 'network') {
          if (speechLang === 'ro-RO') {
            speechLang = navigator.language || 'en-US';
            recognizer.lang = speechLang;
          }
        }
      };

      recognizer.onend = () => {
        // Auto-restart recognition if still in conversation mode and mic is listening
        if (conversationView.classList.contains('active-view') && audio.isListening) {
          setTimeout(() => {
            try {
              recognizer.start();
            } catch (err) {}
          }, 300);
        }
      };

      return recognizer;
    } catch (err) {
      console.warn('Failed to init speech recognizer:', err);
      return null;
    }
  }

  speechRecognizer = initSpeechEngine();

  function startLiveSpeech() {
    if (!speechRecognizer) {
      speechRecognizer = initSpeechEngine();
    }
    if (speechRecognizer) {
      try {
        speechRecognizer.start();
      } catch (e) {
        // Already started or busy
      }
    }
  }

  function stopLiveSpeech() {
    if (speechRecognizer) {
      try {
        speechRecognizer.stop();
      } catch (e) {}
    }
  }

  // ==========================================================================
  // Mode Switching
  // ==========================================================================
  goToConversationBtn.addEventListener('click', () => {
    highlightConversationButton(false);
    standbyView.classList.remove('active-view');
    conversationView.classList.add('active-view');
    miniRadar.resize();
    if (audio.isListening) {
      startLiveSpeech();
    }
  });

  backToStandbyBtn.addEventListener('click', () => {
    conversationView.classList.remove('active-view');
    standbyView.classList.add('active-view');
    mainRadar.resize();
    stopLiveSpeech();
  });

  // Drawer Toggle
  function toggleDrawer(open) {
    if (open) {
      simulationDrawer.classList.add('open');
      backdrop.classList.add('open');
    } else {
      simulationDrawer.classList.remove('open');
      backdrop.classList.remove('open');
    }
  }

  simToggleBtn.addEventListener('click', () => toggleDrawer(true));
  closeSimBtn.addEventListener('click', () => toggleDrawer(false));
  backdrop.addEventListener('click', () => toggleDrawer(false));

  // Preset Trigger Click Handlers (For the 8 ONNX classes)
  document.querySelectorAll('.sim-card').forEach(card => {
    card.addEventListener('click', () => {
      const soundCls = card.getAttribute('data-sound') || 'crying_baby';
      const title = card.getAttribute('data-title');
      const direction = card.getAttribute('data-direction');
      const angle = parseFloat(card.getAttribute('data-angle'));
      const icon = card.getAttribute('data-icon');
      const color = card.getAttribute('data-color');
      const intensity = parseFloat(card.getAttribute('data-intensity') || '1.0');

      if (soundCls === 'background' || soundCls === 'conversation') {
        const isConv = soundCls === 'conversation';
        const probs = {
          door_wood_knock: 0.1,
          car_horn: 0.2,
          crying_baby: 0.1,
          dog: 0.1,
          background: isConv ? 0.5 : 98.9,
          conversation: isConv ? 98.0 : 0.0,
          siren: 0.0,
          clapping: 0.3
        };
        displayAmbient({
          title,
          is_speech: isConv,
          probabilities: probs,
          rms_energy: 0.12
        });

        // If conversation clicked in simulator, add sample subtitle ready in conversation mode
        if (isConv) {
          appendFinalCaption("Salut! Te aud clar, sistemul acustic funcționează perfect.", angle, "Microsoft Foundry Simulation");
        }
      } else {
        const conf = Math.floor(82 + Math.random() * 17);
        const mockProbs = {
          door_wood_knock: 0.1,
          car_horn: 0.1,
          crying_baby: 0.1,
          dog: 0.1,
          background: 0.5,
          conversation: 0.0,
          siren: 0.1,
          clapping: 0.1
        };
        mockProbs[soundCls] = conf;

        displayAlert({
          title,
          direction: `${direction} (${angle}°)`,
          angle,
          icon,
          color,
          intensity,
          confidence: conf,
          probabilities: mockProbs
        });
      }

      toggleDrawer(false);
    });
  });

  // Manual Angle Slider
  manualAngleSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    angleDisplay.textContent = `${val}°`;
  });

  triggerCustomSoundBtn.addEventListener('click', () => {
    const angle = parseFloat(manualAngleSlider.value);
    let dir = 'in front';
    if (angle > 20 && angle < 160) dir = 'on your right';
    else if (angle >= 160 && angle <= 200) dir = 'behind you';
    else if (angle > 200 && angle < 340) dir = 'on your left';

    displayAlert({
      title: 'Acoustic Sound Peak',
      direction: `${dir} (${angle}°)`,
      angle: angle,
      icon: '⚡',
      color: '#38BDF8',
      intensity: 1.0,
      confidence: 91
    });
    toggleDrawer(false);
  });

  // Toggle Ambient Chatter Button
  toggleAmbientNoiseBtn.addEventListener('click', () => {
    const isActive = !mainRadar.options.ambientActive;
    mainRadar.setAmbientNoise(isActive, isActive ? 1.2 : 0.0);
    toggleAmbientNoiseBtn.textContent = `Ambient Chatter: ${isActive ? 'ON' : 'OFF'}`;
  });

  // Live Audio File Upload & Classification
  if (uploadAudioBtn && audioFileInput) {
    uploadAudioBtn.addEventListener('click', () => audioFileInput.click());

    audioFileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      try {
        uploadAudioBtn.querySelector('span').textContent = 'Analyzing with ONNX & Foundry...';
        const res = await fetch('/api/classify-audio', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        
        if (data.success) {
          if (data.is_alert) {
            displayAlert({
              title: data.title,
              direction: data.direction_label,
              angle: data.direction_angle,
              icon: data.icon,
              color: data.color,
              confidence: data.confidence,
              probabilities: data.probabilities,
              intensity: 1.2,
              is_speech: data.is_speech
            });
          } else {
            displayAmbient({
              title: data.title,
              is_speech: data.is_speech,
              probabilities: data.probabilities,
              rms_energy: data.rms_energy
            });
          }
          toggleDrawer(false);
        } else {
          alert('Classification failed: ' + (data.error || 'Unknown error'));
        }
      } catch (err) {
        console.error('File classification error:', err);
      } finally {
        uploadAudioBtn.querySelector('span').textContent = 'Choose Audio File to Classify';
        audioFileInput.value = '';
      }
    });
  }

  // Live Microphone Integration
  micToggleBtn.addEventListener('click', async () => {
    if (!audio.isListening) {
      const ok = await audio.startMicrophone();
      if (ok) {
        micToggleBtn.classList.add('active');
        micToggleBtn.querySelector('.btn-label').textContent = 'Mic Active';
        if (idleLabel) idleLabel.textContent = 'Listening for sounds...';
        if (conversationView.classList.contains('active-view')) {
          startLiveSpeech();
        }
      }
    } else {
      audio.stopMicrophone();
      stopLiveSpeech();
      micToggleBtn.classList.remove('active');
      micToggleBtn.querySelector('.btn-label').textContent = 'Mic Live';
      mainRadar.setAmbientNoise(false, 0.0);
      mainRadar.clearAlert();
      if (idleLabel) idleLabel.textContent = 'Microphone idle';
    }
  });

  // Real-time microphone audio energy monitoring
  audio.onEnergy((energy, buffer) => {
    if (conversationEnergyBar) {
      conversationEnergyBar.style.width = `${Math.min(100, energy * 200)}%`;
    }

    if (audio.isListening && !mainRadar.activeAlert.active) {
      if (energy > 0.03) {
        mainRadar.setAmbientNoise(true, Math.min(2.5, energy * 3.5));
      } else {
        mainRadar.setAmbientNoise(false, 0.0);
      }
    }
  });

  // Stream live WAV audio chunks to backend WebSocket
  audio.onAudioChunk((blob) => {
    blob.arrayBuffer().then(buffer => {
      socket.sendAudioChunk(buffer);
    });
  });

  // Connect WebSocket & Listeners
  socket.on('status', (status) => {
    const el = document.getElementById('connectionStatus');
    if (el) el.textContent = status.label;
  });

  socket.on('ambient', (payload) => {
    displayAmbient(payload);
  });

  socket.on('alert', (payload) => {
    displayAlert(payload);
  });

  // Listen for Azure Foundry / Azure Speech Subtitles from Backend
  socket.on('transcription', (payload) => {
    if (payload.is_final) {
      appendFinalCaption(payload.text, payload.angle, payload.source || 'Microsoft Foundry');
    } else {
      updateInterimCaption(payload.text);
    }
  });

  socket.connect();
});
