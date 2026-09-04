/**
 * Radial Acoustic Waveform & Directional Radar Visualizer
 * Implements 360-degree directional sound waves, ambient chatter oscillations,
 * and highlighted alert peaks + directional pointer.
 */
class AcousticRadar {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    
    this.ctx = this.canvas.getContext('2d');
    this.options = {
      isMini: options.isMini || false,
      numPoints: options.numPoints || 240, // Wave resolution around circumference
      baseRadiusRatio: options.baseRadiusRatio || 0.58, // Base circle radius ratio
      ambientActive: true,
      ...options
    };

    // Wave state per angular point
    this.points = new Float32Array(this.options.numPoints);
    this.targetPoints = new Float32Array(this.options.numPoints);

    this.options.ambientActive = false; // Start inactive until audio/chatter is present
    this.options.ambientIntensity = 0.0;

    // Active alert state - start completely inactive on startup
    this.activeAlert = {
      angle: 0,
      intensity: 0.0,
      color: '#FACC15',
      glowColor: 'rgba(250, 204, 21, 0.5)',
      active: false,
      decayTimer: 0,
      decayDuration: 3000 // ms
    };

    // Arrow orientation state with smooth damping
    this.currentArrowAngle = 0;
    this.targetArrowAngle = 0;

    // Time ticker for organic wave ripples
    this.time = 0;
    this.pixelRatio = window.devicePixelRatio || 1;

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.startLoop();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;

    // Handle high DPI displays for ultra crisp lines
    this.canvas.width = this.width * this.pixelRatio;
    this.canvas.height = this.height * this.pixelRatio;
    this.ctx.scale(this.pixelRatio, this.pixelRatio);

    this.centerX = this.width / 2;
    this.centerY = this.height / 2;
    this.baseRadius = Math.min(this.width, this.height) * this.options.baseRadiusRatio * 0.5;
  }

  /**
   * Trigger an alert sound in a given direction
   * @param {number} angleDeg - Angle in degrees (0 = front, 90 = right, 180 = back, 270 = left)
   * @param {number} intensity - Magnitude from 0.1 to 1.5
   * @param {string} color - Hex or rgb color for the highlight waves
   */
  triggerAlert(angleDeg, intensity = 1.0, color = '#FACC15') {
    this.activeAlert = {
      angle: angleDeg,
      intensity: intensity,
      color: color,
      glowColor: color.startsWith('#') ? `${color}66` : color,
      active: true,
      decayTimer: performance.now(),
      decayDuration: 3500
    };

    // Set target angle for arrow in radians (converted so 0° is 12 o'clock / top)
    const rad = (angleDeg - 90) * (Math.PI / 180);
    this.targetArrowAngle = rad;
  }

  setAmbientNoise(enabled) {
    this.options.ambientActive = enabled;
  }

  update(deltaTime) {
    this.time += 0.04;

    // Arrow smooth rotation interpolation (shortest path)
    let diff = this.targetArrowAngle - this.currentArrowAngle;
    while (diff < -Math.PI) diff += Math.PI * 2;
    while (diff > Math.PI) diff -= Math.PI * 2;
    this.currentArrowAngle += diff * 0.12;

    // Alert decay check
    if (this.activeAlert.active) {
      const elapsed = performance.now() - this.activeAlert.decayTimer;
      if (elapsed > this.activeAlert.decayDuration) {
        // Slowly ease out intensity
        this.activeAlert.intensity *= 0.96;
        if (this.activeAlert.intensity < 0.05) {
          this.activeAlert.active = false;
        }
      }
    }

    const numPoints = this.options.numPoints;
    const alertAngleRad = (this.activeAlert.angle - 90) * (Math.PI / 180);

    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      let targetHeight = 0;

      // 1. Ambient noise & chatter ripples (organic spectrogram peaks around circumference)
      if (this.options.ambientActive && (this.options.ambientIntensity || 0) > 0.05) {
        const amp = this.options.ambientIntensity;
        const noise1 = Math.sin(angle * 8 + this.time * 2.8) * 2.5;
        const noise2 = Math.cos(angle * 16 - this.time * 4.2) * 1.8;
        const noise3 = Math.sin(angle * 28 + this.time * 6.0) * 1.2;
        const noise4 = Math.cos(angle * 40 + this.time * 8.0) * 0.9;
        targetHeight = Math.max(0, (Math.abs(noise1) + Math.abs(noise2) + Math.abs(noise3) + Math.abs(noise4)) * 0.8) * amp;
      }

      // 2. Focused alert sound waves (prominent peaks at the detected angle)
      if (this.activeAlert.active) {
        let angleDiff = Math.abs(angle - alertAngleRad);
        if (angleDiff > Math.PI) angleDiff = Math.PI * 2 - angleDiff;

        // Sector spread width (~45 degrees)
        const spread = 0.45;
        if (angleDiff < spread) {
          const proximity = 1 - angleDiff / spread;
          // Harmonic wave spikes
          const harmonic1 = Math.sin(angle * 32 + this.time * 12) * 14;
          const harmonic2 = Math.sin(angle * 48 - this.time * 16) * 9;
          const spikeHeight = (Math.sin(proximity * Math.PI) * 35 + harmonic1 + harmonic2) * this.activeAlert.intensity;
          targetHeight = Math.max(targetHeight, spikeHeight);
        }
      }

      // Smooth spring damping interpolation
      this.points[i] += (targetHeight - this.points[i]) * 0.22;
    }
  }

  draw() {
    const ctx = this.ctx;
    const cx = this.centerX;
    const cy = this.centerY;
    const r = this.baseRadius;
    const numPoints = this.options.numPoints;

    ctx.clearRect(0, 0, this.width, this.height);

    if (r <= 0) return;

    // 1. Draw subtle background radar rings and crosshairs
    if (!this.options.isMini) {
      this.drawRadarGrid(ctx, cx, cy, r);
    }

    // 2. Draw directional waves along the perimeter
    this.drawPerimeterWaves(ctx, cx, cy, r, numPoints);

    // 3. Draw base main circle
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.lineWidth = this.options.isMini ? 2 : 3.5;
    ctx.strokeStyle = '#334155'; // Clean dark slate stroke
    ctx.stroke();

    // Inner glow
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.stroke();
    ctx.restore();

    // 4. Draw directional arrow indicator (matching user sketch)
    if (this.activeAlert.active) {
      this.drawDirectionalArrow(ctx, cx, cy, r);
    }
  }

  drawRadarGrid(ctx, cx, cy, r) {
    ctx.save();
    
    // Concentric guideline rings
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    [r * 0.4, r * 0.7, r * 1.28, r * 1.5].forEach(radius => {
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
    });

    // 8 Cardinal compass ticks
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const x1 = cx + Math.cos(angle) * (r * 0.88);
      const y1 = cy + Math.sin(angle) * (r * 0.88);
      const x2 = cx + Math.cos(angle) * (r * 0.96);
      const y2 = cy + Math.sin(angle) * (r * 0.96);

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.stroke();
    }

    ctx.restore();
  }

  drawPerimeterWaves(ctx, cx, cy, r, numPoints) {
    const alertAngleRad = (this.activeAlert.angle - 90) * (Math.PI / 180);

    // Render waves around the circumference
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      const nextAngle = ((i + 1) / numPoints) * Math.PI * 2;
      const h = this.points[i];
      if (h <= 0.5) continue;

      const outerR = r + h;

      const x1 = cx + Math.cos(angle) * r;
      const y1 = cy + Math.sin(angle) * r;
      const x2 = cx + Math.cos(angle) * outerR;
      const y2 = cy + Math.sin(angle) * outerR;

      // Check if this point is in the highlighted active alert sector
      let angleDiff = Math.abs(angle - alertAngleRad);
      if (angleDiff > Math.PI) angleDiff = Math.PI * 2 - angleDiff;

      const isAlertSector = this.activeAlert.active && angleDiff < 0.45;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);

      if (isAlertSector) {
        // Highlighted wave spikes (Yellow / Alert color with glow)
        ctx.strokeStyle = this.activeAlert.color;
        ctx.lineWidth = this.options.isMini ? 2 : 3;
        ctx.shadowColor = this.activeAlert.glowColor;
        ctx.shadowBlur = 10;
        ctx.lineCap = 'round';
      } else {
        // Ambient noise waves (Subtle slate/white chatter)
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
        ctx.lineWidth = 1.8;
        ctx.lineCap = 'round';
      }

      ctx.stroke();
      ctx.restore();
    }
  }

  clearAlert() {
    this.activeAlert.active = false;
    this.activeAlert.intensity = 0;
  }

  setAmbientNoise(enabled, intensity = 1.0) {
    this.options.ambientActive = enabled;
    this.options.ambientIntensity = intensity;
  }

  drawDirectionalArrow(ctx, cx, cy, r) {
    if (!this.activeAlert.active || this.activeAlert.intensity < 0.08) return;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(this.currentArrowAngle);

    const arrowColor = this.activeAlert.color;
    // Positioned further from center (near 84% radius) so it never overlaps center text or pills
    const arrowDistance = r * 0.84; 

    // Draw Flat Wide Chevron Arrowhead (matching user drawing)
    const tipX = arrowDistance;
    const wingX = arrowDistance - 11;
    const wingSpread = 22; // Wide flatter angle
    const notchX = arrowDistance - 5;

    ctx.beginPath();
    ctx.moveTo(tipX, 0);                 // Apex pointing toward the wave
    ctx.lineTo(wingX, -wingSpread);      // Top wing corner
    ctx.lineTo(wingX + 3, -wingSpread);  // Smooth edge
    ctx.lineTo(notchX, 0);               // Inner notch
    ctx.lineTo(wingX + 3, wingSpread);   // Smooth bottom edge
    ctx.lineTo(wingX, wingSpread);       // Bottom wing corner
    ctx.closePath();

    ctx.fillStyle = arrowColor;
    ctx.shadowColor = this.activeAlert.glowColor;
    ctx.shadowBlur = 12;
    ctx.fill();

    // Clean outline stroke
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    ctx.restore();
  }

  startLoop() {
    let lastTime = performance.now();
    const frame = (now) => {
      const delta = now - lastTime;
      lastTime = now;

      this.update(delta);
      this.draw();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
}

window.AcousticRadar = AcousticRadar;
