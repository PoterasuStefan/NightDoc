/**
 * NightDoc / RespiSense AI – Radial Acoustic Waveform & Biomarker Sentinel Visualizer
 * Implements 360-degree directional sound waves, ambient baseline ripples,
 * and highlighted clinical biomarker peaks (Cough, Wheezing, Snoring, Infant Distress).
 */
class AcousticRadar {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    
    this.ctx = this.canvas.getContext('2d');
    this.options = {
      isMini: options.isMini || false,
      numPoints: options.numPoints || 240,
      baseRadiusRatio: options.baseRadiusRatio || 0.58,
      ambientActive: true,
      ...options
    };

    this.points = new Float32Array(this.options.numPoints);
    this.options.ambientActive = true;
    this.options.ambientIntensity = 0.4;

    this.activeAlert = {
      angle: 0,
      intensity: 0.0,
      color: '#EF4444',
      glowColor: 'rgba(239, 68, 68, 0.5)',
      active: false,
      decayTimer: 0,
      decayDuration: 3000
    };

    this.currentArrowAngle = 0;
    this.targetArrowAngle = 0;
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

    this.canvas.width = this.width * this.pixelRatio;
    this.canvas.height = this.height * this.pixelRatio;
    this.ctx.scale(this.pixelRatio, this.pixelRatio);

    this.centerX = this.width / 2;
    this.centerY = this.height / 2;
    this.baseRadius = Math.min(this.width, this.height) * this.options.baseRadiusRatio * 0.5;
  }

  triggerBurst(params) {
    const angle = params.angle || 0;
    const intensity = params.intensity || 1.0;
    const color = params.color || '#EF4444';

    this.triggerAlert(angle, intensity, color);
  }

  triggerAlert(angleDeg, intensity = 1.0, color = '#EF4444') {
    this.activeAlert = {
      angle: angleDeg,
      intensity: intensity,
      color: color,
      glowColor: color.startsWith('#') ? `${color}66` : color,
      active: true,
      decayTimer: performance.now(),
      decayDuration: 3500
    };

    const rad = (angleDeg - 90) * (Math.PI / 180);
    this.targetArrowAngle = rad;
  }

  update(deltaTime) {
    this.time += 0.04;

    let diff = this.targetArrowAngle - this.currentArrowAngle;
    while (diff < -Math.PI) diff += Math.PI * 2;
    while (diff > Math.PI) diff -= Math.PI * 2;
    this.currentArrowAngle += diff * 0.12;

    if (this.activeAlert.active) {
      const elapsed = performance.now() - this.activeAlert.decayTimer;
      if (elapsed > this.activeAlert.decayDuration) {
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

      // Ambient baseline
      if (this.options.ambientActive) {
        const amp = this.options.ambientIntensity || 0.4;
        const noise1 = Math.sin(angle * 8 + this.time * 2.8) * 2.0;
        const noise2 = Math.cos(angle * 16 - this.time * 4.2) * 1.5;
        const noise3 = Math.sin(angle * 28 + this.time * 6.0) * 1.0;
        targetHeight = Math.max(0, (Math.abs(noise1) + Math.abs(noise2) + Math.abs(noise3)) * 0.7) * amp;
      }

      // Biomarker spike
      if (this.activeAlert.active) {
        let angleDiff = Math.abs(angle - alertAngleRad);
        if (angleDiff > Math.PI) angleDiff = Math.PI * 2 - angleDiff;

        const spread = 0.45;
        if (angleDiff < spread) {
          const proximity = 1 - angleDiff / spread;
          const harmonic1 = Math.sin(angle * 32 + this.time * 12) * 14;
          const harmonic2 = Math.sin(angle * 48 - this.time * 16) * 9;
          const spikeHeight = (Math.sin(proximity * Math.PI) * 35 + harmonic1 + harmonic2) * this.activeAlert.intensity;
          targetHeight = Math.max(targetHeight, spikeHeight);
        }
      }

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

    this.drawRadarGrid(ctx, cx, cy, r);
    this.drawPerimeterWaves(ctx, cx, cy, r, numPoints);

    // Center base ring
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = '#1E293B';
    ctx.stroke();

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.stroke();
    ctx.restore();

    if (this.activeAlert.active) {
      this.drawDirectionalArrow(ctx, cx, cy, r);
    }
  }

  drawRadarGrid(ctx, cx, cy, r) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    [r * 0.4, r * 0.7, r * 1.25, r * 1.5].forEach(radius => {
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
    });

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

    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      const h = this.points[i];
      if (h <= 0.5) continue;

      const outerR = r + h;
      const x1 = cx + Math.cos(angle) * r;
      const y1 = cy + Math.sin(angle) * r;
      const x2 = cx + Math.cos(angle) * outerR;
      const y2 = cy + Math.sin(angle) * outerR;

      let angleDiff = Math.abs(angle - alertAngleRad);
      if (angleDiff > Math.PI) angleDiff = Math.PI * 2 - angleDiff;

      const isAlertSector = this.activeAlert.active && angleDiff < 0.45;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);

      if (isAlertSector) {
        ctx.strokeStyle = this.activeAlert.color;
        ctx.lineWidth = 3;
        ctx.shadowColor = this.activeAlert.glowColor;
        ctx.shadowBlur = 10;
        ctx.lineCap = 'round';
      } else {
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
        ctx.lineWidth = 1.8;
        ctx.lineCap = 'round';
      }

      ctx.stroke();
      ctx.restore();
    }
  }

  drawDirectionalArrow(ctx, cx, cy, r) {
    if (!this.activeAlert.active || this.activeAlert.intensity < 0.08) return;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(this.currentArrowAngle);

    const arrowColor = this.activeAlert.color;
    const arrowDistance = r * 0.84; 

    const tipX = arrowDistance;
    const wingX = arrowDistance - 11;
    const wingSpread = 22;
    const notchX = arrowDistance - 5;

    ctx.beginPath();
    ctx.moveTo(tipX, 0);
    ctx.lineTo(wingX, -wingSpread);
    ctx.lineTo(wingX + 3, -wingSpread);
    ctx.lineTo(notchX, 0);
    ctx.lineTo(wingX + 3, wingSpread);
    ctx.lineTo(wingX, wingSpread);
    ctx.closePath();

    ctx.fillStyle = arrowColor;
    ctx.shadowColor = this.activeAlert.glowColor;
    ctx.shadowBlur = 12;
    ctx.fill();

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
