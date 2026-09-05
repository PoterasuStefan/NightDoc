---
name: RespiSense Sentinel
colors:
  surface: '#0c1324'
  surface-dim: '#0c1324'
  surface-bright: '#33394c'
  surface-container-lowest: '#070d1f'
  surface-container-low: '#151b2d'
  surface-container: '#191f31'
  surface-container-high: '#23293c'
  surface-container-highest: '#2e3447'
  on-surface: '#dce1fb'
  on-surface-variant: '#bcc9cd'
  inverse-surface: '#dce1fb'
  inverse-on-surface: '#2a3043'
  outline: '#869397'
  outline-variant: '#3d494c'
  surface-tint: '#4cd7f6'
  primary: '#4cd7f6'
  on-primary: '#003640'
  primary-container: '#06b6d4'
  on-primary-container: '#00424f'
  inverse-primary: '#00687a'
  secondary: '#4edea3'
  on-secondary: '#003824'
  secondary-container: '#00a572'
  on-secondary-container: '#00311f'
  tertiary: '#ffb2b7'
  on-tertiary: '#67001b'
  tertiary-container: '#ff7f8b'
  on-tertiary-container: '#7d0023'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#acedff'
  primary-fixed-dim: '#4cd7f6'
  on-primary-fixed: '#001f26'
  on-primary-fixed-variant: '#004e5c'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#ffdadb'
  tertiary-fixed-dim: '#ffb2b7'
  on-tertiary-fixed: '#40000d'
  on-tertiary-fixed-variant: '#92002a'
  background: '#0c1324'
  on-background: '#dce1fb'
  surface-variant: '#2e3447'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.01em
  telemetry-hero:
    fontFamily: JetBrains Mono
    fontSize: 56px
    fontWeight: '600'
    lineHeight: 56px
    letterSpacing: -0.03em
  telemetry-hero-mobile:
    fontFamily: JetBrains Mono
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: 0em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0.01em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.01em
  telemetry-val:
    fontFamily: JetBrains Mono
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 24px
    letterSpacing: -0.02em
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.08em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.02em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  space-2xs: 0.125rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-base: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
  space-2xl: 3rem
  gutter-mobile: 0.75rem
  gutter-desktop: 1.25rem
  margin-screen-sm: 1rem
  margin-screen-lg: 2rem
---

## Brand & Style
The design system embodies a clinical-grade, nocturnal sentinel aesthetic: authoritative, quiet, reassuring, and impeccably accurate. Designed for high-stress intensive care monitoring, post-acute bedside observation, and overnight acoustic pulmonary diagnostics, the UI prioritizes zero light pollution, zero cognitive friction, and instantaneous signal readability.

The visual style merges tactical high-contrast minimalism with glowing bio-luminescent data instrumentation. It relies on pitch-black OLED surfaces to minimize circadian disruption during night monitoring, accented by focused spectral lasers—pure cyans, calibrated emeralds, warning ambers, and alert roses. Physical metaphors manifest through tactile, engineered glass-panel cartridges with subtle micro-grooves, recessed LED indicators, and crisp, clinical telemetry readouts.

## Colors
The palette is built strictly for high-acuity OLED dark mode. Absolute black grounds the interface to preserve visual recovery and contrast ratios in low-light clinical stations.

- **Primary (`#06b6d4` - Cyan Bio-Telemetry):** The primary signal carrier. Represents ongoing acoustic listening, pulmonary spectrograms, frequency analysis, and active audio capture.
- **Secondary (`#10b981` - Normative Emerald):** Clinical benchmark color. Signals safe laminar airflow, optimal tidal volume, stable respiration rates, and clean baseline acoustic stamps.
- **Tertiary (`#f43f5e` - Acute Exacerbation Rose):** Immediate emergency attention. Signifies paroxysmal coughing fits, stridor, acute desaturation triggers, and severe airway obstruction.
- **Warning Accent (`#f59e0b` - Telemetry Amber):** Borderline indicators, shallow respiration, wheeze onset, or degraded acoustic signal-to-noise ratio.
- **Neutral Surface Foundations:**
  - `Base / Canvas`: Deep Slate OLED `#020617` (Slate 950)
  - `Surface 1 (Panels/Cards)`: `#090d16` (Engineered Obsidian)
  - `Surface 2 (Elevated Telemetry Trays)`: `#111827` (Zinc-Slate 900)
  - `Surface Interactive`: `#1e293b`
  - `Borders (Ghost Clinical)`: `#1e293b` with 60% opacity; `#334155` for interactive states
  - `Text Primary`: `#f8fafc` (Slate 50)
  - `Text Muted / Units`: `#64748b` (Slate 500)

## Typography
The typographic system segregates clinical identity, descriptive clinical notes, and tabular acoustic metrics:

- **Plus Jakarta Sans (Headings):** Modern, geometric, clean. Renders anatomical headers, patient identifiers, and module names with crisp confidence.
- **Inter (Body & Narrative):** High-legibility neutral workhorse designed for clinical annotations, diagnostic summaries, and triage instructions.
- **JetBrains Mono (Telemetry & Status Tags):** Strictly monospaced for numerical precision. Used for respiratory rates (BrPM), decibel acoustic readings, tidal volume estimations, SpO2, and cough timeline counts. Tabular lining guarantees zero layout jitter during rapid real-time telemetry streaming. All numerical units are paired with `label-caps` in uppercase format.

## Layout & Spacing
The layout follows a dense, disciplined 12-column telemetry dashboard grid on desktop/tablets and a 4-column stack on mobile. Because acoustic waveform visualizations require uninterrupted horizontal scan paths, telemetry plots break standard column barriers into full-width or dual-span sensor bands.

- **Rhythm:** An ultra-compact 4px/8px incremental rhythm keeps visual density high without causing clutter. Component interiors use `space-md` (12px) padding to mimic medical instrument modules.
- **Breakpoints:**
  - `Mobile (360px - 767px)`: 4-column layout, compact margins (`margin-screen-sm`), stacked live waveforms, vertical triage alerts pinned to the bottom viewport.
  - `Tablet (768px - 1023px)`: 8-column layout, dual-pane waveform and timeline view, 16px gutters.
  - `Desktop / Clinical Monitor (1024px+)`: 12-column layout, persistent live audio spectral stream, side-docked patient queue and critical alert feeds, 20px gutters (`gutter-desktop`).

## Elevation & Depth
Depth avoids heavy, muddy drop shadows in favor of OLED-tailored luminance layers, precise 1px borders, and localized photon glows.

- **Layer 0 (True Void):** `#020617` — The infinite monitor backdrop.
- **Layer 1 (Recessed Well):** `#050811` inset with a subtle `box-shadow: inset 0 1px 3px rgba(0,0,0,0.8)` for inactive waveform buffers and background audio spectra.
- **Layer 2 (Instrument Enclosures):** `#090d16` bounded by a crisp `1px solid rgba(255, 255, 255, 0.08)`.
- **Layer 3 (Floating Diagnostic Overlays):** `#111827` bounded by `1px solid rgba(6, 182, 212, 0.25)` and a subtle ambient cyan back-glow: `0 8px 32px -4px rgba(6, 182, 212, 0.12)`.
- **Bioluminescent Signaling:** Normal status creates a soft emerald aura (`0 0 12px rgba(16, 185, 129, 0.25)`). Exacerbation alerts invoke a high-priority pulse aura (`0 0 20px rgba(244, 63, 94, 0.4)`).

## Shapes
The shape system utilizes deliberate, low-radius curvature (`roundedness: 1` — Soft 4px to 8px base) to convey clinical precision and equipment-grade industrial design. Full pill forms are strictly reserved for live recording indicators, audio playback scrubbers, and severity badges. Sharp inner corners on nested telemetry metrics maintain a calibrated, diagnostic feel.

## Components

### Telemetry & Medical Waveform Cards
- **Structure:** Encased in Layer 2 obsidian surfaces with a 1px border (`#1e293b`). The card header holds a monospaced module title (e.g., `ACOUSTIC_SPECTROGRAM_CH1`), current value, units in slate-500, and a status beacon.
- **Waveform Canvas:** Recessed Layer 1 well housing SVG/Canvas-rendered continuous polyline audio streams. Active audio frequency spikes are rendered in `#06b6d4` with a multi-layered drop-glow (`feDropShadow`). Exacerbation spikes dynamically shift line rendering to `#f43f5e`.

### Buttons
- **Primary Clinical Action:** Background `#06b6d4`, foreground `#020617`, typography `JetBrains Mono` 13px weight 600. Subtle cyan edge glow on hover; active press scales down to `0.98`.
- **Emergency Triage Action:** Background `#f43f5e`, text `#ffffff`. Pulsing box-shadow for persistent alerts.
- **Secondary Ghost Button:** Background `rgba(30, 41, 59, 0.5)`, border `1px solid rgba(255, 255, 255, 0.12)`, text `#f8fafc`. Hover increases border to `#06b6d4`.

### Badges & Telemetry Chips
- **Geometry:** Height 22px, `rounded-full`, padding 0 8px.
- **Safe / Normative:** Dark emerald background (`rgba(16, 185, 129, 0.12)`), text `#10b981`, border `1px solid rgba(16, 185, 129, 0.3)`. Prefixed with a 4px solid glowing green dot.
- **Exacerbation Alert:** Dark rose background (`rgba(244, 63, 94, 0.15)`), text `#f43f5e`, border `1px solid rgba(244, 63, 94, 0.4)`. Accompanied by a flashing indicator dot.

### Numeric Data Displays (Telemetry Blocks)
- Large tabular digits (`JetBrains Mono`, 24px to 56px) stacked immediately above a micro uppercase title (`label-caps`, 10px, Slate-400). Unit tags are half-sized and aligned to the numerical baseline.

### Inputs & Controls
- **Audio Sensitivity Sliders:** Thin 2px track in `#1e293b` with a glowing `#06b6d4` fill track. The thumb is a crisp 12px diamond or rounded pill with an outer glow.
- **Form Inputs:** Flush dark background (`#090d16`), 1px slate-800 border, turning cyan on focus with zero offset ring (`box-shadow: 0 0 0 1px #06b6d4`). Monospaced text input for patient parameters.

### Real-Time Sentinel Status Indicator
- A persistent, pulsing acoustic sentinel icon positioned at the app bar. A concentric dual-ring ripple animation visually indicates active AI acoustic feature extraction.