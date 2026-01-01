# ReasonAudio: The Sound of Reasoning
## Industrial ASMR - UI Sound System

> **Classification:** Multi-Sensory Brand Extension
> **Purpose:** Provide haptic-like audio feedback for ReasonUI components
> **Philosophy:** "Machines make noise" - Industrial, precise, mechanical

---

## Core Concept

ReasonKit is a machine. Machines make noise. These UI sounds provide **haptic-like feedback** for the `ReasonUI` components, creating a multi-sensory experience that reinforces the "Industrial" brand identity.

**Audio Profile:**
- **Texture:** High-voltage hums, solenoid clicks, ceramic snaps
- **Avoid:** Musical chimes, standard "blips," cheerful "ding" sounds
- **Aesthetic:** Industrial ASMR - precise, mechanical, authoritative

---

## Sound Event Catalog

### 1. GigaThink Start

**Event:** Protocol execution begins (GigaThink module)

**Sound Description:**
- Deep, rising turbine spin-up
- Mechanical whirr building from silence
- Sense of power being engaged

**Frequency Profile:**
- **Low-freq rumble:** 60Hz foundation
- **Rising tone:** 60Hz → 200Hz over 1.5s
- **High-freq whine:** 2kHz overlay (subtle)
- **Attack:** 0ms (instant start)
- **Decay:** 1.5s (gradual fade-in)
- **Sustain:** 0.3s (steady state)
- **Release:** 0.5s (fade-out)

**Implementation:**
```javascript
// Web Audio API example
const audioContext = new AudioContext();
const oscillator = audioContext.createOscillator();
const gainNode = audioContext.createGain();

oscillator.type = 'sawtooth';
oscillator.frequency.setValueAtTime(60, audioContext.currentTime);
oscillator.frequency.exponentialRampToValueAtTime(200, audioContext.currentTime + 1.5);

gainNode.gain.setValueAtTime(0, audioContext.currentTime);
gainNode.gain.linearRampToValueAtTime(0.3, audioContext.currentTime + 0.1);
gainNode.gain.linearRampToValueAtTime(0, audioContext.currentTime + 2.0);
```

**File:** `reasonkit-core/brand/audio/gigathink-start.wav`

---

### 2. Node Connection

**Event:** TraceNode connects to another node in reasoning chain

**Sound Description:**
- Sharp, crystalline "snap" (like glass fitting together)
- Precise mechanical engagement
- Satisfying click

**Frequency Profile:**
- **High-freq transient:** 4kHz primary
- **Harmonic:** 8kHz (subtle)
- **Attack:** 0ms (instant)
- **Decay:** 50ms (quick snap)
- **Sustain:** 0ms (no sustain)
- **Release:** 100ms (quick fade)

**Implementation:**
```javascript
const oscillator = audioContext.createOscillator();
const gainNode = audioContext.createGain();

oscillator.type = 'sine';
oscillator.frequency.setValueAtTime(4000, audioContext.currentTime);

gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);
gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.15);
```

**File:** `reasonkit-core/brand/audio/node-connection.wav`

---

### 3. Logic Lock

**Event:** Protocol step completes, confidence threshold met

**Sound Description:**
- Heavy pneumatic air-brake release
- Mechanical confirmation
- Solid, authoritative

**Frequency Profile:**
- **White noise burst:** 0-8kHz (50ms)
- **Low thud:** 80Hz (200ms)
- **Attack:** 0ms
- **Decay:** 50ms (noise) + 200ms (thud)
- **Sustain:** 0ms
- **Release:** 100ms

**Implementation:**
```javascript
// White noise burst
const bufferSize = audioContext.sampleRate * 0.05;
const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
const data = buffer.getChannelData(0);
for (let i = 0; i < bufferSize; i++) {
  data[i] = Math.random() * 2 - 1;
}
const whiteNoise = audioContext.createBufferSource();
whiteNoise.buffer = buffer;

// Low thud
const thud = audioContext.createOscillator();
thud.type = 'sine';
thud.frequency.setValueAtTime(80, audioContext.currentTime);
```

**File:** `reasonkit-core/brand/audio/logic-lock.wav`

---

### 4. Error/Glitch

**Event:** Protocol step fails, confidence below threshold, validation error

**Sound Description:**
- Short burst of static interference
- Geiger counter click
- Discordant, granular synthesis
- Warning signal

**Frequency Profile:**
- **Static burst:** 0-10kHz (100ms)
- **Geiger click:** 2kHz transient (20ms)
- **Discordant tone:** 440Hz + 443Hz (beating, 200ms)
- **Attack:** 0ms
- **Decay:** 100ms (static) + 20ms (click) + 200ms (tone)
- **Sustain:** 0ms
- **Release:** 50ms

**Implementation:**
```javascript
// Static burst
const staticBuffer = audioContext.createBuffer(1, audioContext.sampleRate * 0.1, audioContext.sampleRate);
const staticData = staticBuffer.getChannelData(0);
for (let i = 0; i < staticData.length; i++) {
  staticData[i] = (Math.random() * 2 - 1) * 0.3;
}

// Geiger click
const geiger = audioContext.createOscillator();
geiger.type = 'square';
geiger.frequency.setValueAtTime(2000, audioContext.currentTime);

// Discordant tone (beating)
const tone1 = audioContext.createOscillator();
const tone2 = audioContext.createOscillator();
tone1.frequency.setValueAtTime(440, audioContext.currentTime);
tone2.frequency.setValueAtTime(443, audioContext.currentTime);
```

**File:** `reasonkit-core/brand/audio/error-glitch.wav`

---

### 5. Success/Verify

**Event:** Protocol completes successfully, verification passes

**Sound Description:**
- Clean, pure sine wave ping
- Infinite reverb (The "Beam")
- Satisfying, complete
- Sense of resolution

**Frequency Profile:**
- **Pure C5 note:** 523.25Hz
- **Attack:** 0ms (instant)
- **Decay:** 200ms (quick fade)
- **Sustain:** 0ms
- **Release:** 2s (long reverb tail)
- **Reverb:** Convolution reverb with long tail (3-5s)

**Implementation:**
```javascript
const oscillator = audioContext.createOscillator();
const gainNode = audioContext.createGain();
const reverb = audioContext.createConvolver();

oscillator.type = 'sine';
oscillator.frequency.setValueAtTime(523.25, audioContext.currentTime); // C5

gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);
gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 2.2);

// Reverb impulse response (long tail)
const reverbBuffer = createReverbImpulse(audioContext, 3.0);
reverb.buffer = reverbBuffer;
```

**File:** `reasonkit-core/brand/audio/success-verify.wav`

---

## Integration with ReasonUI Components

### TraceNode
- **Connection:** `node-connection.wav`
- **Completion:** `logic-lock.wav`

### ConfidenceMeter
- **Update:** Subtle `node-connection.wav` (lower volume)
- **Threshold reached:** `logic-lock.wav`

### LogStream
- **New entry:** Subtle `node-connection.wav` (very low volume)
- **Error entry:** `error-glitch.wav`

### StatusToggle
- **Activate:** `logic-lock.wav`
- **Deactivate:** `node-connection.wav` (reversed)

---

## Audio File Specifications

### Format
- **Container:** WAV (uncompressed) or OGG (compressed)
- **Sample Rate:** 44.1kHz or 48kHz
- **Bit Depth:** 16-bit or 24-bit
- **Channels:** Mono (preferred) or Stereo

### File Naming
```
reasonkit-core/brand/audio/
├── gigathink-start.wav
├── node-connection.wav
├── logic-lock.wav
├── error-glitch.wav
└── success-verify.wav
```

### Compression
- **WAV:** Uncompressed, highest quality
- **OGG:** Compressed for web (Vorbis codec)
- **MP3:** Fallback (not preferred)

---

## Implementation Guidelines

### Web (JavaScript)
```javascript
// Audio manager
class ReasonAudio {
  constructor() {
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.sounds = new Map();
    this.loadSounds();
  }

  async loadSounds() {
    const soundFiles = [
      'gigathink-start',
      'node-connection',
      'logic-lock',
      'error-glitch',
      'success-verify'
    ];

    for (const sound of soundFiles) {
      const audio = new Audio(`/brand/audio/${sound}.wav`);
      audio.volume = 0.5; // Default volume
      this.sounds.set(sound, audio);
    }
  }

  play(soundName, volume = 0.5) {
    const sound = this.sounds.get(soundName);
    if (sound) {
      sound.volume = volume;
      sound.currentTime = 0;
      sound.play().catch(e => console.warn('Audio play failed:', e));
    }
  }
}

// Usage
const audio = new ReasonAudio();
audio.play('gigathink-start', 0.7);
```

### Rust (Desktop)
```rust
// Use rodio or cpal for audio playback
use rodio::{Decoder, OutputStream, Sink};
use std::fs::File;
use std::io::BufReader;

pub struct ReasonAudio {
    sink: Sink,
}

impl ReasonAudio {
    pub fn new() -> Self {
        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let sink = Sink::try_new(&stream_handle).unwrap();
        Self { sink }
    }

    pub fn play(&self, sound_file: &str) {
        let file = File::open(format!("brand/audio/{}", sound_file)).unwrap();
        let source = Decoder::new(BufReader::new(file)).unwrap();
        self.sink.append(source);
    }
}
```

---

## Accessibility

### User Preferences
- **Respect `prefers-reduced-motion`:** Disable audio if motion is reduced
- **Volume control:** User-adjustable volume slider
- **Mute option:** Toggle to disable all sounds

### Implementation
```javascript
// Check for reduced motion preference
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

if (!prefersReducedMotion) {
  audio.play('gigathink-start');
}
```

---

## Brand Alignment

### Industrial Aesthetic
- ✅ High-voltage, mechanical sounds
- ✅ Precise, authoritative feedback
- ✅ No musical elements (except pure tones)
- ✅ Machine-like, systematic

### "Designed, Not Dreamed"
- ✅ Every sound has a purpose
- ✅ Consistent frequency profiles
- ✅ Predictable, reliable
- ✅ Professional, not playful

---

## File Structure

```
reasonkit-core/brand/
├── audio/
│   ├── gigathink-start.wav
│   ├── gigathink-start.ogg
│   ├── node-connection.wav
│   ├── node-connection.ogg
│   ├── logic-lock.wav
│   ├── logic-lock.ogg
│   ├── error-glitch.wav
│   ├── error-glitch.ogg
│   ├── success-verify.wav
│   └── success-verify.ogg
└── REASONAUDIO_SPEC.md (this file)
```

---

## Next Steps

1. **Audio Production:**
   - Create audio files using synthesis or field recording
   - Process with industrial effects (distortion, reverb)
   - Export in WAV and OGG formats

2. **Integration:**
   - Add to ReasonUI components
   - Implement in web and desktop versions
   - Test with accessibility preferences

3. **Documentation:**
   - Add usage examples
   - Create audio preview page
   - Document volume levels and mixing

---

**Last Updated:** 2025-01-01  
**Status:** ✅ Specification Complete

