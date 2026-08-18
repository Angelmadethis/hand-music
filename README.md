# 🎵 Hand Orchestra

Control a virtual orchestra with hand gestures via your webcam. Each finger activates a different instrument playing a note in a Cmaj7 chord — raise multiple fingers to layer instruments together like a real orchestra.

---

## 🎹 Instruments

| Finger | Instrument | Sound | Note |
|--------|-----------|-------|------|
| 🟣 Pinky | **Strings** | Warm detuned saw waves with vibrato (violin section) | C3 |
| 🟠 Ring | **Flute** | Pure sine + soft harmonics, breathy | E4 |
| 🔵 Middle | **Brass** | Rich saw with bright harmonics (French horn) | G4 |
| 🟢 Index | **Harp** | Plucked string with sharp attack + natural decay | B4 |
| 🟡 Thumb | **Cello** | Deep bass with sub-octave richness | C5 |

Raise **multiple fingers** to hear instruments layer together — the Cmaj7 chord (C-E-G-B-C) creates a lush orchestral sound.

---

## 🎮 Controls

| Gesture | Effect |
|---------|--------|
| **Raise fingers** | Add that instrument to the mix |
| **Closed fist** | Silence all instruments |
| **Move hand up** | Shift octave higher (+1) |
| **Move hand down** | Shift octave lower (-1) |
| **Move hand left** | Brighter sound (filter opens) |
| **Move hand right** | Darker sound (filter closes) |

---

## 🎛️ Audio Features

- **Schroeder Reverb** — Concert-hall ambience with 4 comb filters + 2 allpass filters
- **Stereo Ping-Pong Delay** — Spatial depth with cross-channel feedback
- **Stereo Panning** — Each instrument is panned to a different position in the stereo field
- **ADSR Envelopes** — Per-instrument attack, decay, sustain, release for realistic dynamics
- **Lowpass Filter** — Hand-controlled brightness (cutoff sweeps 400Hz → 10kHz)
- **Soft Clipping** — Smooth saturation to prevent digital harshness

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- A webcam
- A working audio output (speakers or headphones)

### Setup

```bash
# Clone or download this repository
cd "motion tracking"

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe numpy sounddevice
```

### Quick Start

```bash
python v1.py
```

Press **q** in the video window to quit.

---

## 🖥️ How It Works

1. **Webcam** captures your hand at 640×480
2. **MediaPipe Hands** (running in a background thread) detects 21 hand landmarks
3. **Finger detection** checks which fingers are raised using axis-projected extension tests (works at any hand angle)
4. **Orchestra engine** maps each raised finger to its instrument voice
5. **Audio callback** renders all active instruments in real-time with reverb, delay, and stereo panning
6. **Stereo output** is sent to your speakers/headphones

The hand detection runs in a **separate thread** from the display loop, ensuring smooth FPS even when MediaPipe is slow.

---

## ⚙️ Configuration

Edit the constants at the top of `v1.py`:

```python
# Chord notes (change the musical key)
CHORD = {
    "pinky":  {"note": "C3",  "freq": 130.81},
    "ring":   {"note": "E4",  "freq": 329.63},
    "middle": {"note": "G4",  "freq": 392.00},
    "index":  {"note": "B4",  "freq": 493.88},
    "thumb":  {"note": "C5",  "freq": 523.25},
}

# Effects levels
REVERB_MIX     = 0.28   # 0 = dry, 1 = fully wet
DELAY_MIX      = 0.15   # 0 = no delay, 1 = full delay
DELAY_TIME     = 0.37   # seconds between echoes
DELAY_FEEDBACK = 0.35   # echo repeats (0 = one, 0.9 = many)

# Stereo panning per instrument (0.0 = left, 1.0 = right)
PANNING = {
    "pinky":  0.25,  # strings → left-ish
    "ring":   0.60,  # flute → right-ish
    "middle": 0.50,  # brass → center
    "index":  0.40,  # harp → center-left
    "thumb":  0.75,  # cello → right
}
```

### Try Different Chords

| Chord | Notes |
|-------|-------|
| **Cmaj7** (default) | C3, E4, G4, B4, C5 |
| **Am7** | A2, C4, E4, G4, A4 |
| **Fmaj7** | F2, A3, C4, E4, F4 |
| **G7** | G2, B3, D4, F4, G4 |
| **Dm9** | D2, F3, A3, C4, E4 |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **No sound** | Check your default audio output device. Try `pip install --upgrade sounddevice` |
| **Low FPS** | Close other applications using the webcam or CPU. The detection thread should keep the display smooth |
| **Hand not detected** | Ensure good lighting. Position your hand 30-80cm from the camera |
| **Wrong fingers detected** | The detection uses axis-projection — try holding your hand more flat/open to the camera |
| **Audio crackling** | Try increasing `BLOCK_SIZE` (e.g., 1024) in v1.py at the cost of slightly more latency |
| **Module not found** | Run `pip install opencv-python mediapipe numpy sounddevice` in your activated venv |

---

## 📁 Project Structure

```
motion tracking/
├── v1.py              # Main application (Hand Orchestra)
├── hand_mouse.py      # Previous version (simple hand synth)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 📝 License

Free to use and modify. Built with [MediaPipe](https://google.github.io/mediapipe/), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), and [SoundDevice](https://python-sounddevice.readthedocs.io/).
