"""
Hand Orchestra
==============
Control a virtual orchestra with hand gestures via webcam.

Each finger activates a different instrument playing a note in a Cmaj7 chord.
Raise multiple fingers to layer instruments together like an orchestra.

    Pinky  → Strings  (detuned saw waves, vibrato)
    Ring   → Flute    (sine + harmonics, breathy)
    Middle → Brass    (rich saw, punchy)
    Index  → Harp     (plucked, sharp attack + decay)
    Thumb  → Cello    (deep bass, sub-octave)

    Hand height    → octave shift  (-1 / 0 / +1)
    Hand left/right → filter brightness
    Closed fist    → silence

Features:
    - Algorithmic reverb (Schroeder) for concert-hall ambience
    - Stereo delay effect for spatial depth
    - Per-instrument ADSR envelopes
    - Stereo output with instrument panning
    - Threaded MediaPipe for smooth FPS

Requirements:
    pip install opencv-python mediapipe numpy sounddevice

Press 'q' to quit.
"""

import math
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
import sounddevice as sd

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
CAM_INDEX      = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
SAMPLE_RATE    = 44100
BLOCK_SIZE     = 512
PROC_W, PROC_H = 320, 240
DETECT_EVERY   = 1

# Cmaj7 voicing — each finger plays one chord tone
CHORD = {
    "pinky":  {"note": "C3",  "freq": 130.81},
    "ring":   {"note": "E4",  "freq": 329.63},
    "middle": {"note": "G4",  "freq": 392.00},
    "index":  {"note": "B4",  "freq": 493.88},
    "thumb":  {"note": "C5",  "freq": 523.25},
}

FINGER_ORDER = ["pinky", "ring", "middle", "index", "thumb"]

# Stereo panning per instrument (L, R weights, 0=left 1=right)
PANNING = {
    "pinky":  0.25,
    "ring":   0.60,
    "middle": 0.50,
    "index":  0.40,
    "thumb":  0.75,
}

# Reverb / delay settings
REVERB_MIX   = 0.28   # wet amount 0..1
DELAY_MIX    = 0.15   # wet amount 0..1
DELAY_TIME   = 0.37   # seconds
DELAY_FEEDBACK = 0.35

# ═══════════════════════════════════════════════════════════════════
# MediaPipe constants
# ═══════════════════════════════════════════════════════════════════
mp_hands = mp.solutions.hands

THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17
WRIST = 0

COLORS = {
    "pinky":  (255, 0, 255),
    "ring":   (0, 165, 255),
    "middle": (255, 0, 0),
    "index":  (0, 255, 0),
    "thumb":  (0, 255, 255),
}

PALM_CONNECTIONS = [(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17)]


# ═══════════════════════════════════════════════════════════════════
# Threaded hand detector
# ═══════════════════════════════════════════════════════════════════
class HandDetector:
    def __init__(self):
        self.lock = threading.Lock()
        self.landmarks = None
        self._frame = None
        self._new_frame = threading.Event()
        self._running = True
        self._hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, frame):
        if self._new_frame.is_set():
            return
        self._frame = frame
        self._new_frame.set()

    def get_landmarks(self):
        with self.lock:
            return self.landmarks

    def _run(self):
        while self._running:
            if not self._new_frame.wait(timeout=0.05):
                continue
            self._new_frame.clear()
            frame = self._frame
            if frame is None:
                continue
            small = cv2.resize(frame, (PROC_W, PROC_H))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self._hands.process(rgb)
            if result.multi_hand_landmarks:
                raw = result.multi_hand_landmarks[0]
                lm = {i: (int(l.x * FRAME_WIDTH), int(l.y * FRAME_HEIGHT))
                      for i, l in enumerate(raw.landmark)}
            else:
                lm = None
            with self.lock:
                self.landmarks = lm

    def stop(self):
        self._running = False
        self._new_frame.set()
        self._thread.join(timeout=1.0)
        self._hands.close()


# ═══════════════════════════════════════════════════════════════════
# Finger helpers
# ═══════════════════════════════════════════════════════════════════
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_is_extended(lm, tip_idx, pip_idx, mcp_idx):
    wrist = lm[WRIST]
    tip, pip, mcp = lm[tip_idx], lm[pip_idx], lm[mcp_idx]
    if distance(tip, wrist) <= distance(pip, wrist) * 0.95:
        return False
    dx, dy = pip[0] - mcp[0], pip[1] - mcp[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1:
        return False
    tip_proj = ((tip[0] - mcp[0]) * dx + (tip[1] - mcp[1]) * dy) / length_sq
    return tip_proj > 1.0


def get_raised_fingers(lm):
    raised = set()
    thumb_tip, thumb_ip, index_mcp = lm[THUMB_TIP], lm[THUMB_IP], lm[INDEX_MCP]
    thumb_mcp = lm[THUMB_MCP]
    if (distance(thumb_tip, index_mcp) > distance(thumb_mcp, index_mcp) * 1.4
            and distance(thumb_tip, index_mcp) > distance(thumb_ip, index_mcp) * 1.1):
        raised.add("thumb")
    for name, ti, pi, mi in [
        ("index",  INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
        ("middle", MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        ("ring",   RING_TIP,   RING_PIP,   RING_MCP),
        ("pinky",  PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
    ]:
        if _finger_is_extended(lm, ti, pi, mi):
            raised.add(name)
    return raised


# ═══════════════════════════════════════════════════════════════════
# Effects — Schroeder reverb + stereo delay
# ═══════════════════════════════════════════════════════════════════
class SchroederReverb:
    """4-comb + 2-allpass Schroeder reverb (mono, fed from mono mix)."""

    def __init__(self, sr=SAMPLE_RATE):
        # comb delays (prime-ish sample counts for diffusion)
        comb_delays = [int(sr * d) for d in [0.0297, 0.0371, 0.0411, 0.0437]]
        self._combs = [np.zeros(d) for d in comb_delays]
        self._comb_idx = [0] * 4
        self._comb_gains = [0.82, 0.80, 0.78, 0.76]
        # allpass delays
        ap_delays = [int(sr * 0.0053), int(sr * 0.0017)]
        self._aps = [np.zeros(d) for d in ap_delays]
        self._ap_idx = [0] * 2

    def process(self, x):
        """x: 1-D numpy array (mono). Returns wet reverb signal."""
        out = np.empty_like(x)
        for i in range(len(x)):
            s = x[i]
            # 4 parallel combs
            comb_sum = 0.0
            for c in range(4):
                idx = self._comb_idx[c]
                val = self._combs[c][idx]
                self._combs[c][idx] = s + val * self._comb_gains[c]
                self._comb_idx[c] = (idx + 1) % len(self._combs[c])
                comb_sum += val
            s = comb_sum * 0.25
            # 2 series allpass
            for a in range(2):
                idx = self._ap_idx[a]
                val = self._aps[a][idx]
                self._aps[a][idx] = s + val * 0.5
                self._ap_idx[a] = (idx + 1) % len(self._aps[a])
                s = val - 0.5 * self._aps[a][idx]
            out[i] = s
        return out


class StereoDelay:
    """Ping-pong stereo delay."""

    def __init__(self, sr=SAMPLE_RATE, time_s=DELAY_TIME, feedback=DELAY_FEEDBACK):
        n = int(sr * time_s)
        self._buf_L = np.zeros(n)
        self._buf_R = np.zeros(n)
        self._idx = 0
        self._fb = feedback
        self._n = n

    def process(self, left, right):
        L = np.empty(len(left))
        R = np.empty(len(right))
        for i in range(len(left)):
            # read from delay buffers
            dl = self._buf_L[self._idx]
            dr = self._buf_R[self._idx]
            # write current + cross-feed
            self._buf_L[self._idx] = left[i]  + dr * self._fb
            self._buf_R[self._idx] = right[i] + dl * self._fb
            self._idx = (self._idx + 1) % self._n
            L[i] = dl
            R[i] = dr
        return L, R


# ═══════════════════════════════════════════════════════════════════
# Instruments — each with unique timbre + ADSR
# ═══════════════════════════════════════════════════════════════════
class Instrument:
    """Base instrument with ADSR envelope."""

    def __init__(self, freq):
        self.freq = freq
        self.gain = 0.0
        self.target_gain = 0.0
        self._phase = 0.0
        # ADSR times (seconds)
        self._attack  = 0.015
        self._decay   = 0.08
        self._sustain = 0.75
        self._release = 0.25
        self._note_on_time = 0.0

    def _adsr(self, frames):
        atk_c  = 1.0 - math.exp(-1.0 / (self._attack  * SAMPLE_RATE))
        dec_c  = 1.0 - math.exp(-1.0 / (self._decay   * SAMPLE_RATE))
        rel_c  = 1.0 - math.exp(-1.0 / (self._release * SAMPLE_RATE))
        env = np.empty(frames)
        g = self.gain
        for i in range(frames):
            if self.target_gain > 0.5:
                if g < 1.0:
                    g += (1.0 - g) * atk_c
                elif g > self._sustain:
                    g += (self._sustain - g) * dec_c
                else:
                    g += (self._sustain - g) * 0.001
            else:
                g *= (1.0 - rel_c)
                if g < 1e-5:
                    g = 0.0
            env[i] = g
        self.gain = g
        return env

    def _advance_phase(self, freq, frames):
        self._phase += 2 * np.pi * freq * frames / SAMPLE_RATE
        self._phase %= 2 * np.pi

    def render(self, frames, t_arr):
        raise NotImplementedError


class Strings(Instrument):
    """Violin section — detuned saw waves + vibrato."""
    _attack = 0.06
    _decay = 0.15
    _sustain = 0.85
    _release = 0.35

    def render(self, frames, t_arr):
        env = self._adsr(frames)
        f = self.freq
        # slight detuning between voices for chorus effect
        offsets = [-1.5, -0.5, 0.0, 0.5, 1.5]  # cents
        wave = np.zeros(frames)
        for cents in offsets:
            detune = f * (2.0 ** (cents / 1200.0))
            vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.2 * t_arr + self._phase)
            ph = 2 * np.pi * detune * vib * t_arr + self._phase + cents * 0.01
            for h in range(1, 9):
                wave += (1.0 / h) * np.sin(h * ph)
        wave *= 0.06  # normalize for 5 voices × 8 harmonics
        self._advance_phase(f, frames)
        return wave * env


class Flute(Instrument):
    """Concert flute — sine + soft harmonics + breath noise."""
    _attack = 0.025
    _decay = 0.10
    _sustain = 0.80
    _release = 0.20

    def render(self, frames, t_arr):
        env = self._adsr(frames)
        vib = 1.0 + 0.007 * np.sin(2 * np.pi * 5.5 * t_arr + self._phase)
        ph = 2 * np.pi * self.freq * vib * t_arr + self._phase
        wave = (0.65 * np.sin(ph) +
                0.20 * np.sin(2 * ph) +
                0.06 * np.sin(3 * ph) +
                0.02 * np.sin(4 * ph))
        # breath noise (very quiet filtered noise)
        noise = np.random.randn(frames) * 0.015
        wave += noise
        self._advance_phase(self.freq, frames)
        return wave * env * 0.55


class Brass(Instrument):
    """French horn / trumpet — bright harmonics + swell."""
    _attack = 0.035
    _decay = 0.08
    _sustain = 0.80
    _release = 0.18

    def render(self, frames, t_arr):
        env = self._adsr(frames)
        ph = 2 * np.pi * self.freq * t_arr + self._phase
        wave = (0.30 * np.sin(ph) +
                0.28 * np.sin(2 * ph) +
                0.22 * np.sin(3 * ph) +
                0.12 * np.sin(4 * ph) +
                0.05 * np.sin(5 * ph) +
                0.02 * np.sin(6 * ph))
        self._advance_phase(self.freq, frames)
        return wave * env * 0.45


class Harp(Instrument):
    """Plucked string — sharp transient + exponential decay."""
    _attack = 0.002
    _decay = 0.80
    _sustain = 0.0
    _release = 0.05

    def __init__(self, freq):
        super().__init__(freq)
        self._pluck_env = 0.0
        self._prev_gain = 0.0

    def render(self, frames, t_arr):
        # Detect fresh pluck (target just went high)
        if self.target_gain > 0.5 and self._prev_gain <= 0.5:
            self._pluck_env = 1.0
        self._prev_gain = self.target_gain

        # Decay the pluck envelope
        decay_c = 1.0 - math.exp(-1.0 / (self._decay * SAMPLE_RATE))
        env = np.empty(frames)
        p = self._pluck_env
        for i in range(frames):
            p *= (1.0 - decay_c)
            env[i] = p
        self._pluck_env = p

        ph = 2 * np.pi * self.freq * t_arr + self._phase
        # Karplus-Strong-ish: mix harmonics with brightness decay
        wave = np.zeros(frames)
        for h in range(1, 12):
            brightness = 1.0 / (1.0 + 0.3 * h)
            wave += brightness * np.sin(h * ph + h * 0.7)
        wave *= 0.25
        self._advance_phase(self.freq, frames)
        return wave * env


class Cello(Instrument):
    """Cello / double bass — deep, rich, warm."""
    _attack = 0.040
    _decay = 0.12
    _sustain = 0.80
    _release = 0.40

    def render(self, frames, t_arr):
        env = self._adsr(frames)
        vib = 1.0 + 0.005 * np.sin(2 * np.pi * 4.2 * t_arr + self._phase)
        ph = 2 * np.pi * self.freq * vib * t_arr + self._phase
        wave = np.zeros(frames)
        for h in range(1, 7):
            wave += (1.0 / h) * np.sin(h * ph)
        # sub-octave sine for depth
        wave += 0.45 * np.sin(0.5 * ph)
        wave *= 0.22
        self._advance_phase(self.freq, frames)
        return wave * env


INSTRUMENT_CLASSES = {
    "pinky": Strings,
    "ring":  Flute,
    "middle": Brass,
    "index": Harp,
    "thumb": Cello,
}


# ═══════════════════════════════════════════════════════════════════
# Orchestra — 5 voices + reverb + delay + stereo
# ═══════════════════════════════════════════════════════════════════
class Orchestra:
    def __init__(self):
        self.instruments = {}
        for name in FINGER_ORDER:
            cls = INSTRUMENT_CLASSES[name]
            self.instruments[name] = cls(CHORD[name]["freq"])

        self._filter_x = 0.5
        self._lp_L = 0.0
        self._lp_R = 0.0
        self._reverb = SchroederReverb()
        self._delay = StereoDelay()

        self.stream = sd.OutputStream(
            channels=2, callback=self._audio_callback,
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
        )
        self.stream.start()

    def set_active(self, name, active):
        self.instruments[name].target_gain = 1.0 if active else 0.0

    def set_octave(self, shift):
        for name in FINGER_ORDER:
            self.instruments[name].freq = CHORD[name]["freq"] * (2 ** shift)

    def set_filter(self, x_norm):
        self._filter_x = min(1.0, max(0.0, x_norm))

    def silence(self):
        for inst in self.instruments.values():
            inst.target_gain = 0.0

    def close(self):
        self.stream.stop()
        self.stream.close()

    def _audio_callback(self, outdata, frames, time_info, status):
        t_arr = np.arange(frames) / SAMPLE_RATE

        # render each instrument once, store for panning
        inst_buffers = {}
        mono_mix = np.zeros(frames)
        for name in FINGER_ORDER:
            buf = self.instruments[name].render(frames, t_arr)
            inst_buffers[name] = buf
            mono_mix += buf

        # reverb on mono mix
        wet = self._reverb.process(mono_mix)

        # stereo panning
        left  = np.zeros(frames)
        right = np.zeros(frames)
        for name in FINGER_ORDER:
            pan = PANNING[name]
            buf = inst_buffers[name]
            left  += buf * math.sqrt(1.0 - pan)
            right += buf * math.sqrt(pan)
        # add reverb to both channels
        left  = left  * (1.0 - REVERB_MIX) + wet * REVERB_MIX
        right = right * (1.0 - REVERB_MIX) + wet * REVERB_MIX

        # lowpass filter per channel (brightness from hand X)
        cutoff = 400.0 * (10000.0 / 400.0) ** self._filter_x
        rc = 1.0 / (2 * np.pi * cutoff)
        a_coeff = (1.0 / SAMPLE_RATE) / (rc + 1.0 / SAMPLE_RATE)

        out_L = np.empty(frames)
        out_R = np.empty(frames)
        lp_L, lp_R = self._lp_L, self._lp_R
        for i in range(frames):
            lp_L += a_coeff * (left[i]  - lp_L)
            lp_R += a_coeff * (right[i] - lp_R)
            out_L[i] = lp_L
            out_R[i] = lp_R
        self._lp_L, self._lp_R = lp_L, lp_R

        # stereo delay
        dl, dr = self._delay.process(out_L, out_R)
        out_L = out_L * (1.0 - DELAY_MIX) + dl * DELAY_MIX
        out_R = out_R * (1.0 - DELAY_MIX) + dr * DELAY_MIX

        # soft clip + master volume
        out_L = np.tanh(out_L * 1.4) * 0.40
        out_R = np.tanh(out_R * 1.4) * 0.40

        outdata[:, 0] = out_L
        outdata[:, 1] = out_R


# ═══════════════════════════════════════════════════════════════════
# Drawing
# ═══════════════════════════════════════════════════════════════════
PALM_CONN = [(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17)]

FINGER_BONES = [
    ([WRIST,1,2,THUMB_IP,THUMB_TIP], "thumb"),
    ([WRIST,5,6,7,INDEX_TIP],        "index"),
    ([WRIST,9,10,11,MIDDLE_TIP],     "middle"),
    ([WRIST,13,14,15,RING_TIP],      "ring"),
    ([WRIST,17,18,19,PINKY_TIP],     "pinky"),
]

INSTRUMENT_LABELS = {
    "pinky":  "STR",
    "ring":   "FLT",
    "middle": "BRS",
    "index":  "HRP",
    "thumb":  "CEL",
}


def draw_skeleton(frame, lm, active_fingers):
    for a, b in PALM_CONN:
        if a in lm and b in lm:
            cv2.line(frame, lm[a], lm[b], (60, 60, 60), 2, cv2.LINE_AA)

    for chain, name in FINGER_BONES:
        color = COLORS[name]
        active = name in active_fingers
        thick = 3 if active else 2
        bright = tuple(min(c + 60, 255) for c in color) if active else (140, 140, 140)
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            if a in lm and b in lm:
                cv2.line(frame, lm[a], lm[b], bright, thick, cv2.LINE_AA)

    tip_map = {
        "thumb": THUMB_TIP, "index": INDEX_TIP, "middle": MIDDLE_TIP,
        "ring": RING_TIP, "pinky": PINKY_TIP,
    }
    for name in FINGER_ORDER:
        tip_idx = tip_map[name]
        if tip_idx not in lm:
            continue
        pt = lm[tip_idx]
        color = COLORS[name]
        active = name in active_fingers
        r = 9 if active else 6

        if active:
            for ring_r in range(15, r, -2):
                alpha = int(50 * (15 - ring_r) / 9)
                gc = tuple(min(c + alpha, 255) for c in color)
                cv2.circle(frame, pt, ring_r, gc, 2, cv2.LINE_AA)
            lbl = INSTRUMENT_LABELS[name]
            cv2.putText(frame, lbl, (pt[0] - 10, pt[1] - 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 2, cv2.LINE_AA)
        cv2.circle(frame, pt, r, color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (pt[0] - 2, pt[1] - 2), 2, (255, 255, 255),
                   cv2.FILLED, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    detector = HandDetector()
    orchestra = Orchestra()
    fps_display = 0
    fps_counter = 0
    fps_timer = time.time()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║            H A N D   O R C H E S T R A          ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Pinky  → Strings   (♩ C3)                     ║")
    print("║  Ring   → Flute     (♩ E4)                     ║")
    print("║  Middle → Brass     (♩ G4)                     ║")
    print("║  Index  → Harp      (♩ B4)                     ║")
    print("║  Thumb  → Cello     (♩ C5)                     ║")
    print("║                                                  ║")
    print("║  Hand height     → octave shift                  ║")
    print("║  Hand left/right → filter brightness             ║")
    print("║  Closed fist     → silence                       ║")
    print("║                                                  ║")
    print("║  Press 'q' in the video window to quit.          ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            detector.update(frame)
            lm = detector.get_landmarks()

            if lm:
                active = get_raised_fingers(lm)
                draw_skeleton(frame, lm, active)

                for name in FINGER_ORDER:
                    orchestra.set_active(name, name in active)

                index_y = lm[INDEX_TIP][1] / h
                octave_shift = round((0.5 - index_y) * 2)
                orchestra.set_octave(octave_shift)

                index_x = lm[INDEX_TIP][0] / w
                orchestra.set_filter(index_x)

                active_names = [n.upper() for n in FINGER_ORDER if n in active]
                if active_names:
                    label = " + ".join(active_names)
                    cv2.putText(frame, f"PLAYING: {label}", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Octave: {4+octave_shift}  Brightness: {index_x:.0%}",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "MUTE (fist)", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                orchestra.silence()
                cv2.putText(frame, "NO HAND DETECTED", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer = now
            cv2.putText(frame, f"FPS: {fps_display}", (w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Hand Orchestra (press q)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        orchestra.close()
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()