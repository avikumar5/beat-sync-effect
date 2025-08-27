# beatviz/renderer.py
from __future__ import annotations
from pathlib import Path
import random, uuid
import numpy as np
import cv2
import moviepy.editor as mpy

from .audio import extract_audio, detect_onsets, detect_beats, band_energies_over_time
from .tracking import TrackedPoint
from .drawing import draw_connections, draw_boxes_and_labels
from .utils import sample_size_bell

def _nearest_value(a: np.ndarray, v: float) -> float:
    if a.size == 0:
        return 0.0
    idx = np.searchsorted(a, v)
    if idx <= 0: return a[0]
    if idx >= len(a): return a[-1]
    return a[idx] if abs(a[idx] - v) < abs(a[idx-1] - v) else a[idx-1]

def _color_from_bands(low: float, mid: float, high: float) -> tuple[int, int, int]:
    """
    Map low/mid/high energies (0..1) to BGR.
    - Low -> Blue channel
    - Mid -> Green channel
    - High -> Red channel
    """
    B = int(40 + 215 * low)
    G = int(40 + 215 * mid)
    R = int(40 + 215 * high)
    return (B, G, R)

def render_tracked_effect(
    *,
    video_in: Path,
    video_out: Path,
    fps: float | None,
    pts_per_beat: int,
    ambient_rate: float,
    jitter_px: float,
    life_frames: int,
    min_size: int,
    max_size: int,
    neighbor_links: int,
    orb_fast_threshold: int,
    bell_width: float,
    seed: int | None,

    # New audio-driven knobs (safe defaults)
    beat_pulse_strength: float = 0.35,   # extra growth on beats (0..1)
    beat_tolerance_s: float = 0.05,      # how close to a beat (sec) to trigger pulse
    bass_size_boost: float = 0.25,       # how much low band scales squares (0..1)
    colorize_from_bands: bool = True,    # enable BGR based on (low,mid,high)
):
    """
    Same behavior as before, now enhanced with:
    - librosa beat detection for beat pulses
    - mel-band energies to colorize and scale overlays
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clip = mpy.VideoFileClip(str(video_in))
    if fps is None:
        fps = clip.fps

    # --- Precompute audio features ---
    wav_path = extract_audio(video_in)
    onset_times = detect_onsets(wav_path)              # keep if you still want onsets for spawns
    beat_times = detect_beats(wav_path)                # new: beat timestamps (sec)
    bands = band_energies_over_time(wav_path)          # new: per-time low/mid/high energies
    band_t = bands["t"]

    # ORB for seeding points
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)

    active: list[TrackedPoint] = []
    onset_idx = 0
    prev_gray: np.ndarray | None = None

    # for faster beat lookup
    beat_times = np.asarray(beat_times, dtype=np.float32)

    def make_frame(t: float):
        nonlocal prev_gray, onset_idx, active

        frame = clip.get_frame(t).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        f_idx = int(round(t * fps))

        # Get band energies for this timestamp (nearest sample)
        bt = _nearest_value(band_t, t)
        # map bt -> index in band_t quickly
        # (linear scan replaced by searchsorted inside _nearest_value)
        # Grab energies by nearest time
        idx = np.searchsorted(band_t, bt)
        idx = max(0, min(idx, len(band_t) - 1))
        low_e = float(bands["low"][idx])
        mid_e = float(bands["mid"][idx])
        high_e = float(bands["high"][idx])

        # Determine if this frame is close to a beat
        is_beat = False
        if beat_times.size:
            # quick nearest check
            nb = _nearest_value(beat_times, t)
            is_beat = abs(nb - t) <= beat_tolerance_s

        # 1) Track existing with LK
        if prev_gray is not None and active:
            prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=3)
            new_active: list[TrackedPoint] = []
            for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue
                x, y = new_pt
                if 0 <= x < w and 0 <= y < h and tp.life > 0:
                    tp.pos = new_pt
                    tp.life -= 1
                    if jitter_px > 0:
                        tp.pos += np.random.normal(0, jitter_px, size=2)
                        tp.pos[0] = np.clip(tp.pos[0], 0, w - 1)
                        tp.pos[1] = np.clip(tp.pos[1], 0, h - 1)
                    new_active.append(tp)
            active = new_active

        # 2) Spawn on onset (same as before)
        while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
            kps = orb.detect(gray, None)
            kps = sorted(kps, key=lambda k: k.response, reverse=True)
            target_spawn = random.randint(1, pts_per_beat)
            spawned = 0
            for kp in kps:
                if spawned >= target_spawn:
                    break
                x, y = kp.pt
                if any(np.linalg.norm(tp.pos - (x, y)) < 10 for tp in active):
                    continue
                base_size = sample_size_bell(min_size, max_size, bell_width)

                # Bass-driven size boost (subtle)
                boosted = base_size * (1.0 + bass_size_boost * low_e)

                # Label
                r = random.random()
                if r < 0.33:
                    label = ''.join(random.choices('ABCDEF0123456789', k=6))
                elif r < 0.66:
                    label = str(random.randint(1, 999))
                else:
                    label = str(uuid.uuid4())[:8]

                font_scale = random.uniform(1.0, 1.8)
                # Color either fixed choices or band-colorized
                if colorize_from_bands:
                    text_color = _color_from_bands(low_e, mid_e, high_e)
                else:
                    text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])

                vertical = random.random() < 0.25
                active.append(TrackedPoint((x, y), life_frames, int(boosted), label, font_scale, text_color, vertical))
                spawned += 1
            onset_idx += 1

        # 2b) Ambient spawns (noise)
        if ambient_rate > 0:
            noise_n = np.random.poisson(ambient_rate / fps)
            for _ in range(noise_n):
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                base_size = sample_size_bell(min_size, max_size, bell_width)
                boosted = base_size * (1.0 + bass_size_boost * low_e)
                label = random.choice([
                    ''.join(random.choices('ABCDEF0123456789', k=6)),
                    str(random.randint(1, 999)),
                    str(uuid.uuid4())[:8],
                ])
                font_scale = random.uniform(1.0, 1.8)
                text_color = _color_from_bands(low_e, mid_e, high_e) if colorize_from_bands else random.choice([(255,255,255),(0,0,0),(255,0,255)])
                vertical = random.random() < 0.25
                active.append(TrackedPoint((x, y), life_frames, int(boosted), label, font_scale, text_color, vertical))

        # 3) Draw neighbor links (thicker on beat)
        line_thickness = 1 + (2 if is_beat else 0)
        coords = [tp.pos for tp in active]
        for i, p in enumerate(coords):
            dists = [(j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i]
            dists.sort(key=lambda x: x[1])
            for j, _ in dists[:neighbor_links]:
                color = _color_from_bands(low_e, mid_e, high_e) if colorize_from_bands else (255, 255, 255)
                cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), color, line_thickness)

        # 4) Draw squares & labels (pulse on beat)
        for tp in active:
            x, y = tp.pos
            s = tp.size
            # Beat pulse growth
            if is_beat:
                s = int(s * (1.0 + beat_pulse_strength))
            tl = (int(x - s // 2), int(y - s // 2))
            br = (int(x + s // 2), int(y + s // 2))
            tl_clamped = (max(0, tl[0]), max(0, tl[1]))
            br_clamped = (min(w, br[0]), min(h, br[1]))

            # pop invert
            roi = frame[tl_clamped[1]:br_clamped[1], tl_clamped[0]:br_clamped[0]]
            if roi.size:
                frame[tl_clamped[1]:br_clamped[1], tl_clamped[0]:br_clamped[0]] = 255 - roi

            box_color = _color_from_bands(low_e, mid_e, high_e) if colorize_from_bands else (255, 255, 255)
            cv2.rectangle(frame, tl, br, box_color, 1)

            # text
            if tp.vertical:
                y_cursor = tl[1] + 2
                line_h = int(12 * tp.font_scale)
                for ch in tp.label:
                    cv2.putText(frame, ch, (tl[0] + 2, y_cursor),
                                cv2.FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv2.LINE_AA)
                    y_cursor += line_h
                    if y_cursor > br[1] - 2:
                        break
            else:
                cv2.putText(frame, tp.label, (tl[0] + 2, br[1] - 4),
                            cv2.FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv2.LINE_AA)

        prev_gray = gray
        return frame

    out_clip = mpy.VideoClip(make_frame, duration=clip.duration)
    out_clip = out_clip.set_audio(clip.audio)
    out_clip.write_videofile(str(video_out), fps=fps, codec="libx264", audio_codec="aac")
