#!/usr/bin/env python3
"""
beatmI / twin — deconstruct audio, MIDI and Resonarium states into a digital twin.

The twin is a probability model of how one person makes beats: where their kick
lands, which scale degrees they reach for, how their lines move, how fast they
work. SPINE reconstructs from it.

Nothing here needs pip. Audio decoding goes through ffmpeg; the DSP is numpy and
scipy only. Files are read locally and never leave the machine.

    python3 analyze.py ~/Music/*.mp3 -o twin.json
    python3 analyze.py --add ~/Desktop/*.wav -o twin.json
    python3 analyze.py --resonarium ~/Downloads/state.json -o twin.json
    python3 analyze.py --inspect twin.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import subprocess
import sys
import datetime as _dt

import numpy as np
from scipy.ndimage import median_filter, maximum_filter1d, uniform_filter1d

# ── musical vocabulary — must stay in step with SPINE's SCALES table ──────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES = {
    "minor":     [0, 2, 3, 5, 7, 8, 10],
    "phrygian":  [0, 1, 3, 5, 7, 8, 10],
    "phrydom":   [0, 1, 4, 5, 7, 8, 10],
    "harmonic":  [0, 2, 3, 5, 7, 8, 11],
    "dorian":    [0, 2, 3, 5, 7, 9, 10],
    "minpent":   [0, 3, 5, 7, 10],
    "hirajoshi": [0, 2, 3, 7, 8],
    "major":     [0, 2, 4, 5, 7, 9, 11],
}

# tonal weight per interval — root and fifth anchor a key, thirds colour it
DEGREE_WEIGHT = {0: 3.2, 7: 2.1, 3: 1.7, 4: 1.7, 10: 1.2, 11: 1.2, 5: 1.1}

SR = 22050
N_FFT = 2048
HOP = 512
FPS = SR / HOP
STEPS = 16
VOICES = ["kick", "snare", "hat", "sub"]

# band edges in Hz for percussive voice separation
BANDS = {
    "sub":   (25, 70),
    "kick":  (45, 130),
    "snare": (180, 2600),
    "hat":   (6500, 11000),
}


# ── io ───────────────────────────────────────────────────────────────────────

def decode(path: str, seconds: float | None = 120.0) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at SR."""
    cmd = ["ffmpeg", "-v", "quiet", "-i", path]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not decode {os.path.basename(path)}")
    x = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if x.size < SR:
        raise RuntimeError("less than one second of audio")
    peak = float(np.max(np.abs(x)))
    return x / peak if peak > 0 else x


def stft_mag(x: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram, shape (bins, frames)."""
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    win = np.hanning(N_FFT).astype(np.float32)
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=1)).T.astype(np.float32)


def hpss(S: np.ndarray, k: int = 17):
    """Median-filter harmonic/percussive separation.

    Smoothing across time keeps sustained partials; smoothing across frequency
    keeps broadband transients. Soft masks split the spectrogram between them.
    """
    H = median_filter(S, size=(1, k), mode="nearest")
    P = median_filter(S, size=(k, 1), mode="nearest")
    eps = 1e-9
    h2, p2 = H ** 2, P ** 2
    total = h2 + p2 + eps
    return S * (h2 / total), S * (p2 / total)


# ── rhythm ───────────────────────────────────────────────────────────────────

def band_flux(S: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Half-wave rectified spectral flux inside a frequency band."""
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        return np.zeros(S.shape[1] - 1, dtype=np.float32)
    band = np.log1p(S[sel] * 12.0)
    flux = np.maximum(0.0, np.diff(band, axis=1)).sum(axis=0)
    if flux.max() > 0:
        flux = flux / flux.max()
    return flux.astype(np.float32)


def estimate_tempo(env: np.ndarray) -> tuple[float, float]:
    """Autocorrelate the onset envelope; return (bpm, confidence 0..1)."""
    e = env - env.mean()
    if not np.any(e):
        return 0.0, 0.0
    n = int(2 ** math.ceil(math.log2(len(e) * 2)))
    spec = np.fft.rfft(e, n)
    ac = np.fft.irfft(spec * np.conj(spec), n)[: len(e)]
    if ac[0] > 0:
        ac /= ac[0]

    lo_lag = max(2, int(FPS * 60 / 200.0))   # 200 BPM
    hi_lag = min(len(ac) - 1, int(FPS * 60 / 55.0))   # 55 BPM
    if hi_lag <= lo_lag:
        return 0.0, 0.0

    lags = np.arange(lo_lag, hi_lag)
    bpms = 60.0 * FPS / lags
    # producers of this kind work in a band; a broad log-normal prior around 140
    # keeps autocorrelation from settling on a half- or double-time lag
    prior = np.exp(-0.5 * (np.log(bpms / 140.0) / 0.42) ** 2)
    score = ac[lo_lag:hi_lag] * prior

    best = int(np.argmax(score))
    lag = float(lags[best])
    # integer lags are coarse up here - at 140 BPM one frame of lag is nearly 4 BPM -
    # so interpolate a parabola through the peak to recover the fractional lag
    if 0 < best < len(score) - 1:
        y0, y1, y2 = score[best - 1], score[best], score[best + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            lag += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))

    bpm = 60.0 * FPS / lag
    conf = float(np.clip(score[best] / (score.mean() + 1e-9) / 6.0, 0, 1))
    return bpm, conf


def grid_phase(env_full: np.ndarray, env_low: np.ndarray, bpm: float) -> float:
    """Locate the downbeat by rotating the bar until it matches metric strength.

    Searching only a beat's worth of offsets lets the grid lock onto beat 2 and
    call it beat 1, which silently rotates every pattern by four steps. The
    search has to cover a whole bar, and it has to score against where energy
    belongs metrically rather than simply where energy is loudest. Weighting the
    low band favours the kick, which is what usually marks bar one.
    """
    step = 60.0 / bpm / 4.0 * FPS
    bar = step * STEPS
    if bar < 8 or len(env_full) < bar * 2:
        return 0.0

    mix = 0.6 * env_low + 0.4 * env_full if env_low.size == env_full.size else env_full
    positions = np.arange(0, len(mix) - bar, bar)
    if positions.size == 0:
        return 0.0

    tmpl = METRIC - METRIC.mean()
    tnorm = math.sqrt(float((tmpl ** 2).sum()))
    w = max(1, int(step * 0.4))

    best_off, best_score = 0.0, -np.inf
    for off in range(0, max(1, int(round(bar)))):
        vals = np.zeros(STEPS)
        for s in range(STEPS):
            idx = np.round(positions + off + s * step).astype(int)
            idx = idx[(idx >= 0) & (idx < len(mix))]
            if idx.size == 0:
                continue
            lo = np.maximum(idx - w, 0)
            vals[s] = float(np.mean([mix[a:b + 1].max()
                                     for a, b in zip(lo, np.minimum(idx + w, len(mix) - 1))]))
        vc = vals - vals.mean()
        n = math.sqrt(float((vc ** 2).sum())) * tnorm
        score = float((vc * tmpl).sum() / n) if n > 0 else -np.inf
        if score > best_score:
            best_score, best_off = score, float(off)
    return best_off


def onset_peaks(env: np.ndarray, min_sep: int = 3) -> np.ndarray:
    """Frame indices of genuine transients.

    A dense mix never lets the flux envelope fall to zero, so raw magnitude says
    more about loudness than about hits. Subtracting a local mean whitens that
    floor away, and a minimum separation stops one transient being counted twice.
    """
    if env.size < 8:
        return np.array([], dtype=int)
    local = uniform_filter1d(env, size=max(3, int(FPS * 0.4)), mode="nearest")
    d = np.maximum(0.0, env - local - 0.015)
    if d.max() <= 0:
        return np.array([], dtype=int)
    strong = d >= maximum_filter1d(d, size=min_sep * 2 + 1, mode="nearest")
    return np.flatnonzero(strong & (d > d[d > 0].mean() * 0.55))


def fold_to_grid(env: np.ndarray, bpm: float, phase: float) -> np.ndarray:
    """Per-step hit probability across the bar: how often step k actually fires.

    Counting bars-with-a-hit rather than summing energy keeps the result on a
    scale that means something when SPINE samples from it later.
    """
    step = 60.0 / bpm / 4.0 * FPS
    bar = step * STEPS
    if step < 1 or len(env) < bar:
        return np.zeros(STEPS)

    n_bars = max(1, int((len(env) - phase) / bar))
    peaks = onset_peaks(env, min_sep=max(2, int(step * 0.6)))
    if peaks.size == 0:
        return np.zeros(STEPS)

    pos = (peaks - phase) / step
    nearest = np.round(pos)
    keep = np.abs(pos - nearest) < 0.32          # ignore what lands off any 16th
    if not keep.any():
        return np.zeros(STEPS)

    slots = nearest[keep].astype(int)
    bars = slots // STEPS
    hits = {(int(b), int(s % STEPS)) for b, s in zip(bars, slots)}
    out = np.zeros(STEPS)
    for _, s in hits:
        out[s] += 1.0
    return np.clip(out / n_bars, 0.0, 1.0)


def estimate_swing(env: np.ndarray, bpm: float, phase: float) -> float:
    """Mean late-ness of odd 16ths, as a fraction of a step (0..0.6)."""
    step = 60.0 / bpm / 4.0 * FPS
    if step < 2:
        return 0.0
    peaks = env >= np.maximum(maximum_filter1d(env, size=5) * 0.98, 0.12)
    frames = np.flatnonzero(peaks)
    if frames.size < 8:
        return 0.0
    pos = (frames - phase) / step
    nearest = np.round(pos)
    dev = pos - nearest
    odd = (nearest.astype(int) % 2) == 1
    good = odd & (np.abs(dev) < 0.4)
    if good.sum() < 4:
        return 0.0
    return float(np.clip(np.mean(dev[good]) * 2.0, 0.0, 0.6))


# metric strength of each 16th — downbeat strongest, odd 16ths weakest
METRIC = np.array([1.0, .15, .4, .15, .75, .15, .4, .15,
                   .9, .15, .4, .15, .75, .15, .4, .25])


def syncopation(grid: np.ndarray) -> float:
    """0 = everything on strong beats, 1 = everything off them."""
    total = grid.sum()
    if total <= 0:
        return 0.0
    return float(np.clip(1.0 - (grid * METRIC).sum() / total, 0.0, 1.0))


# ── pitch ────────────────────────────────────────────────────────────────────

def chromagram(H: np.ndarray, lo: float = 190.0, hi: float = 2600.0) -> np.ndarray:
    """Per-frame 12-bin chroma from the harmonic spectrogram, shape (12, frames).

    The low bound keeps 808s and kick fundamentals from swamping the profile —
    otherwise every track reads as its own root and nothing else.
    """
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    sel = (freqs >= lo) & (freqs <= hi)
    f = freqs[sel]
    pc = np.round(12 * np.log2(f / 440.0) + 69).astype(int) % 12
    mag = H[sel] ** 0.6                     # compress; loud partials shouldn't dominate
    out = np.zeros((12, mag.shape[1]), dtype=np.float32)
    for k in range(12):
        m = pc == k
        if m.any():
            out[k] = mag[m].sum(axis=0)
    return out


def scale_template(ivs: list[int]) -> np.ndarray:
    """Krumhansl-style tonal hierarchy for one scale.

    Non-members sit at a low floor rather than zero, because real music passes
    through notes outside the scale without changing key.
    """
    t = np.full(12, 2.0)
    for i in ivs:
        t[i] += 3.6
    t[0] += 2.6                                  # tonic dominates
    if 7 in ivs:
        t[7] += 1.6                              # then the fifth
    for third in (3, 4):
        if third in ivs:
            t[third] += 1.2                      # the third decides the mode's colour
    for colour in (1, 2, 6, 9, 11):              # each mode's characteristic tone
        if colour in ivs:
            t[colour] += 0.45
    return t


_TEMPLATES = {name: scale_template(ivs) for name, ivs in SCALES.items()}


def detect_key(chroma: np.ndarray) -> tuple[int, str, float]:
    """Correlate the chroma against every (root, scale) pair in SPINE's vocabulary.

    Pearson correlation rather than an in-scale sum minus an out-of-scale penalty:
    the sum form quietly rewards whichever scale has the most notes, which made
    everything come back as major.
    """
    if chroma.size == 0 or chroma.sum() <= 0:
        return 0, "minor", 0.0
    # normalise per frame first, so a loud drop cannot outvote a quiet verse
    if chroma.shape[1] > 1:
        norm = chroma.sum(axis=0, keepdims=True)
        frames = np.divide(chroma, norm, out=np.zeros_like(chroma), where=norm > 1e-9)
        v = frames.sum(axis=1)
    else:
        v = chroma[:, 0].astype(float)
    if v.sum() <= 0:
        return 0, "minor", 0.0
    v = v / v.sum()
    if v.std() <= 0:
        return 0, "minor", 0.0

    results = []
    for root in range(12):
        rot = np.roll(v, -root)
        rc = rot - rot.mean()
        for name, tmpl in _TEMPLATES.items():
            tc = tmpl - tmpl.mean()
            denom = np.sqrt((rc ** 2).sum() * (tc ** 2).sum())
            results.append((float((rc * tc).sum() / denom) if denom > 0 else 0.0, root, name))

    results.sort(key=lambda r: -r[0])
    best = results[0]
    # confidence compares the winner against the best root that is not its own
    other = next((r[0] for r in results[1:] if r[1] != best[1]), 0.0)
    conf = float(np.clip((best[0] - other) / max(abs(best[0]), 1e-9) * 2.2, 0, 1))
    return best[1], best[2], conf


def degree_sequence(chroma: np.ndarray, root: int, scale: str,
                    bpm: float, phase: float) -> list[int]:
    """Dominant in-scale degree per 16th step — the melodic skeleton.

    This is not a transcription. It is the sequence of scale degrees carrying
    the most harmonic energy, which is what the twin's motion model needs.
    """
    ivs = SCALES[scale]
    step = 60.0 / bpm / 4.0 * FPS
    if step < 1 or chroma.shape[1] < step * 2:
        return []

    energy = chroma.sum(axis=0)
    live = energy[energy > 0]
    floor = float(np.median(live)) * 0.55 if live.size else 0.0

    n_steps = int((chroma.shape[1] - phase) / step)
    seq: list[int] = []
    for i in range(max(0, n_steps)):
        a = int(phase + i * step)
        b = max(a + 1, int(phase + (i + 1) * step))
        if b > chroma.shape[1]:
            break
        frame = chroma[:, a:b].mean(axis=1)
        # a step quieter than the track's typical frame is a rest, not a note
        if frame.sum() <= 0 or frame.sum() < floor:
            seq.append(-1)
            continue
        frame = frame / frame.sum()
        val, deg = max((frame[(root + iv) % 12], d) for d, iv in enumerate(ivs))
        # and a winner that does not clear the frame's own spread is just noise
        seq.append(deg if val > frame.mean() + 1.15 * frame.std() else -1)
    return seq


def note_runs(seq: list[int]) -> list[tuple[int, int]]:
    """Group a degree sequence into (degree, length) notes.

    A held note occupies many steps but is one note. Counting voiced steps
    instead of note onsets reports a four-note bar as a twelve-note bar.
    """
    runs: list[tuple[int, int]] = []
    prev: int | None = None
    for d in seq:
        if d < 0:
            prev = None                   # a rest ends whatever was sounding
            continue
        if prev == d and runs:
            runs[-1] = (d, runs[-1][1] + 1)
        else:
            runs.append((d, 1))
        prev = d
    return runs


def motion_model(seq: list[int], n_deg: int):
    """Markov transitions and interval spread over scale degrees."""
    trans = np.zeros((n_deg, n_deg))
    intervals: dict[int, float] = {}
    hist = np.zeros(n_deg)
    prev = -1
    for d in seq:
        if d < 0:
            prev = -1
            continue
        hist[d] += 1
        if prev >= 0 and prev != d:
            trans[prev][d] += 1
            delta = d - prev
            if delta > n_deg // 2:
                delta -= n_deg
            if delta < -(n_deg // 2):
                delta += n_deg
            intervals[delta] = intervals.get(delta, 0) + 1
        prev = d
    return trans, intervals, hist


# ── per-file analysis ────────────────────────────────────────────────────────

def analyze_audio(path: str, seconds: float | None) -> dict:
    x = decode(path, seconds)
    S = stft_mag(x)
    H, P = hpss(S)

    env = {v: band_flux(P, *BANDS[v]) for v in VOICES}
    full = band_flux(P, 30, 11000)
    bpm, conf = estimate_tempo(full)
    if bpm <= 0:
        raise RuntimeError("no usable pulse found")

    phase = grid_phase(full, env["kick"], bpm)
    grids = {v: fold_to_grid(env[v], bpm, phase) for v in VOICES}
    swing = estimate_swing(env["hat"] if env["hat"].max() > 0 else full, bpm, phase)

    chroma = chromagram(H)
    root, scale, kconf = detect_key(chroma)
    seq = degree_sequence(chroma, root, scale, bpm, phase)
    trans, intervals, dhist = motion_model(seq, len(SCALES[scale]))

    runs = note_runs(seq)
    lens = [n for _, n in runs] or [1]

    return {
        "kind": "audio",
        "file": os.path.basename(path),
        "seconds": round(len(x) / SR, 1),
        "bpm": round(bpm, 1),
        "bpm_confidence": round(conf, 3),
        "swing": round(swing, 3),
        "key": root,
        "scale": scale,
        "key_confidence": round(kconf, 3),
        "chroma": (chroma.sum(axis=1) / max(chroma.sum(), 1e-9)).round(5).tolist(),
        "rhythm": {v: grids[v].round(4).tolist() for v in VOICES},
        "syncopation": round(float(np.mean([syncopation(grids[v]) for v in VOICES])), 3),
        "degree_hist": (dhist / max(dhist.sum(), 1)).round(4).tolist(),
        "transitions": trans.tolist(),
        "intervals": {str(k): v for k, v in intervals.items()},
        "notes_per_bar": round(float(np.clip(len(runs) / max(len(seq) / STEPS, 1), 0, 16)), 2),
        "avg_len_steps": round(float(np.clip(np.mean(lens), 1, 8)), 2),
        "weight": round(0.35 + 0.65 * conf, 3),
    }


# ── MIDI ─────────────────────────────────────────────────────────────────────

def _vlq(buf: bytes, i: int):
    n = 0
    while True:
        b = buf[i]
        i += 1
        n = (n << 7) | (b & 0x7F)
        if not b & 0x80:
            return n, i


def parse_midi(path: str):
    """Minimal type-0/1 reader. Returns (notes, ticks_per_quarter, bpm)."""
    buf = open(path, "rb").read()
    if buf[:4] != b"MThd":
        raise RuntimeError("not a MIDI file")
    ntrk, div = struct.unpack(">HH", buf[10:14])
    if div & 0x8000:
        raise RuntimeError("SMPTE timing is not supported")
    pos, notes, bpm = 14, [], 120.0
    for _ in range(ntrk):
        if buf[pos:pos + 4] != b"MTrk":
            break
        length = struct.unpack(">I", buf[pos + 4:pos + 8])[0]
        p, end, t, status = pos + 8, pos + 8 + length, 0, 0
        open_notes: dict[tuple[int, int], int] = {}
        while p < end:
            delta, p = _vlq(buf, p)
            t += delta
            if buf[p] & 0x80:
                status = buf[p]
                p += 1
            if status == 0xFF:
                mt = buf[p]; p += 1
                ln, p = _vlq(buf, p)
                if mt == 0x51 and ln == 3:
                    us = int.from_bytes(buf[p:p + 3], "big")
                    if us:
                        bpm = 60_000_000 / us
                p += ln
            elif status in (0xF0, 0xF7):
                ln, p = _vlq(buf, p)
                p += ln
            else:
                hi, ch = status & 0xF0, status & 0x0F
                if hi in (0x80, 0x90):
                    pitch, vel = buf[p], buf[p + 1]; p += 2
                    if hi == 0x90 and vel > 0:
                        open_notes[(ch, pitch)] = t
                    else:
                        st = open_notes.pop((ch, pitch), None)
                        if st is not None:
                            notes.append({"ch": ch, "pitch": pitch,
                                          "start": st, "dur": max(1, t - st)})
                elif hi in (0xA0, 0xB0, 0xE0):
                    p += 2
                else:
                    p += 1
        pos = end
    if not notes:
        raise RuntimeError("no notes found")
    return notes, div, bpm


def analyze_midi(path: str) -> dict:
    notes, tpq, bpm = parse_midi(path)
    tps = tpq / 4.0                                   # ticks per 16th step

    drums = [n for n in notes if n["ch"] == 9]
    pitched = [n for n in notes if n["ch"] != 9]

    gm = {"kick": {35, 36}, "snare": {37, 38, 39, 40},
          "hat": {42, 44, 46, 51, 59}, "sub": {41, 43, 45, 47}}
    grids = {v: np.zeros(STEPS) for v in VOICES}
    for n in drums:
        slot = int(round(n["start"] / tps)) % STEPS
        for v, ps in gm.items():
            if n["pitch"] in ps:
                grids[v][slot] += 1
    # a MIDI kick with no GM mapping is still a kick — fall back to the low register
    if not drums and pitched:
        for n in (x for x in pitched if x["pitch"] < 45):
            grids["sub"][int(round(n["start"] / tps)) % STEPS] += 1
    for v in VOICES:
        if grids[v].max() > 0:
            grids[v] /= grids[v].max()

    chroma = np.zeros((12, 1), dtype=np.float32)
    for n in pitched:
        chroma[n["pitch"] % 12, 0] += n["dur"] / tpq
    root, scale, kconf = detect_key(chroma)

    ivs = SCALES[scale]
    lead = sorted((n for n in pitched if n["pitch"] >= 48), key=lambda n: n["start"])
    seq: list[int] = []
    if lead:
        span = int(max(n["start"] for n in lead) / tps) + 1
        by_slot: dict[int, int] = {}
        for n in lead:
            by_slot.setdefault(int(round(n["start"] / tps)), n["pitch"])
        for i in range(span):
            p = by_slot.get(i)
            if p is None:
                seq.append(-1)
            else:
                pc = (p - root) % 12
                seq.append(ivs.index(pc) if pc in ivs
                           else min(range(len(ivs)), key=lambda d: abs(ivs[d] - pc)))
    trans, intervals, dhist = motion_model(seq, len(ivs))
    lens = [n["dur"] / tps for n in lead] or [2.0]

    return {
        "kind": "midi",
        "file": os.path.basename(path),
        "bpm": round(bpm, 1),
        "bpm_confidence": 1.0,
        "swing": 0.0,
        "key": root,
        "scale": scale,
        "key_confidence": round(kconf, 3),
        "chroma": (chroma[:, 0] / max(chroma.sum(), 1e-9)).round(5).tolist(),
        "rhythm": {v: grids[v].round(4).tolist() for v in VOICES},
        "syncopation": round(float(np.mean([syncopation(grids[v]) for v in VOICES])), 3),
        "degree_hist": (dhist / max(dhist.sum(), 1)).round(4).tolist(),
        "transitions": trans.tolist(),
        "intervals": {str(k): v for k, v in intervals.items()},
        "notes_per_bar": round(len(lead) / max(len(seq) / STEPS, 1), 2) if seq else 0.0,
        "avg_len_steps": round(float(np.clip(np.mean(lens), 1, 8)), 2),
        "weight": 1.4,                                # exact data outranks estimates
    }


# ── Resonarium bridge ────────────────────────────────────────────────────────

def analyze_resonarium(path: str) -> dict:
    """Fold a Resonarium state into musical terms.

    Carrier, sweep and tone frequencies are already pitches — they collapse onto
    pitch classes directly. A binaural beat frequency is a rate in Hz, so it is
    a tempo: 2.33 Hz is 140 BPM once you fold it into a musical range.
    """
    st = json.load(open(path))
    schema = str(st.get("schema", ""))
    chroma = np.zeros((12, 1), dtype=np.float32)
    carriers, beats, tempos = [], [], []

    def add_freq(f, lvl):
        if f and f > 20:
            chroma[int(round(12 * math.log2(f / 440.0) + 69)) % 12, 0] += max(lvl, 0.05)
            carriers.append(round(float(f), 2))

    for s in st.get("sweeps", []):
        if s.get("on", True):
            add_freq(s.get("f"), s.get("lvl", 0.3))
    for b in st.get("bins", []):
        if b.get("on", True):
            add_freq(b.get("carrier"), b.get("lvl", 0.3))
            hz = b.get("beat")
            if hz and hz > 0:
                beats.append(round(float(hz), 3))
                bpm = hz * 60.0
                while bpm < 70:
                    bpm *= 2
                while bpm > 190:
                    bpm /= 2
                tempos.append(bpm)
    for s in st.get("singles", []):
        if s.get("on", True):
            add_freq(s.get("f"), s.get("lvl", 0.3))

    # hologram variant: modes carry ratios rather than Hz, over a phi bedrock
    for m in st.get("modes", []):
        if m.get("on", True) and m.get("freq"):
            add_freq(110.0 * float(m["freq"]), m.get("amplitude", 0.2))

    if chroma.sum() <= 0:
        raise RuntimeError("no active voices in this state")

    root, scale, kconf = detect_key(chroma)
    bpm = float(np.median(tempos)) if tempos else 0.0

    return {
        "kind": "resonarium",
        "file": os.path.basename(path),
        "schema": schema,
        "bpm": round(bpm, 1),
        "bpm_confidence": 0.8 if tempos else 0.0,
        "swing": 0.0,
        "key": root,
        "scale": scale,
        "key_confidence": round(kconf, 3),
        "chroma": (chroma[:, 0] / chroma.sum()).round(5).tolist(),
        "rhythm": {v: [0.0] * STEPS for v in VOICES},
        "syncopation": 0.0,
        "degree_hist": [0.0] * len(SCALES[scale]),
        "transitions": [[0.0] * len(SCALES[scale]) for _ in SCALES[scale]],
        "intervals": {},
        "notes_per_bar": 0.0,
        "avg_len_steps": 2.0,
        "carriers": sorted(set(carriers)),
        "beats": sorted(set(beats)),
        "natal_seed": st.get("natalSeed"),
        "weight": 0.9,
    }


# ── twin accumulation ────────────────────────────────────────────────────────

def blank_twin() -> dict:
    return {
        "schema": "beatmi.twin.v1",
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "updated": None,
        "sources": [],
        "tempo": {"mean": 0.0, "median": 0.0, "spread": 0.0, "hist": {}},
        "swing": 0.0,
        "key_weights": [0.0] * 12,
        "scale_weights": {},
        "rhythm": {v: [0.0] * STEPS for v in VOICES},
        "density": {v: 0.0 for v in VOICES},
        "syncopation": 0.0,
        "melody": {
            "degree_hist": [], "transitions": [], "intervals": {},
            "notes_per_bar": 0.0, "avg_len_steps": 2.0,
        },
        "resonarium": {"carriers": [], "beats": [], "natal_seed": None},
    }


def consolidate(twin: dict) -> dict:
    """Recompute every aggregate from the source list."""
    src = twin["sources"]
    if not src:
        return twin
    w = np.array([s.get("weight", 1.0) for s in src], dtype=float)

    # tempo — confidence-weighted, ignoring sources with no pulse
    tw = [(s["bpm"], s.get("weight", 1.0)) for s in src
          if s.get("bpm", 0) > 0 and s.get("bpm_confidence", 0) > 0.12]
    if tw:
        b = np.array([t[0] for t in tw]); bw = np.array([t[1] for t in tw])
        twin["tempo"] = {
            "mean": round(float(np.average(b, weights=bw)), 1),
            "median": round(float(np.median(b)), 1),
            "spread": round(float(np.std(b)), 1),
            "hist": {str(int(k)): round(float(v), 4) for k, v in
                     zip(*np.unique(np.round(b / 5) * 5, return_counts=True))},
        }
        total = sum(twin["tempo"]["hist"].values())
        twin["tempo"]["hist"] = {k: round(v / total, 4)
                                 for k, v in twin["tempo"]["hist"].items()}

    sw = [(s.get("swing", 0.0), s.get("weight", 1.0)) for s in src if s.get("kind") == "audio"]
    twin["swing"] = round(float(np.average([x[0] for x in sw],
                                           weights=[x[1] for x in sw])), 3) if sw else 0.0

    # key preference from raw chroma, so partial agreement still counts
    kw = np.zeros(12)
    for s, ww in zip(src, w):
        kw += np.array(s.get("chroma", [0] * 12)) * ww
    twin["key_weights"] = (kw / kw.sum()).round(5).tolist() if kw.sum() > 0 else [0.0] * 12

    sc: dict[str, float] = {}
    for s, ww in zip(src, w):
        sc[s["scale"]] = sc.get(s["scale"], 0.0) + ww * (0.35 + s.get("key_confidence", 0))
    tot = sum(sc.values()) or 1.0
    twin["scale_weights"] = {k: round(v / tot, 4)
                             for k, v in sorted(sc.items(), key=lambda x: -x[1])}

    # rhythm — only from sources that actually carry drums
    for v in VOICES:
        acc = np.zeros(STEPS); tw_ = 0.0
        for s, ww in zip(src, w):
            g = np.array(s.get("rhythm", {}).get(v, [0] * STEPS), dtype=float)
            if g.sum() > 0:
                acc += g * ww; tw_ += ww
        if tw_ > 0:
            acc /= tw_
            if acc.max() > 0:
                acc /= acc.max()
        twin["rhythm"][v] = acc.round(4).tolist()
        twin["density"][v] = round(float((acc > 0.28).sum()), 2)

    sy = [(s.get("syncopation", 0), ww) for s, ww in zip(src, w) if s.get("syncopation", 0) > 0]
    twin["syncopation"] = round(float(np.average([x[0] for x in sy],
                                                 weights=[x[1] for x in sy])), 3) if sy else 0.0

    # melody model lives in the dominant scale's degree space
    dom = next(iter(twin["scale_weights"]), "minor")
    n = len(SCALES[dom])
    dh = np.zeros(n); tr = np.zeros((n, n)); iv: dict[str, float] = {}
    npb, als, mw = 0.0, 0.0, 0.0
    for s, ww in zip(src, w):
        if s.get("scale") != dom or not s.get("degree_hist"):
            continue
        h = np.array(s["degree_hist"], dtype=float)
        t = np.array(s["transitions"], dtype=float)
        if h.shape[0] != n or t.shape != (n, n):
            continue
        dh += h * ww
        tr += t * ww
        for k, val in s.get("intervals", {}).items():
            iv[k] = iv.get(k, 0.0) + val * ww
        npb += s.get("notes_per_bar", 0) * ww
        als += s.get("avg_len_steps", 2) * ww
        mw += ww
    if mw > 0:
        rows = tr.sum(axis=1, keepdims=True)
        twin["melody"] = {
            "scale": dom,
            "degree_hist": (dh / dh.sum()).round(4).tolist() if dh.sum() > 0 else [0.0] * n,
            "transitions": np.where(rows > 0, tr / np.maximum(rows, 1e-9), 1.0 / n).round(4).tolist(),
            "intervals": {k: round(v / sum(iv.values()), 4) for k, v in
                          sorted(iv.items(), key=lambda x: -x[1])} if iv else {},
            "notes_per_bar": round(npb / mw, 2),
            "avg_len_steps": round(als / mw, 2),
        }

    res = {"carriers": [], "beats": [], "natal_seed": twin["resonarium"].get("natal_seed")}
    for s in src:
        if s.get("kind") == "resonarium":
            res["carriers"] += s.get("carriers", [])
            res["beats"] += s.get("beats", [])
            if s.get("natal_seed") is not None:
                res["natal_seed"] = s["natal_seed"]
    res["carriers"] = sorted(set(res["carriers"]))[:32]
    res["beats"] = sorted(set(res["beats"]))[:16]
    twin["resonarium"] = res

    twin["updated"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return twin


# ── reporting ────────────────────────────────────────────────────────────────

try:
    import braille as _braille
except ImportError:                          # report degrades, analysis does not
    _braille = None


def bar(v: float, width: int = 16) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    return blocks[min(len(blocks) - 1, max(0, int(v * len(blocks))))] * 1


def report(twin: dict) -> str:
    L = []
    src = twin["sources"]
    kinds: dict[str, int] = {}
    for s in src:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    L.append(f"  sources    {len(src)}  ({', '.join(f'{v} {k}' for k, v in kinds.items())})")
    t = twin["tempo"]
    L.append(f"  tempo      {t['median']:.0f} BPM median, {t['mean']:.0f} mean, ±{t['spread']:.0f}")
    L.append(f"  swing      {twin['swing'] * 100:.0f}%")
    top_keys = sorted(enumerate(twin["key_weights"]), key=lambda x: -x[1])[:4]
    L.append("  keys       " + ", ".join(f"{NOTE_NAMES[i]} {v * 100:.0f}%" for i, v in top_keys))
    L.append("  scales     " + ", ".join(f"{k} {v * 100:.0f}%"
                                         for k, v in list(twin["scale_weights"].items())[:4]))
    L.append(f"  syncopation {twin['syncopation']:.2f}")
    L.append("")
    L.append("  rhythmic fingerprint        1   .   .   .   2   .   .   .   3   .   .   .   4   .   .   .")
    for v in VOICES:
        g = twin["rhythm"][v]
        spark = f"   {_braille.sparkline(g)}" if _braille else ""
        L.append(f"    {v:<8}" + " " * 16 + "  ".join(bar(x) for x in g) + spark)
    if _braille:
        code = _braille.pattern_to_braille(twin["rhythm"], threshold=0.45)
        L.append(f"    {'barcode':<8}" + " " * 16 + code
                 + "   (paste into SPINE or braille.py --pattern)")
    m = twin.get("melody", {})
    if m.get("degree_hist"):
        L.append("")
        ivs = SCALES.get(m.get("scale", "minor"), SCALES["minor"])
        names = {0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
                 6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"}
        L.append(f"  melodic shape ({m.get('scale')})")
        L.append("    degrees  " + "  ".join(
            f"{names[iv]}{bar(h)}" for iv, h in zip(ivs, m["degree_hist"])))
        top = list(m.get("intervals", {}).items())[:5]
        if top:
            L.append("    moves    " + ", ".join(
                f"{'+' if int(k) > 0 else ''}{k} steps {v * 100:.0f}%" for k, v in top))
        L.append(f"    density  {m.get('notes_per_bar', 0):.1f} notes/bar, "
                 f"{m.get('avg_len_steps', 0):.1f} steps long")
    r = twin.get("resonarium", {})
    if r.get("carriers"):
        L.append("")
        L.append(f"  resonarium {len(r['carriers'])} carriers, {len(r['beats'])} beat rates"
                 + (f", natal seed {r['natal_seed']}" if r.get("natal_seed") else ""))
    return "\n".join(L)


# ── cli ──────────────────────────────────────────────────────────────────────

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".aiff", ".aif", ".wma"}
MIDI_EXT = {".mid", ".midi"}


def expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        hits = glob.glob(os.path.expanduser(p))
        if not hits and os.path.exists(os.path.expanduser(p)):
            hits = [os.path.expanduser(p)]
        for h in sorted(hits):
            if os.path.isdir(h):
                for root, _, files in os.walk(h):
                    out += [os.path.join(root, f) for f in sorted(files)]
            else:
                out.append(h)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deconstruct audio, MIDI and Resonarium states into a beat-making digital twin.")
    ap.add_argument("files", nargs="*", help="audio, MIDI or directories (globs fine)")
    ap.add_argument("-o", "--out", default="twin.json", help="twin file to write")
    ap.add_argument("--add", action="store_true", help="accumulate into an existing twin")
    ap.add_argument("--resonarium", nargs="+", default=[], help="Resonarium state JSON")
    ap.add_argument("--seconds", type=float, default=120.0, help="audio to analyse per file")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files")
    ap.add_argument("--inspect", metavar="TWIN", help="print an existing twin and exit")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        print(f"\n\033[1mtwin\033[0m  {args.inspect}\n")
        print(report(json.load(open(args.inspect))))
        print()
        return 0

    paths = expand(args.files)
    res_paths = expand(args.resonarium)
    if not paths and not res_paths:
        ap.print_help()
        return 1

    twin = blank_twin()
    if args.add and os.path.exists(args.out):
        twin = json.load(open(args.out))
        if twin.get("schema") != "beatmi.twin.v1":
            print(f"! {args.out} is not a beatmi.twin.v1 file", file=sys.stderr)
            return 1

    seen = {s["file"] for s in twin["sources"]}
    todo = [p for p in paths
            if os.path.splitext(p)[1].lower() in AUDIO_EXT | MIDI_EXT
            and os.path.basename(p) not in seen]
    if args.limit:
        todo = todo[:args.limit]

    ok = err = 0
    for i, p in enumerate(todo, 1):
        name = os.path.basename(p)
        try:
            ext = os.path.splitext(p)[1].lower()
            s = analyze_midi(p) if ext in MIDI_EXT else analyze_audio(p, args.seconds)
            twin["sources"].append(s)
            ok += 1
            if not args.quiet:
                print(f"  [{i}/{len(todo)}] {name[:46]:<46} "
                      f"{s['bpm']:>5.1f} BPM  {NOTE_NAMES[s['key']]:<2} {s['scale']}")
        except Exception as e:
            err += 1
            if not args.quiet:
                print(f"  [{i}/{len(todo)}] {name[:46]:<46} \033[2mskipped: {e}\033[0m")

    for p in res_paths:
        try:
            twin["sources"].append(analyze_resonarium(p))
            ok += 1
            if not args.quiet:
                print(f"  resonarium  {os.path.basename(p)}")
        except Exception as e:
            err += 1
            print(f"  resonarium  {os.path.basename(p)} \033[2mskipped: {e}\033[0m")

    if not twin["sources"]:
        print("\nNothing could be analysed.", file=sys.stderr)
        return 1

    consolidate(twin)
    json.dump(twin, open(args.out, "w"), separators=(",", ":"))

    print(f"\n\033[1mtwin\033[0m  {args.out}   {ok} analysed, {err} skipped\n")
    print(report(twin))
    print(f"\n  Load {args.out} into SPINE with LOAD TWIN to generate from it.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
