#!/usr/bin/env python3
"""
Ground-truth tests for the twin analyzer.

Renders synthetic tracks whose tempo, key, scale and drum pattern are known by
construction, then checks that analysis recovers them. Without this the analyzer
can only be eyeballed, and eyeballing a spectrogram tells you nothing about
whether the numbers are right.

    python3 test_analyze.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

SR = A.SR
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


# ── synthesis ────────────────────────────────────────────────────────────────

def env_exp(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (SR * decay))


def synth_kick(dur=0.4):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 120 * np.exp(-t / 0.03) + 45
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 0.09)


def synth_snare(dur=0.25):
    n = int(SR * dur)
    rng = np.random.default_rng(7)
    body = np.sin(2 * np.pi * 190 * np.arange(n) / SR) * 0.5
    return (rng.standard_normal(n) * 0.9 + body) * env_exp(n, 0.05)


def synth_hat(dur=0.06):
    n = int(SR * dur)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(n)
    # crude highpass so it sits in the hat band
    return np.convolve(x, [1, -0.92], mode="same") * env_exp(n, 0.012) * 0.5


def synth_tone(midi: int, dur: float):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 440 * 2 ** ((midi - 69) / 12)
    x = sum(np.sin(2 * np.pi * f * h * t) / (h ** 1.4) for h in (1, 2, 3, 4))
    return x * env_exp(n, dur * 0.5) * 0.35


def render(path: str, bpm: float, key: int, scale: str,
           kick_steps, snare_steps, hat_steps, melody, bars=16):
    """Write a wav whose ground truth is exactly the arguments given."""
    step = 60.0 / bpm / 4.0
    total = int(SR * step * 16 * bars) + SR
    buf = np.zeros(total)

    def place(sig, at):
        i = int(at * SR)
        j = min(total, i + len(sig))
        if j > i:
            buf[i:j] += sig[: j - i]

    ivs = A.SCALES[scale]
    for b in range(bars):
        base = b * 16 * step
        for s in kick_steps:
            place(synth_kick(), base + s * step)
        for s in snare_steps:
            place(synth_snare(), base + s * step)
        for s in hat_steps:
            place(synth_hat(), base + s * step)
        for s, deg, ln in melody:
            midi = 60 + key + ivs[deg % len(ivs)] + 12 * (deg // len(ivs))
            place(synth_tone(midi, ln * step * 0.95), base + s * step)

    buf /= max(np.max(np.abs(buf)), 1e-9)
    pcm = (buf * 0.9 * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "s16le", "-ar", str(SR),
                    "-ac", "1", "-i", "-", path], input=pcm, check=True)


# ── tests ────────────────────────────────────────────────────────────────────

def main() -> int:
    tmp = tempfile.mkdtemp(prefix="twin_test_")

    # ---- case 1: dark trap, C phrygian, 140 BPM ----------------------------
    KICK = [0, 6, 10]
    SNARE = [4, 12]
    HAT = list(range(0, 16, 2))
    MEL = [(0, 0, 2), (4, 2, 2), (7, 4, 1), (10, 3, 2), (14, 0, 2)]
    p1 = os.path.join(tmp, "trap_C_phrygian_140.wav")
    render(p1, 140, 0, "phrygian", KICK, SNARE, HAT, MEL)

    print("\n\033[1mcase 1\033[0m  C phrygian, 140 BPM, kick on 0/6/10, snare on 4/12")
    r = A.analyze_audio(p1, 60)
    check(abs(r["bpm"] - 140) < 3.5, f"tempo {r['bpm']} recovers 140")
    check(r["key"] == 0, f"key {A.NOTE_NAMES[r['key']]} recovers C")
    check(r["scale"] in ("phrygian", "minor", "minpent"),
          f"scale {r['scale']} is phrygian or a subset of it")

    kg = np.array(r["rhythm"]["kick"])
    got = set(np.flatnonzero(kg > 0.45).tolist())
    check(set(KICK) <= got, f"kick grid {sorted(got)} contains the placed steps {KICK}")
    check(len(got) <= 6, f"kick grid found {len(got)} steps, not a wall of hits")

    sg = np.array(r["rhythm"]["snare"])
    sgot = set(np.flatnonzero(sg > 0.45).tolist())
    check(bool(set(SNARE) & sgot), f"snare grid {sorted(sgot)} hits the backbeat {SNARE}")

    check(1.0 <= r["notes_per_bar"] <= 9.0,
          f"{r['notes_per_bar']} notes/bar is a melody, not a wall")
    check(1.0 <= r["avg_len_steps"] <= 8.0, f"{r['avg_len_steps']} steps average length")

    # ---- case 2: halftime, F# minor, 84 BPM, sparse ------------------------
    K2, S2, H2 = [0, 11], [8], [0, 4, 8, 12]
    M2 = [(0, 0, 4), (8, 4, 2), (12, 2, 4)]
    p2 = os.path.join(tmp, "halftime_Fs_minor_84.wav")
    render(p2, 84, 6, "minor", K2, S2, H2, M2, bars=12)

    print("\n\033[1mcase 2\033[0m  F# minor, 84 BPM, sparse kick on 0/11")
    r2 = A.analyze_audio(p2, 60)
    check(abs(r2["bpm"] - 84) < 3 or abs(r2["bpm"] - 168) < 5,
          f"tempo {r2['bpm']} recovers 84 (or its double)")
    check(r2["key"] == 6, f"key {A.NOTE_NAMES[r2['key']]} recovers F#")
    k2 = set(np.flatnonzero(np.array(r2["rhythm"]["kick"]) > 0.45).tolist())
    check(len(k2) <= 5, f"sparse kick stays sparse: {sorted(k2)}")

    # ---- case 3: key detection across all 12 roots -------------------------
    print("\n\033[1mcase 3\033[0m  key detection across all 12 roots (A minor scale content)")
    hits = 0
    for root in range(12):
        chroma = np.zeros((12, 1), dtype=np.float32)
        for iv, w in zip(A.SCALES["minor"], [5.0, 1.2, 3.0, 1.6, 4.2, 1.5, 2.4]):
            chroma[(root + iv) % 12, 0] = w
        got, sc, _ = A.detect_key(chroma)
        hits += got == root
    check(hits >= 11, f"{hits}/12 roots identified from weighted minor content")

    print("\n\033[1mcase 4\033[0m  mode discrimination")
    for scale in ("phrygian", "dorian", "major", "minor"):
        chroma = np.zeros((12, 1), dtype=np.float32)
        ivs = A.SCALES[scale]
        for i, iv in enumerate(ivs):
            chroma[iv % 12, 0] = 5.0 if i == 0 else (3.6 if iv == 7 else 2.4)
        got_root, got_scale, _ = A.detect_key(chroma)
        same = set(A.SCALES[got_scale]) == set(ivs)
        check(got_root == 0 and same,
              f"{scale} content -> {A.NOTE_NAMES[got_root]} {got_scale}")

    # ---- case 5: MIDI round trip through the parser ------------------------
    print("\n\033[1mcase 5\033[0m  MIDI parser")
    mid = os.path.join(tmp, "t.mid")
    write_test_midi(mid)
    notes, tpq, bpm = A.parse_midi(mid)
    check(len(notes) == 5, f"parsed {len(notes)} notes, expected 5")
    check(tpq == 96, f"ticks per quarter {tpq}")
    check(abs(bpm - 150) < 0.5, f"tempo meta {bpm:.1f} recovers 150")
    check(sorted(n["pitch"] for n in notes) == [36, 36, 42, 60, 63],
          "pitches round-trip")
    rm = A.analyze_midi(mid)
    check(rm["rhythm"]["kick"][0] > 0, "GM kick mapped into the kick lane")

    # ---- case 6: Resonarium bridge -----------------------------------------
    print("\n\033[1mcase 6\033[0m  Resonarium bridge")
    import json
    st = os.path.join(tmp, "state.json")
    json.dump({"schema": "resonarium.state.v2",
               "sweeps": [{"f": 110, "lvl": .4, "on": True}],
               "bins": [{"carrier": 164.81, "beat": 2.333, "lvl": .4, "on": True},
                        {"carrier": 220, "beat": 4.666, "lvl": .3, "on": True}],
               "singles": [{"f": 130.81, "lvl": .3, "on": True}]}, open(st, "w"))
    rr = A.analyze_resonarium(st)
    check(abs(rr["bpm"] - 140) < 1.5,
          f"binaural beat 2.333 Hz -> {rr['bpm']} BPM (140 expected)")
    check(110.0 in rr["carriers"] and 220.0 in rr["carriers"],
          f"carriers captured: {rr['carriers']}")
    check(rr["key"] in range(12), f"state resolved to a key: {A.NOTE_NAMES[rr['key']]}")

    # ---- case 7: twin consolidation ----------------------------------------
    print("\n\033[1mcase 7\033[0m  twin consolidation")
    twin = A.blank_twin()
    twin["sources"] = [r, r2, rm, rr]
    A.consolidate(twin)
    check(abs(sum(twin["key_weights"]) - 1.0) < 1e-3, "key weights form a distribution")
    check(abs(sum(twin["scale_weights"].values()) - 1.0) < 1e-3,
          "scale weights form a distribution")
    for v in A.VOICES:
        g = twin["rhythm"][v]
        check(len(g) == 16 and all(0 <= x <= 1 for x in g), f"{v} grid is 16 values in 0..1")
    m = twin["melody"]
    if m.get("transitions"):
        rows = [sum(row) for row in m["transitions"]]
        check(all(abs(x - 1.0) < 1e-3 for x in rows),
              "every Markov row sums to 1")
    check(twin["tempo"]["median"] > 0, f"tempo median {twin['tempo']['median']}")
    check(twin["resonarium"]["carriers"] != [], "resonarium carriers carried through")

    print()
    if FAILS:
        print(f"\033[1m{len(FAILS)} FAILURES\033[0m")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("\033[1mALL GROUND-TRUTH TESTS PASSED\033[0m")
    return 0


def write_test_midi(path: str) -> None:
    import struct

    def vlq(n):
        b = [n & 0x7F]
        n >>= 7
        while n:
            b.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(b)

    def track(evs):
        out = b""
        last = 0
        for t, d in sorted(evs, key=lambda e: e[0]):
            out += vlq(t - last) + d
            last = t
        out += vlq(0) + b"\xFF\x2F\x00"
        return b"MTrk" + struct.pack(">I", len(out)) + out

    us = 60_000_000 // 150
    meta = track([(0, b"\xFF\x51\x03" + us.to_bytes(3, "big"))])
    drums = track([(0, b"\x99\x24\x64"), (24, b"\x89\x24\x40"),
                   (96, b"\x99\x24\x64"), (120, b"\x89\x24\x40"),
                   (48, b"\x99\x2A\x50"), (60, b"\x89\x2A\x40")])
    lead = track([(0, b"\x90\x3C\x60"), (96, b"\x80\x3C\x40"),
                  (96, b"\x90\x3F\x60"), (192, b"\x80\x3F\x40")])
    hdr = b"MThd" + struct.pack(">IHHH", 6, 1, 3, 96)
    open(path, "wb").write(hdr + meta + drums + lead)


if __name__ == "__main__":
    sys.exit(main())
