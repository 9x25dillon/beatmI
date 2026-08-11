#!/usr/bin/env python3
"""
beatmI / braille — Braille as a 2x4 pixel grid, and as a pattern barcode.

A Unicode Braille cell carries eight dots in two columns of four. That makes it
a tiny bitmap, which is why Braille art works at all. It also makes it a compact
container for a drum pattern: four voices down the rows, two steps across the
columns, so a whole 16-step bar of kick / snare / hat / 808 fits in eight
characters you can paste into a chat, a filename or a commit message.

    python3 braille.py --demo
    python3 braille.py --decode art.txt
    python3 braille.py --decode art.txt --style density
    python3 braille.py --pattern '⡇⠄⢠⡄⣿⠁⡆⠿'

No dependencies. Mirrored by the same encoder inside SPINE, so a code copied out
of the browser decodes here and back again.
"""

from __future__ import annotations

import argparse
import sys

# ── dot layouts ──────────────────────────────────────────────────────────────
#
# Unicode Braille Patterns run U+2800..U+28FF. The low six bits are the classic
# cell, numbered down the left column then down the right; bits 7 and 8 are the
# extra bottom row added for 8-dot Braille. Written as (column, row):
#
#     dot1 0x01 (0,0)   dot4 0x08 (1,0)
#     dot2 0x02 (0,1)   dot5 0x10 (1,1)
#     dot3 0x04 (0,2)   dot6 0x20 (1,2)
#     dot7 0x40 (0,3)   dot8 0x80 (1,3)

UNICODE_MAP = {
    0x01: (0, 0), 0x02: (0, 1), 0x04: (0, 2), 0x40: (0, 3),
    0x08: (1, 0), 0x10: (1, 1), 0x20: (1, 2), 0x80: (1, 3),
}

# The other convention in circulation: bits 0-3 are the left column top to
# bottom, bits 4-7 the right. Some generators emit this. It is not Unicode's
# layout, so decoding Unicode Braille art with it transposes dots 4-7.
LINEAR_MAP = {
    0x01: (0, 0), 0x02: (0, 1), 0x04: (0, 2), 0x08: (0, 3),
    0x10: (1, 0), 0x20: (1, 1), 0x40: (1, 2), 0x80: (1, 3),
}

LAYOUTS = {"unicode": UNICODE_MAP, "linear": LINEAR_MAP}

BRAILLE_LO, BRAILLE_HI = 0x2800, 0x28FF
RAMP = " .:-=+*#%@"


def _bit_for(layout: dict[int, tuple[int, int]], col: int, row: int) -> int:
    for mask, (dx, dy) in layout.items():
        if (dx, dy) == (col, row):
            return mask
    return 0


# ── decoding ─────────────────────────────────────────────────────────────────

def to_bitmap(text: str, layout: str = "unicode") -> list[list[int]]:
    """Braille text -> a 0/1 bitmap at full 2x4-per-character resolution."""
    dots = LAYOUTS[layout]
    lines = text.splitlines()
    width = max((len(l) for l in lines), default=0) * 2
    out: list[list[int]] = []
    for line in lines:
        rows = [[0] * width for _ in range(4)]
        for x, ch in enumerate(line):
            code = ord(ch)
            if BRAILLE_LO <= code <= BRAILLE_HI:
                bits = code - BRAILLE_LO
                for mask, (dx, dy) in dots.items():
                    if bits & mask:
                        rows[dy][x * 2 + dx] = 1
        out.extend(rows)
    return out


def braille_to_ascii(text: str, on: str = "#", off: str = " ",
                     layout: str = "unicode") -> str:
    """Braille text -> ASCII at full resolution, one output char per dot.

    Non-Braille characters are preserved in place so labelled art survives.
    """
    dots = LAYOUTS[layout]
    output: list[str] = []
    for line in text.splitlines():
        rows = [[off] * (len(line) * 2) for _ in range(4)]
        for x, ch in enumerate(line):
            code = ord(ch)
            if BRAILLE_LO <= code <= BRAILLE_HI:
                bits = code - BRAILLE_LO
                for mask, (dx, dy) in dots.items():
                    if bits & mask:
                        rows[dy][x * 2 + dx] = on
            elif ch != " ":
                rows[0][x * 2] = ch
        output.extend("".join(r).rstrip() for r in rows)
    return "\n".join(output)


def braille_to_density(text: str, block: int = 2, ramp: str = RAMP,
                       layout: str = "unicode") -> str:
    """Braille text -> a shaded ASCII ramp, downsampled by `block`.

    Counting dots in a block is what produces a real grey ramp. Reading a single
    dot column gives at most two levels, which is why a naive density pass comes
    out as flat blotches rather than shading.
    """
    bmp = to_bitmap(text, layout)
    if not bmp:
        return ""
    h, w = len(bmp), len(bmp[0])
    cells = block * block
    lines: list[str] = []
    for y in range(0, h, block):
        row = []
        for x in range(0, w, block):
            count = sum(bmp[yy][xx]
                        for yy in range(y, min(y + block, h))
                        for xx in range(x, min(x + block, w)))
            row.append(ramp[count * (len(ramp) - 1) // cells])
        lines.append("".join(row).rstrip())
    return "\n".join(lines)


def ascii_to_braille(text: str, on: str = "#", layout: str = "unicode") -> str:
    """ASCII bitmap -> Braille, the inverse of braille_to_ascii."""
    dots = LAYOUTS[layout]
    lines = text.splitlines()
    if not lines:
        return ""
    width = max(len(l) for l in lines)
    out: list[str] = []
    for y0 in range(0, len(lines), 4):
        row = []
        for x0 in range(0, width, 2):
            bits = 0
            for dy in range(4):
                for dx in range(2):
                    y, x = y0 + dy, x0 + dx
                    if y < len(lines) and x < len(lines[y]) and lines[y][x] == on:
                        bits |= _bit_for(dots, dx, dy)
            row.append(chr(BRAILLE_LO + bits))
        out.append("".join(row))
    return "\n".join(out)


# ── pattern barcode ──────────────────────────────────────────────────────────

VOICES = ["kick", "snare", "hat", "sub"]
STEPS = 16


def pattern_to_braille(grid: dict[str, list], threshold: float = 0.5) -> str:
    """A 4-voice x 16-step drum pattern as eight Braille characters.

    Rows are kick / snare / hat / 808 top to bottom; each cell spans two steps.
    Values may be booleans, 0/1 ints or the 0..1 probabilities a twin carries.
    """
    dots = UNICODE_MAP
    out = []
    for cell in range(STEPS // 2):
        bits = 0
        for row, v in enumerate(VOICES):
            lane = grid.get(v) or []
            for dx in range(2):
                s = cell * 2 + dx
                val = lane[s] if s < len(lane) else 0
                if (val >= threshold) if isinstance(val, float) else bool(val):
                    bits |= _bit_for(dots, dx, row)
        out.append(chr(BRAILLE_LO + bits))
    return "".join(out)


def braille_to_pattern(code: str) -> dict[str, list[int]]:
    """Eight Braille characters back into a 4-voice x 16-step grid."""
    dots = UNICODE_MAP
    grid = {v: [0] * STEPS for v in VOICES}
    cells = [c for c in code if BRAILLE_LO <= ord(c) <= BRAILLE_HI][: STEPS // 2]
    for cell, ch in enumerate(cells):
        bits = ord(ch) - BRAILLE_LO
        for mask, (dx, row) in dots.items():
            if bits & mask and row < len(VOICES):
                s = cell * 2 + dx
                if s < STEPS:
                    grid[VOICES[row]][s] = 1
    return grid


def show_pattern(grid: dict[str, list], threshold: float = 0.5) -> str:
    """Human-readable grid, for checking a barcode decoded to what you meant."""
    lines = ["        1 . . . 2 . . . 3 . . . 4 . . ."]
    for v in VOICES:
        lane = grid.get(v) or []
        cells = []
        for s in range(STEPS):
            val = lane[s] if s < len(lane) else 0
            hit = (val >= threshold) if isinstance(val, float) else bool(val)
            cells.append("x" if hit else ".")
        lines.append(f"  {v:<6}" + " ".join(cells))
    return "\n".join(lines)


# ── sparkline ────────────────────────────────────────────────────────────────

def sparkline(values, height: int = 4) -> str:
    """A 0..1 series as a Braille bar chart, two samples per character."""
    dots = UNICODE_MAP
    vals = list(values)
    out = []
    for i in range(0, len(vals), 2):
        bits = 0
        for dx in range(2):
            if i + dx >= len(vals):
                break
            lvl = max(0, min(height, round(float(vals[i + dx]) * height)))
            for row in range(height - lvl, height):
                bits |= _bit_for(dots, dx, row)
        out.append(chr(BRAILLE_LO + bits))
    return "".join(out)


# ── cli ──────────────────────────────────────────────────────────────────────

DEMO = {
    "kick":  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "hat":   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    "sub":   [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Braille bitmaps and pattern barcodes.")
    ap.add_argument("--decode", metavar="FILE", help="Braille art to convert (- for stdin)")
    ap.add_argument("--encode", metavar="FILE", help="ASCII art to convert into Braille")
    ap.add_argument("--style", choices=["ascii", "density"], default="ascii")
    ap.add_argument("--layout", choices=list(LAYOUTS), default="unicode",
                    help="dot layout; 'linear' for generators that number bits 0-3 down the left column")
    ap.add_argument("--block", type=int, default=2, help="density downsample factor")
    ap.add_argument("--on", default="#")
    ap.add_argument("--pattern", metavar="CODE", help="decode a pattern barcode")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        code = pattern_to_braille(DEMO)
        print(f"\n  pattern barcode   {code}   ({len(code)} chars, 4 voices x 16 steps)\n")
        print(show_pattern(DEMO))
        back = braille_to_pattern(code)
        same = all(back[v] == DEMO[v] for v in VOICES)
        print(f"\n  round trip        {'exact' if same else 'MISMATCH'}")
        print(f"  sparkline         {sparkline([i / 15 for i in range(16)])}  (a 0..1 ramp)\n")
        print("  the same barcode as a bitmap:")
        print(braille_to_ascii(code, on="#"))
        print()
        return 0

    if args.pattern:
        grid = braille_to_pattern(args.pattern)
        print()
        print(show_pattern(grid))
        print(f"\n  re-encoded  {pattern_to_braille(grid)}\n")
        return 0

    if args.encode:
        src = sys.stdin.read() if args.encode == "-" else open(args.encode, encoding="utf-8").read()
        print(ascii_to_braille(src, on=args.on, layout=args.layout))
        return 0

    if args.decode:
        src = sys.stdin.read() if args.decode == "-" else open(args.decode, encoding="utf-8").read()
        if args.style == "density":
            print(braille_to_density(src, block=args.block, layout=args.layout))
        else:
            print(braille_to_ascii(src, on=args.on, layout=args.layout))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
