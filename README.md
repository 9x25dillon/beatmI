# beatmI — SPINE

A beat instrument for producers who can program drums fine but keep writing melodies that wander and rhythms that don't lock together.

Single file. No build step, no dependencies, no install. Open `index.html` in a browser and it's already playing.

**→ [Open the instrument](https://9x25dillon.github.io/beatmI/)** *(enable GitHub Pages on `main` to activate this link)*

---

## The idea

Most beat tools give you a blank grid and infinite freedom, which is exactly the problem when consistency is what you're missing. SPINE removes the choices that go wrong and keeps the ones that matter.

### 1. Scale lock

Every row of the melody grid is a note inside your selected key and scale. There is no cell to click that sounds wrong. Rows carrying the **1** and the **5** are tinted, because those are where a phrase wants to land — the tool shows you the exit before you need it.

Eight scales, weighted toward dark production: natural minor, phrygian, phrygian dominant, harmonic minor, dorian, minor pentatonic, hirajoshi, major.

### 2. Rhythm lock

Point the lock at your **kick**, your **hats**, or straight **8ths**, and melody notes can only be placed on steps where that source hits. Locked columns are marked through both grids.

Melody and drums sharing rhythmic DNA is the mechanical reason a beat sounds glued together rather than like two ideas playing at once. This is the setting that fixes "my rhythms feel inconsistent."

### 3. Motif engine

Write **one** bar you like. Then:

- **A′** is motif A pushed through a single operation — transpose ±2, octave up, invert, retrograde, displace, thin out, ornament.
- **Build B from A** commits that operation into a second motif you can then edit by hand.
- The **phrase** strip assigns each of the four bars to A, A′, B, or rest.

Default is `A A′ A B` — state it, bend it, state it, answer it. Because every variation is *derived* from A rather than newly invented, four bars stay recognisably one idea. That's what "consistent melody" actually means in practice.

**Seed a motif** generates a starting line for you: small in-scale intervals, placed on the locked rhythm, forced to resolve onto the 1 or the 5.

### 4. Mud check

A melody note in the low octave landing on the same step as the kick gets a red outline, and the readout counts them. Low content stacked on the kick transient is the most common reason a beat sounds cluttered at the bottom.

---

## Using it

| Action | How |
|---|---|
| Play / stop | `Space`, or the transport button |
| Toggle a drum step | Click the cell |
| Hat rolls | Click a HAT cell repeatedly — off → 1 → 2 (16ths) → 3 (triplets). Shift-click cycles backwards |
| Mute a voice | Click its name |
| Place a melody note | Click a cell. One note per column — clicking a new row in an occupied column moves the note |
| Remove a note | Click it again |
| Change what a bar plays | Click it in the phrase strip to cycle A / A′ / B / rest |

The **808 is pitched automatically** from the lowest melody note sounding at that step, falling back to the root. The resolved note letter shows inside the cell. Bass following the harmony is not a thing worth hand-editing.

## Output

- **Download MIDI** — a type-1 `.mid` at 96 PPQ, four bars, three tracks: drums on channel 10 (GM mapped), melody on channel 1, 808 on channel 2. Drag it into any DAW.
- **Copy step grid** — monospace `x-o-.` notation of the full pattern plus both motifs.
- **Copy description** — a plain-language line naming key, scale, tempo, swing, lock source, phrase form, and where the motif resolves.

Swing is applied to playback in the browser; exported MIDI is written straight so your DAW's own groove/quantize stays authoritative.

## Structure

```
index.html    the whole instrument — markup, CSS, Web Audio synthesis, MIDI writer
```

Drum and melody voices are synthesised live with the Web Audio API — no samples, so nothing to load and nothing to license. Kick is a pitch-enveloped sine with a noise transient, snare is filtered noise over a triangle body, clap is a four-burst noise stack, hats are highpassed noise, the 808 is a glided sine through a `tanh` shaper, and the lead is two detuned saws through a resonant lowpass sweep.

## Roadmap

- Chord bed derived from the motif's implied harmony
- Per-note velocity and slide (drill-style 808 glides)
- Pattern save/recall to `localStorage`
- Longer forms than four bars

## License

MIT
