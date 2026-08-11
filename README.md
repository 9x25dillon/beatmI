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

---

## The digital twin

`twin/analyze.py` deconstructs your own catalogue into a probability model of how you make beats. SPINE reconstructs from it. Nothing uploads — files are read locally and the twin is a plain JSON file you own.

**No pip install.** Audio is decoded through `ffmpeg`; the DSP is `numpy` and `scipy` only.

```bash
python3 twin/analyze.py ~/Music/*.mp3 -o twin.json     # build
python3 twin/analyze.py --add ~/Desktop/*.wav          # accumulate more
python3 twin/analyze.py --resonarium state.json        # fold in a Resonarium state
python3 twin/analyze.py --inspect twin.json            # look at it
```

Then hit **LOAD TWIN** in SPINE and it generates in your idiom.

### What gets extracted

| From audio (mp3/wav/flac/m4a/ogg) | From MIDI |
|---|---|
| Tempo by autocorrelation with a parabolic peak fit | Exact tempo from the meta event |
| Downbeat located by rotating the bar against metric strength | Exact onsets |
| Per-voice hit probabilities — kick / snare / hat / 808, band-split after harmonic-percussive separation | Exact GM drum mapping |
| Key and mode by correlating chroma against tonal templates | Exact pitches |
| Scale-degree histogram and a Markov chain over degree motion | Exact melodic intervals |
| Swing, syncopation, note density, average note length | Exact note lengths |

Audio analysis is **statistical, not a transcription.** Polyphonic melody extraction from a finished master is not reliable, and this does not pretend otherwise. What it recovers is how your lines *move* — which degrees you favour, which intervals you reach for, how much space you leave. That is what generation needs. Feed it MIDI or stems and the melodic side gets much sharper.

### The Resonarium bridge

A [Resonarium](https://github.com/9x25dillon) state is already musical data, so it folds straight in:

- **Carrier, sweep and tone frequencies** are pitches — they collapse onto pitch classes and vote on key.
- **Binaural beat rates are tempi.** A 2.333 Hz beat is 140 BPM once folded into a musical range.
- **`natalSeed`** seeds the generator, so the same chart reproduces the same beat.

Both `resonarium.state.v2` and the hologram/cymatic schema are read.

### Reconstruction

Generation samples from the twin rather than replaying it — your habits set the odds, chance picks the take. The **Faithfulness** slider sharpens or flattens every distribution at once: low leans on chance, high copies your habits, and the middle is where it writes something you would have written but didn't.

---

## Pattern barcode

A Unicode Braille cell is two columns of four dots — a tiny bitmap. Four drum voices across sixteen steps therefore pack into **eight characters**:

```
⡅⠄⠆⡅⠄⡅⠆⡤
```

That is a whole bar you can paste into a chat, a filename, or a commit message. SPINE has a field for it; `twin/braille.py` is the same codec in Python, verified byte-identical across both.

```bash
python3 twin/braille.py --demo
python3 twin/braille.py --pattern '⡅⠄⠆⡅⠄⡅⠆⡤'
python3 twin/braille.py --decode art.txt              # Braille art -> ASCII
python3 twin/braille.py --decode art.txt --style density
python3 twin/braille.py --encode art.txt              # and back again
```

**On dot layouts.** Unicode numbers dots 1–3 down the left column, 4–6 down the right, with 7–8 as the bottom row — so `0x08` is *(column 1, row 0)*. The other convention in circulation treats bits 0–3 as the whole left column, which puts `0x08` at *(column 0, row 3)*. Decoding Unicode art with the linear map transposes half the dots. Both are supported; `--layout unicode` is the default and `--layout linear` is there for generators that use the other one.

---

## Structure

```
index.html          the instrument — markup, CSS, Web Audio synthesis, MIDI writer,
                    twin loading, generation, barcode codec
twin/analyze.py     deconstruction: audio, MIDI and Resonarium -> twin.json
twin/braille.py     Braille bitmaps and the pattern barcode codec
twin/test_analyze.py  ground-truth tests
```

## Tests

`twin/test_analyze.py` renders synthetic tracks whose tempo, key, scale and drum pattern are known by construction, then checks that analysis recovers them — because a spectrogram that looks plausible tells you nothing about whether the numbers are right.

```bash
python3 twin/test_analyze.py
```

Covers tempo and downbeat recovery, key detection across all 12 roots, mode discrimination, the MIDI parser, the Resonarium bridge, and twin consolidation.

Drum and melody voices are synthesised live with the Web Audio API — no samples, so nothing to load and nothing to license. Kick is a pitch-enveloped sine with a noise transient, snare is filtered noise over a triangle body, clap is a four-burst noise stack, hats are highpassed noise, the 808 is a glided sine through a `tanh` shaper, and the lead is two detuned saws through a resonant lowpass sweep.

## Roadmap

- Chord bed derived from the motif's implied harmony
- Per-note velocity and slide (drill-style 808 glides)
- Pattern save/recall to `localStorage`
- Longer forms than four bars
- Stem-aware analysis, so isolated drums and leads bypass the statistical melody path
- Twin drift over time — weight recent work above old work
- VST3/AU wrapper (JUCE), so the writing tools live inside the DAW

## License

MIT
