<div align="center">

# Drift-Sense

**Navigation-Error Recovery for Wafer Inspection**

SEMICON India Hackathon 2026 · Problem Statement 2 · Applied Materials

**98.0 %** accuracy at 5 px on 100 reference-generated pairs
against **75.0 %** for plain template matching

*No model weights. No training. numpy and opencv only.*

</div>

---

## The problem

A wafer inspection tool must return to the same site on a die thousands of
times a day. The stage drifts between visits, so it lands a few pixels off
— and it cannot tell, because every die carries the same repeating circuit
layout and the wrong site looks like the right one.

Given a **reference** image (1000 × 1000 at 1 nm/px, a 1 µm field) and a
**search** image (1000 × 1000 at 10 nm/px, a 10 µm field containing the
reference pattern shrunk tenfold), find the centre of the matching region.

The tolerance is 5 px. In DRAM the lattice repeats every 6.4 px in search
coordinates, so **the tolerance is narrower than one lattice cell**. Land
on the neighbouring crossing and it counts as a miss.

---

## Install

```bash
git clone <repository-url>
cd drift-sense
pip install -r requirements.txt
```

Python 3.9 or later. The two scored scripts import only `numpy` and
`opencv-python`.

---

## Generate a dataset

```bash
python generate_dataset.py --style DRAM   --n 30 --out ./data
python generate_dataset.py --style FinFET --n 30 --out ./data_ff --noise severe
```

Writes `reference/`, `search/` and `manifest.csv`. The manifest records the
true centre and every generation parameter, so any pair regenerates exactly
from its row.

| Flag | Meaning |
|:--|:--|
| `--style` | `DRAM` or `FinFET`. Aliases: `--architecture`, `--architectures` |
| `--n` | number of pairs. Aliases: `--num-samples`, `--count` |
| `--out` | output directory. Aliases: `--output-dir`, `--output` |
| `--noise` | `clean`, `low`, `medium`, `high`, `severe`, `extreme` |
| `--seed` | base seed; every sample is a pure function of `(seed, index)` |

Individual imaging parameters (`--dose-search`, `--shear-amplitude-px`,
`--speckle-sigma`, and the rest) can be overridden directly. Run
`--help` for the full list.

---

## Localise

```bash
python localize.py --reference ref.png --search search.png
```

```
746.60,318.80
```

One line on stdout: the centre of the matched region, in search-image
pixels. Diagnostics go to stderr, so the output stays clean for a harness.

| Flag | Meaning |
|:--|:--|
| `--confidence` | append the confidence as a third value |
| `--verbose` | score, confidence, offset and timing to stderr |
| `--sigma` | resolution-matching low-pass; `0` disables |
| `--step` | offset sampling stride; default 2 |
| `--tiebreak` | `search` (default) or `none` — see below |

Runs from any working directory. Handles missing files with exit code 2,
colour input, and references at sizes other than 1000 × 1000.

---

## Method

The reference must be reduced tenfold before it can be compared with the
search image. That is where ordinary template matching loses the problem.

The reference was cropped from the die at an **integer nanometre**
position `x0`, while the search image samples the die in blocks of ten.
Template pixel *m* averages `[x0 + 10m, x0 + 10m + 10)`; search pixel *j*
covers `[10j, 10j + 10)`. Those grids coincide only when `x0` is a multiple
of ten — **one time in ten**. Otherwise the template is a fractionally
shifted rendering of the same structure.

That misalignment falls entirely on the true site. It is the one place
where an **exact** match exists; every wrong site in a periodic layout is
approximate regardless. A misaligned template throws away the true site's
only advantage and costs its competitors nothing.

So instead of building one template, `localize.py` builds a **family** at
sub-pixel sampling offsets and takes the maximum over the family. Somewhere
in that family is an offset close to the true one, where the true site
matches almost exactly. Wrong sites gain nothing comparable.

Measured on 100 pairs from the reference generator:

| | median margin | true site wins |
|:--|--:|--:|
| Plain ZNCC | −0.0393 | 40 / 100 |
| Max over resamplings | **+0.0155** | **74 / 100** |

The margin between the true site and its best competitor **changes sign**.

Two details matter in the implementation.

**Exact resampling.** Templates are built by area-averaging at a fractional
offset, computed from an integral image. Shifting the image and resampling
instead would apply an interpolation filter across the whole template,
giving a blurred approximation of the aligned template rather than the
aligned template. An earlier experiment that did exactly that reported a
false null.

**Resolution matching.** Both images are low-passed by a small Gaussian
before matching. This is not denoising — removing noise from the search
image entirely was measured to fix none of the failures. The reference is
exposed at high dose and then averaged tenfold, cutting its noise by
another factor of ten, so the template is far sharper than anything the
search image can carry. A mild low-pass brings the two into agreement.

Nothing is fitted, trained or estimated. That is why the result transfers
unchanged between independently written generators.

---

## Results

Measured with `experiments/metrics.py` on 100 pairs from the reference
generator, timed from the shipped `localize.py`.

| | Value |
|:--|--:|
| Accuracy at 5 px | **98.0 %** |
| Plain template matching | 75.0 % |
| Accuracy at 1 px | 83.0 % |
| Median error | 0.65 px |
| p95 error | 1.29 px |
| Average precision | 0.9799 |
| Computation time | ~1.5 s / pair |
| Model size | none |

By reference condition — whether a peripheral strip pins each axis, which
is detectable from the reference alone before matching runs:

| Condition | Share | Plain | This method |
|:--|--:|--:|--:|
| Both axes pinned | 27 % | 92.6 % | **96.3 %** |
| One axis pinned | 61 % | 80.3 % | **100.0 %** |
| Neither pinned | 12 % | 8.3 % | **91.7 %** |

Across noise levels, DRAM, n = 40 each:

| Level | low | medium | high | severe | extreme |
|:--|--:|--:|--:|--:|--:|
| Accuracy @ 5 px | 100 % | 96.7 % | 68.3 % | 51.7 % | 17.5 % |
| AP | 1.000 | 0.874 | 0.576 | 0.308 | 0.149 |

Degradation and its causes are analysed in
[`docs/FAILURES.md`](docs/FAILURES.md).

---

## Reproduce

```bash
# every metric, from the shipped implementation
python experiments/metrics.py --data ./data --n 100 --baseline

# success and honest-failure figures for the presentation
python experiments/make_figures.py --data ./data --out ./figures
```

---

## A discrepancy worth stating

The two published descriptions of the tie-break rule disagree.

* The Applied Materials deck says to return the tile closest to the
  **reference** image centre.
* The i4C problem statement says closest to the **search** image centre.

This implementation uses the **search-image centre**, because the reference
centre is not a location within the search image and so cannot order
candidates there. `--tiebreak none` disables the rule.

The tolerance window also matters more than it appears. Applying the rule
with a wide window was measured to cost 15 points, because the rule encodes
a physical prior the test data does not contain: a real tool lands *near*
its target, but the reference generator places crops uniformly at random.
The shipped threshold fires only when two peaks are genuinely
indistinguishable. Measurements in
[`docs/FAILURES.md` §9.2](docs/FAILURES.md).

---

## Repository

```
generate_dataset.py         dataset generator          ← scored artifact
localize.py                 localisation inference     ← scored artifact
requirements.txt            pinned dependencies

driftsense/
  generator/                patterns, zones, SEM imaging chain
  localize.py               shared helpers used by the experiments

docs/
  CITATIONS.md              18 references, one per generator parameter
  FAILURES.md               failure analysis, oracles, 16 rejections

experiments/
  metrics.py                accuracy, AP, timing, conditions
  make_figures.py           success and failure visuals
  ...                       the measurement record behind FAILURES.md
```

**No deep-learning model is used**, so repository items 4 and 5 of the
submission requirements — model weights and a training notebook — are not
applicable. The classical method was chosen because the problem geometry is
fully constrained, and because it leaves nothing that can fail to load on
the evaluator's machine.

`driftsense/verifier.py` and `driftsense/residual.py` implement approaches
that were **measured and rejected**. They are retained so the results in
`docs/FAILURES.md` can be reproduced, and are imported by nothing on either
scored path.

---

## Documentation

| Document | Contents |
|:--|:--|
| [`docs/CITATIONS.md`](docs/CITATIONS.md) | Every structural, noise and augmentation parameter, with its physical justification and 18 verified public sources |
| [`docs/FAILURES.md`](docs/FAILURES.md) | Root-cause analysis, worked failure example, the identifiability predictor, five oracle results, sixteen rejected approaches and three retracted findings |