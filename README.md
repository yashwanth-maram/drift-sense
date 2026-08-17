<div align="center">

# Drift-Sense

### Find the one place that is exactly right.

**Team Lattice** · SEMICON India Hackathon 2026
Problem Statement 2 — Applied Materials

<br>

[![accuracy](https://img.shields.io/badge/accuracy%20%40%205px-98.0%25-2f81f7?style=for-the-badge&labelColor=0d1117)](#result)
[![gain](https://img.shields.io/badge/vs%20template%20matching-%2B23.0%20pp-3fb950?style=for-the-badge&labelColor=0d1117)](#result)
[![ap](https://img.shields.io/badge/average%20precision-0.9799-2f81f7?style=for-the-badge&labelColor=0d1117)](#knowing-when-it-is-wrong)

![python](https://img.shields.io/badge/python-3.9+-8b949e?labelColor=0d1117)
![deps](https://img.shields.io/badge/dependencies-numpy%20·%20opencv-8b949e?labelColor=0d1117)
![weights](https://img.shields.io/badge/model%20weights-none-8b949e?labelColor=0d1117)
![time](https://img.shields.io/badge/inference-1.1%20s%20%2F%20pair-8b949e?labelColor=0d1117)

<br>

*In a repeating layout, many places are approximately right.*
*Only one is exactly right.*

</div>

```bash
pip install -r requirements.txt
python generate_dataset.py --style DRAM --n 5 --out ./data
python localize.py --reference ./data/reference/00000.png --search ./data/search/00000.png
```

> [!TIP]
> That prints one line — `746.60,318.80` — the centre of the matched region
> in search-image pixels. Nothing else to configure, no weights to download.
> [Full setup and all flags ↓](#quick-start)

---

## Contents

| | |
|:--|:--|
| [The problem](#the-problem) | why a repeating layout defeats template matching |
| [Result](#result) | 98.0 % against a 75.0 % baseline |
| [Why it works](#why-it-works) | the mechanism, and the measurement that shows it |
| [Where the difficulty lives](#where-the-difficulty-lives) | what actually fails, and why |
| [Knowing when it is wrong](#knowing-when-it-is-wrong) | the confidence score |
| [Worked failure](#worked-failure) | the honest example |
| [Quick start](#quick-start) | clone, generate, localise |
| [The generator](#the-generator) | five layers, 18 cited parameters |
| [Documentation](#documentation) | citations and failure analysis |

---

## The problem

A wafer inspection tool must return to the same site on a die thousands of
times a day. The stage drifts between visits, so it lands a few pixels off
— and it cannot tell, because every die carries the same repeating circuit
layout and the wrong site looks like the right one.

|  | Reference | Search |
|:--|:--|:--|
| Size | 1000 × 1000 | 1000 × 1000 |
| Pixel | 1 nm | 10 nm |
| Field of view | 1 µm | 10 µm |
| Exposure | high dose | low dose, noisier |

Find the centre of the region in the search image where the reference
pattern appears, shrunk tenfold. **Tolerance: 5 px.**

> In DRAM the lattice repeats every **6.4 px** in search coordinates.
> The tolerance is narrower than a single lattice cell — land on the
> neighbouring crossing and it counts as a miss.

---

## Result

<div align="center">
<img src="figures/01_accuracy.png" width="560">
</div>

Measured on **100 pairs from Applied Materials' own reference generator**,
timed from the shipped `localize.py`.

| Metric | Drift-Sense | Plain matching |
|:--|--:|--:|
| **Accuracy at 5 px** | **98.0 %** | 75.0 % |
| Accuracy at 2 px | 98.0 % | — |
| Accuracy at 1 px | 83.0 % | — |
| Median error | **0.65 px** | 0.90 px |
| p95 error | 1.29 px | — |
| Average precision | **0.9799** | — |
| Time per pair | 1.1 s | 47 ms |
| Model size | **none** | none |

---

## Why it works

The reference must be reduced tenfold before it can be compared with the
search image. That is where ordinary template matching loses the problem.

The reference was cropped from the die at an **integer nanometre** position
`x0`, while the search image samples the die in blocks of ten:

```
template pixel m   averages   [ x0 + 10m , x0 + 10m + 10 )
search   pixel j   covers     [     10j  ,     10j  + 10 )
```

Those grids coincide only when `x0 ≡ 0 (mod 10)` — **one time in ten**.
Otherwise the template is a fractionally shifted rendering of the same
structure.

That misalignment falls **entirely on the true site**. It is the one place
where an exact match exists; every wrong site in a periodic layout is
approximate regardless. A misaligned template throws away the true site's
only advantage and costs its competitors nothing.

So instead of one template, `localize.py` builds a **family** of them at
sub-pixel sampling offsets and takes the maximum over the family.

```mermaid
flowchart LR
    REF["<b>Reference</b><br/>1000 × 1000<br/>1 nm / px"]
    FAM["<b>Template family</b><br/>25 sub-pixel offsets<br/>exact area-average"]
    LP1["low-pass<br/>σ = 1.0"]
    SRC["<b>Search</b><br/>1000 × 1000<br/>10 nm / px"]
    LP2["low-pass<br/>σ = 1.0"]
    ZN["<b>ZNCC</b><br/>one surface per offset"]
    MAX["<b>max over the family</b>"]
    TB["centre tie-break"]
    SP["sub-pixel refine"]
    OUT(["<b>x , y</b><br/>+ confidence"])

    REF --> FAM --> LP1 --> ZN
    SRC --> LP2 --> ZN
    ZN --> MAX --> TB --> SP --> OUT

    style REF fill:#161b22,stroke:#2f81f7,color:#e6edf3
    style SRC fill:#161b22,stroke:#2f81f7,color:#e6edf3
    style MAX fill:#0d2818,stroke:#3fb950,color:#e6edf3
    style OUT fill:#0d2818,stroke:#3fb950,color:#e6edf3
```

<div align="center">
<img src="figures/04_margin.png" width="820">
</div>

| | median margin | true site wins |
|:--|--:|--:|
| Plain ZNCC | +0.0213 | 76 / 100 |
| **Max over resamplings** | **+0.0265** | **98 / 100** |

**22 of the 24 cases where the true site was losing become wins** — which is
exactly the accuracy gain. That is the whole method.

<details>
<summary><b>Two implementation details that decide whether it works</b></summary>

<br>

**Exact resampling.** Templates are built by area-averaging at a fractional
offset, computed from an integral image. Shifting the image and resampling
instead would apply an interpolation filter across the whole template,
giving a *blurred approximation* of the aligned template rather than the
aligned template. An earlier experiment that did exactly that reported a
false null.

**Resolution matching.** Both images are low-passed by a small Gaussian
before matching. This is **not denoising** — removing noise from the search
image entirely was measured to fix none of the failures. The reference is
exposed at high dose and then averaged tenfold, cutting its noise by
another factor of ten, so the template is far sharper than anything the
search image can carry. A mild low-pass brings the two into agreement.

Accuracy at 5 px across noise levels, n = 60 per cell:

| σ | AM data | low | medium | high | severe |
|--:|--:|--:|--:|--:|--:|
| 0.0 | 96.0 | 100.0 | 91.7 | 65.0 | 41.7 |
| **1.0** | **98.0** | **100.0** | **96.7** | **68.3** | **51.7** |
| 1.5 | 99.0 | 95.0 | 83.3 | 68.3 | 51.7 |

σ = 1.0 never loses anywhere. σ = 1.5 scores higher on one dataset but
costs 5 points at *low* and 8 at *medium*, so it is fitted to a single
noise level. **1.0 ships.**

Nothing is fitted, trained or estimated — which is why the result transfers
unchanged between independently written generators.

</details>

---

## Where the difficulty lives

<div align="center">
<img src="figures/02_conditions.png" width="700">
</div>

A **peripheral strip** — the flat band of sense amplifiers and routing
between sub-array mats — is the only aperiodic landmark in the field. Every
measured failure was a reference containing none.

That condition is detectable **from the reference alone, before matching
runs**, which makes it a failure predictor as well as an explanation.

<div align="center">
<img src="figures/03_error_dist.png" width="700">
</div>

The error distribution is bimodal and sharply so. **Nothing lands between
2 px and 50 px.** The method either finds the correct site to sub-pixel
precision or lands in a different mat entirely — there is no intermediate
regime of slightly-wrong answers to recover.

---

## Knowing when it is wrong

<div align="center">
<img src="figures/05_confidence.png" width="880">
</div>

Every failure sits in the bottom fifth by confidence. The **top 80 % by
confidence are 100 % accurate**.

> For an inspection tool this matters more than the accuracy figure. A tool
> that knows it missed can re-acquire. A confidently wrong tool measures
> the wrong site all day.

<div align="center">
<img src="figures/06_noise.png" width="700">
</div>

Average precision tracks accuracy all the way down, from 1.000 to 0.149.
The score does not remain falsely high once the method stops working.

---

## Worked failure

Applied Materials ask for **one honest example of where the method fails,
and why.**

<div align="center">
<img src="figures/failure_00049.png" width="880">
</div>

| | |
|:--|:--|
| True centre | (88.4, 779.5) |
| Predicted | (154.1, 637.1) |
| Error | **156.84 px** |
| Confidence | **0.000** |
| Condition | no strip on either axis |

The two candidate regions, bottom right, are visibly the same pattern. They
differ only in lattice phase and in the noise realisation of that capture.
No comparison of appearance can separate them, because there is nothing
aperiodic in either to compare.

It reported **zero confidence** against a median of 0.105 on correct
predictions. It did not merely fail — it said so.

Full analysis, including sixteen rejected approaches and five oracle
results, in **[`docs/FAILURES.md`](docs/FAILURES.md)**.

---

## Quick start

```bash
git clone https://github.com/yashwanth-maram/drift-sense.git
cd drift-sense
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` lists what the project needs — `numpy` and
> `opencv-python` for the two scored scripts, plus `matplotlib` and
> `scikit-learn` for the experiments.
>
> A complete `pip freeze` of the development environment is in
> [`requirements-freeze.txt`](requirements-freeze.txt), as required by
> submission item 6. That environment is shared with unrelated projects,
> so installing from it is neither necessary nor advisable.
>
> Verified end to end on **Windows / Python 3.12.10 / numpy 2.2.5 /
> cv2 5.0.0** and **Linux / Python 3.12.3 / numpy 2.4.4 / cv2 4.13.0**,
> with identical results.

**Generate data**

```bash
python generate_dataset.py --style DRAM   --n 30 --out ./data
python generate_dataset.py --style FinFET --n 30 --out ./data_ff --noise severe
```

**Localise a single pair**

```bash
python localize.py --reference ./data/reference/00000.png --search ./data/search/00000.png
```

```
746.60,318.80
```

One line on stdout: the centre in search-image pixels. Diagnostics go to
stderr, so the output stays clean for an automated harness.

**Or a whole directory at once**

```bash
python localize.py --batch ./data
python localize.py --batch ./data --out results.csv
```

Finds `reference/` and `search/` anywhere under the given path, pairs them
by filename, and reports accuracy against `manifest.csv` when one is
present. Progress goes to stderr, results to stdout.

<details>
<summary><b>All flags</b></summary>

<br>

**`generate_dataset.py`**

| Flag | Meaning |
|:--|:--|
| `--style` | `DRAM` or `FinFET`. Aliases: `--architecture`, `--architectures` |
| `--n` | number of pairs. Aliases: `--num-samples`, `--count` |
| `--out` | output directory. Aliases: `--output-dir`, `--output` |
| `--noise` | `clean` · `low` · `medium` · `high` · `severe` · `extreme` |
| `--seed` | base seed; every sample is a pure function of `(seed, index)` |

Individual imaging parameters — `--dose-search`, `--shear-amplitude-px`,
`--speckle-sigma` and the rest — can be overridden directly. See `--help`.

**`localize.py`**

| Flag | Meaning |
|:--|:--|
| `--confidence` | append the confidence as a third value |
| `--verbose` | score, confidence, offset and timing to stderr |
| `--sigma` | resolution-matching low-pass; `0` disables |
| `--step` | offset sampling stride; default 2 |
| `--tiebreak` | `search` (default) or `none` |

Runs from any working directory. Handles missing and corrupt files with
exit code 2, colour input, paths containing spaces, and references at sizes
other than 1000 × 1000.

</details>

**Reproduce every number in this document**

```bash
python experiments/metrics.py      --data ./data --n 100 --baseline
python experiments/make_plots.py   --data ./data --n 100 --out ./figures --noise
python experiments/make_figures.py --data ./data --out ./figures
```

---

## The generator

Layers 1–3 build the wafer. Layer 4 photographs it, twice and
independently, because the reference and the search are separate physical
captures rather than two views of one exposure.

```mermaid
flowchart TD
    L1["<b>1 · Layout</b><br/>DRAM 6F² word / bit lines with contacts<br/>or FinFET fins with gate bars"]
    L2["<b>2 · Zones</b><br/>sub-array mats separated by peripheral strips<br/>each mat draws its own preset"]
    L3["<b>3 · Process</b><br/>line-edge roughness · corner rounding<br/>CD bias · missing contacts"]
    PSF["shared beam PSF<br/>one physical beam"]
    C1["<b>4a · Reference capture</b><br/>1 µm field · high dose<br/>no stage shear"]
    C2["<b>4b · Search capture</b><br/>10 µm field · ÷10 · low dose<br/>shear · jitter · charging"]
    MAN(["<b>5 · Manifest</b><br/>ground truth + every parameter"])

    L1 --> L2 --> L3 --> PSF
    PSF --> C1 --> MAN
    PSF --> C2 --> MAN

    style L1 fill:#161b22,stroke:#2f81f7,color:#e6edf3
    style L2 fill:#161b22,stroke:#2f81f7,color:#e6edf3
    style L3 fill:#161b22,stroke:#2f81f7,color:#e6edf3
    style PSF fill:#1c2128,stroke:#8b949e,color:#e6edf3
    style MAN fill:#0d2818,stroke:#3fb950,color:#e6edf3
```

<details>
<summary><b>Three things the reference generator does not have</b></summary>

<br>

**Edge-brightening** — listed as *mandatory* in the problem statement.
Secondary-electron yield rises with surface tilt roughly as `1/cos θ`, so
sidewalls emit more than flat tops and appear bright. It is the
characteristic signature that makes an SEM image recognisable.

**Line-edge roughness** — real lithographic edges wander with a
correlation length of tens of nanometres. Perfectly straight edges are the
clearest tell of a synthetic SEM image.

**Per-sample derived RNG** — three independent streams are spawned from
`(seed, index)`: geometry, reference capture, search capture. Changing an
imaging parameter therefore cannot move the crop position. The reference
generator shares one stream across its whole loop, so changing
`--shear-amplitude-px` silently desynchronises every sample after the
first — which breaks paired comparison without warning.

<br>

Validated against the reference generator on five independent statistics —
strip positions, strip widths, lattice pitches, noise floors and image
format — at **6× the speed**.

Every parameter is justified against public sources in
[`docs/CITATIONS.md`](docs/CITATIONS.md): **18 references**, no proprietary
fab data.

</details>

---

## A discrepancy worth stating

The two published descriptions of the tie-break rule disagree.

- The **Applied Materials deck**: closest to the *reference* image centre
- The **i4C problem statement**: closest to the *search* image centre

This implementation uses the **search-image centre**, because the reference
centre is not a location within the search image and so cannot order
candidates there. `--tiebreak none` disables the rule entirely.

The tolerance window matters more than it looks. Applying the rule with a
wide window was measured to cost **15 points**, because it encodes a
physical prior the test data does not contain: a real tool lands *near* its
target, but the reference generator places crops uniformly at random. The
shipped threshold fires only when two peaks are genuinely
indistinguishable. Measurements in
[`docs/FAILURES.md` §9.2](docs/FAILURES.md).

---

## Repository

```
generate_dataset.py      dataset generator          ← scored artifact
localize.py              localisation inference     ← scored artifact
requirements.txt         pinned dependencies

driftsense/
  generator/             patterns · zones · SEM imaging chain
  localize.py            shared helpers used by experiments

docs/
  CITATIONS.md           18 references, one per generator parameter
  FAILURES.md            root cause · oracles · 16 rejections

experiments/
  metrics.py             accuracy · AP · timing · conditions
  make_plots.py          the figures in this document
  make_figures.py        success and failure visuals
  …                      the measurement record behind FAILURES.md

figures/                 generated, committed for the documentation
```

**No deep-learning model is used**, so submission items 4 and 5 — model
weights and a training notebook — are not applicable. The classical method
was chosen because the problem geometry is fully constrained, and because
it leaves nothing that can fail to load on the evaluator's machine.

`driftsense/verifier.py` and `driftsense/residual.py` implement approaches
that were **measured and rejected**. They are retained so the results in
`docs/FAILURES.md` remain reproducible, and are imported by nothing on
either scored path.

---

## Documentation

| Document | Contents |
|:--|:--|
| [**`docs/CITATIONS.md`**](docs/CITATIONS.md) | Every structural, noise and augmentation parameter, with its physical justification and 18 verified public sources |
| [**`docs/FAILURES.md`**](docs/FAILURES.md) | Root-cause analysis · worked failure example · the identifiability predictor · five oracle results · sixteen rejected approaches · three retracted findings |

---

<div align="center">

**Team Lattice**

*Everything repeats. Exactness does not.*

</div>