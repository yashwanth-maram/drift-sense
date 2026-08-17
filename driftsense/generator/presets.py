"""Device geometry presets.

DRAM presets follow the 6F^2 folded-bitline cell: word-line pitch 2F,
bit-line pitch 3F, feature width ~F, contact diameter ~F. F is the
half-pitch of the technology node.

FinFET presets are parameterised by fin pitch and contacted poly pitch
(CPP, the gate pitch), which are the two numbers the IRDS roadmap tracks
per node.

Grey levels approximate secondary-electron yield ordering: substrate is
darkest, lower metal mid, upper metal brighter, contacts brightest, since
SE emission rises with atomic number and with topographic exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Preset:
    name: str
    kind: str                      # "dram" | "finfet"
    fine_pitch_nm: float           # word-line pitch (DRAM) / fin pitch (FinFET)
    coarse_pitch_nm: float         # bit-line pitch (DRAM) / gate pitch (FinFET)
    fine_width_nm: float
    coarse_width_nm: float
    contact_nm: float
    val_bg: int = 58
    val_fine: int = 112
    val_coarse: int = 150
    val_contact: int = 196

    def as_dict(self):
        return asdict(self)


def _dram(name, F):
    """6F^2 cell: word-line pitch 2F, bit-line pitch 3F, width ~F."""
    return Preset(name=name, kind="dram",
                  fine_pitch_nm=2 * F, coarse_pitch_nm=3 * F,
                  fine_width_nm=F, coarse_width_nm=F,
                  contact_nm=F * 0.85)


def _finfet(name, fin_pitch, cpp, fin_w, gate_w):
    return Preset(name=name, kind="finfet",
                  fine_pitch_nm=fin_pitch, coarse_pitch_nm=cpp,
                  fine_width_nm=fin_w, coarse_width_nm=gate_w,
                  contact_nm=fin_w * 1.1)


PRESETS = {
    "dram_1x":      _dram("dram_1x", 32),
    "dram_dense":   _dram("dram_dense", 24),
    "dram_compact": _dram("dram_compact", 36),
    "dram_loose":   _dram("dram_loose", 48),
    "dram_wide":    _dram("dram_wide", 60),
    "dram_legacy":  _dram("dram_legacy", 80),

    "finfet_7nm":   _finfet("finfet_7nm", 40, 76, 14, 26),
    "finfet_10nm":  _finfet("finfet_10nm", 48, 90, 17, 30),
    "finfet_14nm":  _finfet("finfet_14nm", 60, 110, 21, 38),
    "finfet_22nm":  _finfet("finfet_22nm", 80, 150, 28, 50),
    "finfet_28nm":  _finfet("finfet_28nm", 96, 180, 34, 60),
    "finfet_45nm":  _finfet("finfet_45nm", 140, 260, 50, 90),
}

DRAM_NAMES = [k for k, v in PRESETS.items() if v.kind == "dram"]
FINFET_NAMES = [k for k, v in PRESETS.items() if v.kind == "finfet"]


def presets_for_kind(kind: str):
    names = DRAM_NAMES if kind == "dram" else FINFET_NAMES
    return [PRESETS[n] for n in names]


def resolve_style(style: str) -> str:
    """Accept DRAM / dram / FinFET / finfet and any preset name."""
    s = style.strip().lower()
    if s in ("dram", "dram-style", "dram_style"):
        return "dram"
    if s in ("finfet", "finfet-style", "finfet_style"):
        return "finfet"
    if s in PRESETS:
        return PRESETS[s].kind
    raise ValueError(f"unknown style {style!r}")
