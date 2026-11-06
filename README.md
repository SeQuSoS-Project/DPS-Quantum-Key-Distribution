# SPM-assisted DPS-QKD

This repository holds the code, data and figures for a study of **self-phase modulation (SPM)**,
the Kerr nonlinearity of optical media, in **differential-phase-shift quantum key distribution
(DPS-QKD)**. The study shows that SPM leaves the general-attack finite-key security proof of DPS-QKD
valid at a small, bounded cost in error rate. Against an eavesdropper who taps part of each pulse, it
removes information from her light that no operation on that light can restore, and within a
weak-Kerr window this increases the key margin per accepted slot.

The work was carried out at the [Quantum Programming and Algorithms (QPA) Research Group](https://users.jyu.fi/~mhaghpyz/#qpa),
[University of Jyväskylä](https://www.jyu.fi/en), Finland.

Every number, table and plotted figure (Figures 2–9) in the manuscript can be regenerated from this
repository.
`code/validate_reported_numbers.py` recomputes the numbers quoted in the text and checks them against
the manuscript (23 checks).

## Manuscript

> P. Yousefi, M. Haghparast and T. Mikkonen, *Self-phase-modulation-assisted differential-phase-shift
> quantum key distribution: a weak-Kerr enhancement window under restricted eavesdropping*,
> University of Jyväskylä (2026).

## Main results

- **The security proof still holds.** Kerr evolution inside the transmitter preserves the three source
  assumptions of the Mizutani–Takeuchi–Tamaki finite-key proof, provided it is independent between
  pulses. Kerr evolution in the channel is covered because the proof allows an arbitrary channel.
- **The error cost is small and bounded.** The added visibility error is at most about twice the mean
  photon number in the nonlinear medium. SPM in standard fibre lies six orders of magnitude below the
  finite-key budget (Section 4, Figure 2).
- **There is no effect with full access.** An eavesdropper with access to the whole pulse is affected
  only at second order in the mean photon number (Section 5.1).
- **Heralded cross-phase.** For an eavesdropper who taps a fraction of each pulse after the nonlinear
  segment, SPM acts as a cross-Kerr coupling between her light and Bob's. Bob's public sifting
  announcement heralds the resulting phase. At µ0 = 0.1, this lowers the Holevo quantity of her light
  by 36.5% at a Kerr phase that gives Bob a 2.69% QBER, which is three to four times more than phase
  noise of equal visibility (Sections 5–8).
- **Enhancement window.** At µ0 = 0.05, r = 0.9 and D = 50 km, the margin gain ΔMχ is positive up to
  about 0.15, 0.35 and 0.54 rad for baseline errors of 1%, 3% and 5%. At e_d = 1%, the largest gain is
  about 2.6 × 10⁻⁴ bits per accepted slot, near µ0 = 0.06 and Θ = 0.08 (Section 8.6, Figures 8–9).

## Model and scope

The code evolves each pulse with the exact lossy-Kerr solution for a single temporal mode, without
dispersion and with distributed loss. Raman noise, cross-phase modulation by other channels,
pulse-shape changes and memory between pulses are outside the model. Unless stated otherwise, the
nonlinear segment is lossless (A = 0), Eve taps half of each pulse (r = 0.5) with ideal detectors, and
Bob's link after the tap is 50 km long.

The proof, the error budget and the full-access result hold against general attacks. The heralded
cross-phase and the enhancement window apply only to a restricted, tapping eavesdropper. This is the
model used for monitored line-of-sight and fibre links, where the tapped fraction r is bounded and
measured.

## Getting started

Requires Python 3 with NumPy, SciPy and Matplotlib (`requirements.txt`). The full pipeline was last
run on 2026-11-06 with Python 3.13, NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2.

```bash
git clone https://github.com/SeQuSoS-Project/DPS-Quantum-Key-Distribution.git
cd DPS-Quantum-Key-Distribution
pip install -r requirements.txt
python3 code/validate_reported_numbers.py      # 23 checks, a few seconds
```

## Reproducing the figures

Run from the repository root. The whole sequence took about 4 minutes in the test run, most of it in
`compute_tapping_figures.py`.

```bash
python3 code/compute_tapping_figures.py data/tapping_figures.json
python3 code/plot_tapping_figures.py .
python3 code/compute_window_channel.py data/window_channel.json
python3 code/plot_window_channel.py .
for e in 1 3 5; do python3 code/compute_window_surface.py data/window_surface_ed$e.json 0.0$e 0.9; done
python3 code/plot_window_surface.py .
python3 code/kerr_finite_key_budget.py code data
python3 code/plot_finite_key_budget.py .
```

| Figure | File in `paper/figures/` | Data in `data/` |
| --- | --- | --- |
| 2 | `finite_key_budget` | `finite_key_budget.json` |
| 3 | `operating_windows` | `tapping_figures.json` |
| 4 | `heralded_phase` | `tapping_figures.json` |
| 5 | `interferometric_eve` | `tapping_figures.json` |
| 6 | `holevo_bounds` | `tapping_figures.json` |
| 7 | `exchange_ratio` | `tapping_figures.json` |
| 8 | `window_channel` | `window_channel.json` |
| 9 | `window_surface` | `window_surface_ed1.json`, `_ed3`, `_ed5` |

Each figure is saved as PDF (used in the paper) and PNG (preview). Table 2 comes from
`tapping_figures.json`.

## Code

| File | Purpose |
| --- | --- |
| `code/tapping_model.py` | Joint optical model: lossy-Kerr state, tap, Bob's interferometer, Eve's displacement and interferometric receivers, accepted-slot states, local information, Holevo quantity, trace distance, phase-diffusion control |
| `code/compute_tapping_figures.py`, `code/plot_tapping_figures.py` | Figures 3–7 and Table 2 |
| `code/compute_window_channel.py`, `code/plot_window_channel.py` | Enhancement window and the Kerr phase supplied by fibre (Section 8.6, Figure 8) |
| `code/compute_window_surface.py`, `code/plot_window_surface.py` | 3D map of the enhancement window over Kerr strength and intensity (Section 8.6, Figure 9) |
| `code/kerr_finite_key_budget.py`, `code/plot_finite_key_budget.py` | Finite-key cost of Kerr deformation with the Mizutani–Takeuchi–Tamaki bound (Section 4, Figure 2) |
| `code/audit_numeric.py` | Finite-key kernel (Mizutani, Takeuchi & Tamaki, Phys. Rev. Research 5, 023132 (2023)) |
| `code/validate_reported_numbers.py` | Recomputes the headline numbers quoted in the text and checks them (23 checks) |
| `code/kerr_error_bound_check.py` | Numerical check of the error bound of Section 4.2 (visibility bounds, Poisson populations) |
| `code/full_access_nogo_check.py` | Complete-access values quoted in Section 5.1 |
| `code/preattenuation_check.py` | Check of the Kerr-before-attenuator visibility formula |
| `code/channel_form_check.py` | Channel form of the accepted-slot state against the direct construction (Section 5.4) |
| `code/bound_monotonicity_check.py` | Holevo and trace-distance decrease at 350 scan points (Section 5.4) |

## Repository layout

```text
.
├── README.md
├── requirements.txt       NumPy, SciPy, Matplotlib
├── paper/
│   └── figures/           Figures 2–9 (PDF for the paper, PNG previews)
├── code/                  Optical model, computations, plots and checks
└── data/                  Computed data behind the figures and tables (JSON)
```

## Authors

- Pedram Yousefi - *University of Jyväskylä*
- Majid Haghparast - *University of Jyväskylä* (corresponding author)
- Tommi Mikkonen - *University of Jyväskylä*

## Contact

For questions about the code, open an issue in this repository. Corresponding author: Majid
Haghparast <<majid.m.haghparast@jyu.fi>>.

## Acknowledgement

This work was supported by Business Finland through project SeQuSoS (Grant No. 112/31/2024) and by
the Research Council of Finland through project Profi 8 – qSIME (Grant No. 365343).

## Citing

If you use this code or data, please cite the manuscript:

```bibtex
@unpublished{yousefi2026spmdps,
  author = {Yousefi, Pedram and Haghparast, Majid and Mikkonen, Tommi},
  title  = {Self-phase-modulation-assisted differential-phase-shift quantum key distribution:
            a weak-Kerr enhancement window under restricted eavesdropping},
  note   = {Manuscript, University of Jyv{\"a}skyl{\"a}},
  year   = {2026}
}
```
