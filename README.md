# Self-phase-modulation-assisted DPS-QKD

Code and data for

> P. Yousefi, M. Haghparast and T. Mikkonen, *Self-phase-modulation-assisted differential-phase-shift
> quantum key distribution: a weak-Kerr enhancement window under restricted eavesdropping*
> (University of Jyväskylä).

Every number, table and figure in the manuscript can be regenerated from this repository.

## Layout

```
paper/main.tex              manuscript (LaTeX, article class)
paper/figures/              all figures (PDF for the paper, PNG previews)
code/                       Python code
data/                       computed data behind the figures and tables
```

## Code

| File | Purpose |
|---|---|
| `code/tapping_model.py` | Joint optical model: lossy-Kerr state, tap, Bob's interferometer, Eve's displacement and interferometric receivers, accepted-slot states, local information, Holevo quantity, trace distance, phase-diffusion control |
| `code/compute_tapping_figures.py` | Data for Figures 3–7 and Table 2 → `data/tapping_figures.json` |
| `code/plot_tapping_figures.py` | Draws Figures 3–7 |
| `code/compute_window_channel.py`, `code/plot_window_channel.py` | Enhancement window and Kerr phase supplied by fibre (Section 8.6, Figure 8) |
| `code/compute_window_surface.py`, `code/plot_window_surface.py` | 3D map of the enhancement window over Kerr strength and intensity (Section 8.6) |
| `code/validate_reported_numbers.py` | Recomputes the headline numbers quoted in the text and checks them (23 checks) |
| `code/kerr_finite_key_budget.py` | Finite-key cost of Kerr deformation (Section 4, Figure 2) using the Mizutani–Takeuchi–Tamaki bound |
| `code/audit_numeric.py` | Finite-key kernel (Mizutani, Takeuchi & Tamaki, Phys. Rev. Research 5, 023132 (2023)) |
| `code/plot_finite_key_budget.py` | Draws Figure 2 |
| `code/kerr_error_bound_check.py` | Numerical check of the error bound of Section 4.2 (visibility bounds, Poisson populations) |
| `code/full_access_nogo_check.py` | Complete-access values quoted in Section 5.1 |
| `code/preattenuation_check.py` | Check of the Kerr-before-attenuator visibility formula |
| `code/channel_form_check.py` | Channel form of the accepted-slot state against the direct construction (Section 5.4) |
| `code/bound_monotonicity_check.py` | Holevo and trace-distance decrease at 350 scan points (Section 5.4) |

## Reproduce

```bash
pip install -r requirements.txt
python3 code/validate_reported_numbers.py                 # ~1 min
python3 code/compute_tapping_figures.py data/tapping_figures.json
python3 code/plot_tapping_figures.py .
python3 code/compute_window_channel.py data/window_channel.json
python3 code/plot_window_channel.py .
for e in 1 3 5; do python3 code/compute_window_surface.py data/window_surface_ed$e.json 0.0$e 0.9; done
python3 code/plot_window_surface.py .
python3 code/kerr_finite_key_budget.py code data
python3 code/plot_finite_key_budget.py .
```

Figure 1 is drawn with TikZ inside `paper/main.tex`.
