# Baryon and Dark Matter Densities from Dual-Channel Transport (SFV/dSB)

Reproducible pipeline coupling a two-field O(4) bounce background to dual-channel transport (LSP and baryons). Stabilized numerics (PCHIP, integrals on original grid) and diagnostics plots. The tuned value `S2D1D ≈ 0.94` yields `n_LSP/n_B ≈ 5.7`.

- **Paper (LaTeX)**: see `/paper/` (or Overleaf) and placeholder DOI: https://doi.org/TBD
- **This repo**: https://github.com/SFVdSB/Baryon-and-Dark-Matter-Densities-from-Dual-Channel-Transport

## 0) Environment

### Conda (recommended)
```bash
conda env create -f env/environment.yml
conda activate sfv-dsb
