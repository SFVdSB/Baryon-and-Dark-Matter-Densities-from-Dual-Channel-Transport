#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Yields and Today's Densities from Stage-2 Summary (fixed attrs)

- Reads Stage-2 summary.json (from matterDensities2_Quadrature_v3e_calibrated.py)
- Computes yields Y_B, Y_DM via a single normalization C: Y = C * N
- Two normalization modes:
    --fix rho_b : set C so that today's baryon mass density matches --rho-baryon-SI
    --fix etaB  : set C so that today's eta_B matches --etaB  (eta_B = 7.04 * Y_B)
- Reports number densities n0 and mass densities rho0 for baryons and DM (if m_LSP given)
"""
import json, argparse

GEV_TO_KG = 1.78266192e-27  # 1 GeV/c^2 in kg

def GeV_to_kg(x: float) -> float:
    return float(x) * GEV_TO_KG

def parse_args():
    p = argparse.ArgumentParser(description="Compute Y_B, Y_DM, eta_B, and z=0 densities from Stage-2 summary.")
    p.add_argument("--summary", required=True, help="Path to summary.json produced by Stage-2.")
    p.add_argument("--fix", choices=["rho_b","etaB"], default="rho_b",
                   help="Fix normalization via today's baryon mass density (rho_b) or eta_B (baryon-to-photon)." )
    p.add_argument("--rho-baryon-SI", dest="rho_baryon_SI", type=float, default=4.17e-28,
                   help="Target present-day baryon mass density in kg/m^3 (if --fix rho_b)." )
    p.add_argument("--etaB", type=float, default=6.1e-10, help="Target baryon asymmetry (if --fix etaB)." )
    p.add_argument("--m-baryon-GeV", dest="m_baryon_GeV", type=float, default=0.9382720813,
                   help="Baryon mass (proton approx) in GeV." )
    p.add_argument("--m-lsp-GeV", dest="m_lsp_GeV", type=float, default=None,
                   help="LSP mass in GeV (optional; if omitted, DM mass density isn't computed)." )
    p.add_argument("--s0_per_cm3", type=float, default=2891.0, help="Present-day entropy density s0 in cm^-3 (7.04 * 410.7)." )
    p.add_argument("--nGamma0_per_cm3", type=float, default=410.7, help="Photon number density n_gamma,0 in cm^-3." )
    p.add_argument("--out", default=None, help="Optional CSV path to write results." )
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.summary, "r", encoding="utf-8") as f:
        S = json.load(f)

    N_B = float(S["N_B"])
    N_L = float(S["N_LSP"])
    ratio = float(S.get("ratio", N_L / max(N_B, 1e-300)))

    # Convert constants to SI (per m^3)
    s0 = args.s0_per_cm3 * 1e6

    # Choose normalization: Y = C * N
    if args.fix == "rho_b":
        m_b_kg = GeV_to_kg(args.m_baryon_GeV)
        # rho_b0 = m_b * Y_B * s0 => Y_B = rho_b0 / (m_b * s0)
        rho_b0 = args.rho_baryon_SI
        Y_B = rho_b0 / (m_b_kg * s0)
        C = Y_B / max(N_B, 1e-300)
    else:
        etaB_target = args.etaB
        # eta_B = 7.04 * Y_B => Y_B = eta_B / 7.04
        Y_B = etaB_target / 7.04
        C = Y_B / max(N_B, 1e-300)

    # Predict DM yield from same C
    Y_DM = C * N_L

    # Today's number densities
    n_B0 = Y_B * s0
    n_DM0 = Y_DM * s0

    # Today's mass densities
    m_b_kg = GeV_to_kg(args.m_baryon_GeV)
    rho_b0_model = m_b_kg * n_B0

    rho_dm0_model = None
    if args.m_lsp_GeV is not None:
        m_LSP_kg = GeV_to_kg(args.m_lsp_GeV)
        rho_dm0_model = m_LSP_kg * n_DM0

    # eta_B prediction from Y_B
    etaB_model = 7.04 * Y_B

    # Print summary
    print("=== Yield & Density Summary ===")
    print(f"Normalization mode: {args.fix}")
    print(f"C (yield per N): {C:.6e}")
    print(f"N_LSP/N_B (from Stage-2): {ratio:.6g}")
    print(f"Y_B:  {Y_B:.6e}")
    print(f"Y_DM: {Y_DM:.6e}")
    print(f"eta_B (model): {etaB_model:.6e}")
    print(f"n_B0  [1/m^3]: {n_B0:.6e}")
    print(f"n_DM0 [1/m^3]: {n_DM0:.6e}")
    print(f"rho_b0_model  [kg/m^3]: {rho_b0_model:.6e}")
    if rho_dm0_model is not None:
        print(f"rho_dm0_model [kg/m^3]: {rho_dm0_model:.6e}")
    else:
        print("rho_dm0_model [kg/m^3]: (provide --m-lsp-GeV to compute)")

if __name__ == "__main__":
    main()
