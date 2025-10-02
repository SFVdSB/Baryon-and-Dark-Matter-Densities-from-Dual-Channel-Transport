#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 transport (quadrature) - v3e (calibration-ready) + bias CSV

Adds:
- Console print of the bias factor: exp(-S2D1D).
- CSV 'channel_bias_and_ratios.csv' with columns:
    S2D1D, bias_factor_exp(-S2D1D), N_LSP_over_N_B, rho_dm_over_rho_b_model (if calibration present)
"""

import os, json, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# ---------- Small utils ----------
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def GeV_to_kg(E_GeV: float) -> float:
    # 1 GeV/c^2 = 1.78266192e-27 kg
    return float(E_GeV) * 1.78266192e-27

def load_background_csv(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    def pick(cands):
        for c in cands:
            if c in cols: return c
        low = {c.lower(): c for c in cols}
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        return None
    c_r   = pick(["r","R"])
    c_Phi = pick(["Phi","Phi_bg","PHI"])
    c_phi = pick(["phi","varphi","phi_bg","PHI_small","var"])
    if c_r is None or c_Phi is None or c_phi is None:
        raise ValueError(f"CSV missing required columns. Found: {cols}")
    r   = df[c_r].to_numpy(float)
    Phi = df[c_Phi].to_numpy(float)
    phi = df[c_phi].to_numpy(float)
    order = np.argsort(r)
    r, Phi, phi = r[order], Phi[order], phi[order]
    ur, idx = np.unique(r, return_index=True)
    return ur, Phi[idx], phi[idx]

def cumtrapz_strict(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    if len(x) < 2: return np.zeros_like(x)
    dx = np.diff(x); integ = 0.5*(y[1:] + y[:-1]) * dx
    out = np.empty_like(x); out[0]=0.0; out[1:] = np.cumsum(integ); return out

# ---------- Core physics on a given grid ----------
def compute_on_grid(r, Phi_s, phi_s, mu, S2D1D, r_core):
    Phi  = Phi_s(r);  phi  = phi_s(r)
    dPhi = Phi_s.derivative()(r); dphi = phi_s.derivative()(r)
    r_soft = np.sqrt(r*r + r_core*r_core)
    src_L = (r_soft**3) * (dPhi**2)
    src_B = math.exp(-S2D1D) * (r_soft**3) * (dphi**2)
    ChiL = cumtrapz_strict(src_L, r)
    ChiB = cumtrapz_strict(src_B, r)
    ThetaL_raw = cumtrapz_strict(ChiL/(r_soft**3), r)
    ThetaB_raw = cumtrapz_strict(ChiB/(r_soft**3), r)
    cL = mu - ThetaL_raw[-1]; cB = mu - ThetaB_raw[-1]
    ThetaL, ThetaB = ThetaL_raw + cL, ThetaB_raw + cB
    N_LSP, N_B = float(ChiL[-1]), float(ChiB[-1])
    J_L, J_B = dPhi**2, math.exp(-S2D1D)*(dphi**2)
    ratio_run = np.where(J_B>0, J_L/J_B, np.nan)
    return {"r":r,"Phi":Phi,"phi":phi,"dPhi":dPhi,"dphi":dphi,
            "ChiL":ChiL,"ChiB":ChiB,"ThetaL":ThetaL,"ThetaB":ThetaB,
            "src_L":src_L,"src_B":src_B,"J_L":J_L,"J_B":J_B,"ratio_run":ratio_run,
            "N_LSP":N_LSP,"N_B":N_B,"ratio":float(N_LSP/max(N_B,1e-300))}

def auto_wall_window(r, dPhi, dphi, span_frac=0.04):
    mag = np.abs(dPhi) + np.abs(dphi)
    k0 = int(np.argmax(mag))
    w = max(10, int(len(r)*span_frac))
    i1 = max(0, k0 - w); i2 = min(len(r)-1, k0 + w)
    return i1, i2

# ---------- Visualization & outputs ----------
def write_outputs(outdir, phys, vis, mu, S2D1D, bg_points, eval_points,
                  calib=None):
    ensure_dir(outdir)
    rV = vis["r"]
    # CSVs: main profiles
    pd.DataFrame({
        "r": rV, "ThetaLSP": vis["ThetaL"], "ThetaB": vis["ThetaB"],
        "ChiLSP": vis["ChiL"], "ChiB": vis["ChiB"], "Phi": vis["Phi"], "phi": vis["phi"]
    }).to_csv(os.path.join(outdir,"transport_profile.csv"), index=False)
    pd.DataFrame({
        "r": rV, "J_LSP": vis["J_L"], "J_B": vis["J_B"], "ratio_running": vis["ratio_run"]
    }).to_csv(os.path.join(outdir,"currents_profile.csv"), index=False)
    pd.DataFrame({"r": rV, "N_LSP": vis["ChiL"], "N_B": vis["ChiB"], "Phi":vis["Phi"], "phi":vis["phi"]}
    ).to_csv(os.path.join(outdir,"cumulative_numbers.csv"), index=False)

    # NEW: bias & ratios CSV (single-row)
    bias = math.exp(-S2D1D)
    row = {"S2D1D": S2D1D,
           "bias_factor_exp(-S2D1D)": bias,
           "N_LSP_over_N_B": phys["ratio"]}
    # If calibration present and both densities available, include rho ratio
    if calib is not None:
        pred = calib.get("predicted", {})
        rb = pred.get("rho_baryon_model_SI", None)
        rd = pred.get("rho_dm_model_SI", None)
        if (rb is not None) and (rd is not None):
            try:
                row["rho_dm_over_rho_b_model"] = float(rd) / float(rb)
            except Exception:
                pass
    pd.DataFrame([row]).to_csv(os.path.join(outdir, "channel_bias_and_ratios.csv"), index=False)

    # Plots
    plt.figure(); plt.plot(rV, vis["Phi"], label="Phi"); plt.plot(rV, vis["phi"], label="phi")
    plt.xlabel("r"); plt.ylabel("fields"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"fields.png"), dpi=150); plt.close()

    plt.figure(); plt.semilogy(rV, np.abs(vis["dPhi"]) + 1e-300, label="|dPhi/dr|")
    plt.semilogy(rV, np.abs(vis["dphi"]) + 1e-300, label="|dphi/dr|")
    plt.xlabel("r"); plt.ylabel("abs(derivatives)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"derivatives.png"), dpi=150); plt.close()

    plt.figure(); plt.semilogy(rV, np.maximum(vis["src_L"],1e-300), label="r^3 (dPhi)^2")
    plt.semilogy(rV, np.maximum(vis["src_B"],1e-300), label="e^{-S2} r^3 (dphi)^2")
    plt.xlabel("r"); plt.ylabel("sources"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"sources.png"), dpi=150); plt.close()

    plt.figure(); plt.plot(rV, vis["ThetaL"], label="Theta_LSP"); plt.plot(rV, vis["ThetaB"], label="Theta_B")
    plt.xlabel("r"); plt.ylabel("Theta"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"thetas.png"), dpi=150); plt.close()

    dTL = vis["ThetaL"] - mu; dTB = vis["ThetaB"] - mu
    plt.figure(); plt.plot(rV, dTL, label="Theta_LSP - mu"); plt.plot(rV, dTB, label="Theta_B - mu")
    plt.xlabel("r"); plt.ylabel("Theta - mu"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"thetas_delta.png"), dpi=150); plt.close()

    plt.figure()
    plt.semilogy(rV, np.abs(dTL) + 1e-300, label="|Theta_LSP - mu|")
    plt.semilogy(rV, np.abs(dTB) + 1e-300, label="|Theta_B - mu|")
    plt.xlabel("r"); plt.ylabel("|Theta - mu|"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"thetas_delta_semilog.png"), dpi=150); plt.close()

    plt.figure(); plt.plot(rV, vis["ChiL"], label="N_LSP(r)"); plt.plot(rV, vis["ChiB"], label="N_B(r)")
    plt.xlabel("r"); plt.ylabel("Cumulative number-like"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cumulative_numbers.png"), dpi=150); plt.close()

    fig, ax1 = plt.subplots()
    ax1.plot(rV, vis["ChiL"], label="N_LSP(r)")
    ax1.set_xlabel("r"); ax1.set_ylabel("N_LSP(r)")
    ax2 = ax1.twinx()
    ax2.plot(rV, vis["ChiB"], label="N_B(r)")
    ax2.set_ylabel("N_B(r)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir,"cumulative_dualaxis.png"), dpi=150); plt.close(fig)

    NL = vis["ChiL"]; NB = vis["ChiB"]
    NLn = NL / (NL[-1] if NL[-1]!=0 else 1.0)
    NBn = NB / (NB[-1] if NB[-1]!=0 else 1.0)
    plt.figure(); plt.plot(rV, NLn, label="N_LSP/N_LSP(inf)")
    plt.plot(rV, NBn, label="N_B/N_B(inf)")
    plt.xlabel("r"); plt.ylabel("Cumulative (normalized)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cumulative_normalized.png"), dpi=150); plt.close()

    i1, i2 = auto_wall_window(rV, vis["dPhi"], vis["dphi"])
    plt.figure(); plt.plot(rV[i1:i2+1], NL[i1:i2+1], label="N_LSP(r)")
    plt.plot(rV[i1:i2+1], NB[i1:i2+1], label="N_B(r)")
    plt.xlabel("r (zoom)"); plt.ylabel("Cumulative number-like"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cumulative_zoom.png"), dpi=150); plt.close()

    # Summary JSON (plus optional calibration block)
    summary = {"mu":mu,"S2D1D":S2D1D,
               "N_LSP":phys["N_LSP"],"N_B":phys["N_B"],"ratio":phys["ratio"],
               "bg_points":int(bg_points),"eval_points":int(eval_points),
               "notes":"Integrals on background grid; v3e adds optional density calibration."}
    if calib is not None:
        summary.update({"calibration": calib})
        pd.DataFrame([calib]).to_csv(os.path.join(outdir,"densities_calibration_summary.csv"), index=False)

    with open(os.path.join(outdir,"summary.json"),"w") as f: json.dump(summary,f,indent=2)

# ---------- Calibration helpers ----------
def run_calibration(phys, rho_b_obs_SI, rho_dm_obs_SI,
                    m_b_GeV, m_lsp_GeV,
                    common_norm: bool):
    """
    Map model cumulants (N_B, N_LSP) to mass densities in SI via linear scale(s).
    Define alpha_B, alpha_L such that:
        rho_b_model_SI  = alpha_B * N_B
        rho_dm_model_SI = alpha_L * N_LSP
    If common_norm == True, enforce alpha_B == alpha_L == alpha, solve alpha from
    whichever observed density is provided (prefer baryon if given), then predict the other.
    Optionally, if BOTH observed densities are provided AND common_norm is True,
    we also report the 'implied' m_LSP if you interpret alpha as mapping to number
    density and then multiply by masses. This is heuristic.
    """
    N_B   = phys["N_B"]; N_L = phys["N_LSP"]
    out = {
        "observed": {"rho_baryon_SI": rho_b_obs_SI, "rho_dm_SI": rho_dm_obs_SI},
        "masses": {"m_baryon_GeV": m_b_GeV, "m_LSP_GeV": m_lsp_GeV},
        "model": {"N_B": N_B, "N_LSP": N_L, "ratio_NL_over_NB": phys["ratio"]},
        "assumptions": {"common_norm": bool(common_norm)},
    }

    m_b_kg   = GeV_to_kg(m_b_GeV) if (m_b_GeV is not None) else None
    m_lsp_kg = GeV_to_kg(m_lsp_GeV) if (m_lsp_GeV is not None) else None

    if not common_norm:
        alpha_B = rho_b_obs_SI / N_B if (rho_b_obs_SI is not None) else None
        alpha_L = rho_dm_obs_SI / N_L if (rho_dm_obs_SI is not None) else None

        rho_b_model = (alpha_B * N_B) if (alpha_B is not None) else None
        rho_dm_model= (alpha_L * N_L) if (alpha_L is not None) else None

        n_b = (rho_b_model / m_b_kg) if (rho_b_model is not None and m_b_kg) else None
        n_dm= (rho_dm_model / m_lsp_kg) if (rho_dm_model is not None and m_lsp_kg) else None

        out["scales"] = {"alpha_B_SI": alpha_B, "alpha_L_SI": alpha_L}
        out["predicted"] = {"rho_baryon_model_SI": rho_b_model,
                            "rho_dm_model_SI": rho_dm_model,
                            "n_baryon_model_m3": n_b,
                            "n_dm_model_m3": n_dm}
        return out

    alpha = None
    if rho_b_obs_SI is not None:
        alpha = rho_b_obs_SI / N_B
    elif rho_dm_obs_SI is not None:
        alpha = rho_dm_obs_SI / N_L

    rho_b_model = alpha * N_B if alpha is not None else None
    rho_dm_model= alpha * N_L if alpha is not None else None

    n_b = (rho_b_model / m_b_kg) if (rho_b_model is not None and m_b_kg) else None
    n_dm= (rho_dm_model / m_lsp_kg) if (rho_dm_model is not None and m_lsp_kg) else None

    out["scales"] = {"alpha_common_SI": alpha}
    out["predicted"] = {"rho_baryon_model_SI": rho_b_model,
                        "rho_dm_model_SI": rho_dm_model,
                        "n_baryon_model_m3": n_b,
                        "n_dm_model_m3": n_dm}

    if (rho_b_obs_SI is not None) and (rho_dm_obs_SI is not None):
        implied_mLSP_GeV = None
        if (m_b_GeV is not None) and (phys["ratio"] != 0):
            implied_mLSP_GeV = (rho_dm_obs_SI / max(rho_b_obs_SI,1e-300)) * (phys["N_B"] / max(phys["N_LSP"],1e-300)) * m_b_GeV
        out["consistency"] = {
            "observed_rho_dm_over_rho_b": (rho_dm_obs_SI / rho_b_obs_SI) if (rho_b_obs_SI and rho_dm_obs_SI) else None,
            "model_NL_over_NB": phys["ratio"],
            "implied_mLSP_GeV_if_common_norm": implied_mLSP_GeV
        }
    return out

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 transport - v3e (with optional density calibration).")
    p.add_argument("--bg-csv", required=True)
    p.add_argument("--mu", type=float, default=0.03)
    p.add_argument("--S2D1D", type=float, default=4.2)
    p.add_argument("--target-ratio", type=float, default=None)
    p.add_argument("--r-core", type=float, default=0.02)
    p.add_argument("--eval-n", type=int, default=20000)
    p.add_argument("--downsample-keep", type=int, default=None)
    p.add_argument("--outdir", default="runs/transport_stage2")

    # Calibration flags (all optional)
    p.add_argument("--rho-baryon-SI", type=float, default=None, help="Observed present-day baryon mass density in kg/m^3 (e.g., 4.17e-28).")
    p.add_argument("--rho-dm-SI", type=float, default=None, help="Observed present-day DM mass density in kg/m^3 (e.g., 2.23e-27).")
    p.add_argument("--m-baryon-GeV", type=float, default=0.9382720813, help="Baryon mass to use (GeV). Default: proton mass.")
    p.add_argument("--m-lsp-GeV", type=float, default=None, help="Assumed LSP mass in GeV. If omitted, some derived quantities will be left blank.")
    p.add_argument("--common-norm", action="store_true", help="Use a single normalization constant for both channels.")
    return p.parse_args()

# ---------- Optional: tuner ----------
def tune_S2_for_target(r_int, Phi_s, phi_s, mu, r_core, target_ratio, S2_lo=-5.0, S2_hi=10.0, tol=1e-3, max_iter=30):
    def ratio_at(S2):
        return compute_on_grid(r_int, Phi_s, phi_s, mu, S2, r_core)["ratio"]
    r_lo, r_hi = ratio_at(S2_lo), ratio_at(S2_hi); tries=0
    while not (min(r_lo,r_hi) <= target_ratio <= max(r_lo,r_hi)) and tries<6:
        S2_lo -= 2.0
        S2_hi += 2.0
        r_lo, r_hi = ratio_at(S2_lo), ratio_at(S2_hi); tries+=1
    a,b = S2_lo,S2_hi
    for _ in range(max_iter):
        m = 0.5*(a+b); Rm = ratio_at(m)
        if abs(Rm-target_ratio)/max(target_ratio,1e-12) < tol: return m,Rm
        if Rm < target_ratio: a = m
        else: b = m
    return m,Rm

# ---------- Main ----------
def main():
    args = parse_args()
    ensure_dir(args.outdir)

    r_bg, Phi_bg, phi_bg = load_background_csv(args.bg_csv)
    print(f"[bg] {args.bg_csv}: points={len(r_bg)}; r in [{r_bg[0]:.3e}, {r_bg[-1]:.3f}]")

    Phi_s, phi_s = PchipInterpolator(r_bg, Phi_bg), PchipInterpolator(r_bg, phi_bg)
    r_int = r_bg.copy()
    mu, r_core = float(args.mu), float(args.r_core)

    S2 = float(args.S2D1D)
    tuned_note = ""
    if args.target_ratio is not None:
        S2, Rm = tune_S2_for_target(r_int, Phi_s, phi_s, mu, r_core, float(args.target_ratio))
        tuned_note = f"(auto-tuned to target {args.target_ratio})"
        print(f"[tune] S2D1D ~ {S2:.6g} {tuned_note}; achieved ratio ~ {Rm:.4f}")

    phys = compute_on_grid(r_int, Phi_s, phi_s, mu, S2, r_core)
    N_eval = int(args.eval_n or args.downsample_keep or 20000)
    r_vis = np.linspace(r_bg[0], r_bg[-1], N_eval)
    vis = compute_on_grid(r_vis, Phi_s, phi_s, mu, S2, r_core)

    # Optional calibration
    calib = None
    if any(v is not None for v in [args.rho_baryon_SI, args.rho_dm_SI, args.m_lsp_GeV]):
        calib = run_calibration(phys,
                                rho_b_obs_SI=args.rho_baryon_SI,
                                rho_dm_obs_SI=args.rho_dm_SI,
                                m_b_GeV=args.m_baryon_GeV,
                                m_lsp_GeV=args.m_lsp_GeV,
                                common_norm=args.common_norm)

    # Write outputs (incl. bias CSV if present)
    write_outputs(args.outdir, phys, vis, mu, S2, len(r_bg), len(r_vis), calib=calib)

    # Console summary
    print("\n--- Stage 2 (quadrature, v3e) ---")
    print(f"N_LSP = {phys['N_LSP']:.6e}, N_B = {phys['N_B']:.6e}, N_LSP/N_B = {phys['ratio']:.6g}, mu(rmax) = {mu:.6g}")
    # NEW: print bias factor
    print(f"S2D1D = {S2:.6g} {tuned_note} | bias exp(-S2D1D) = {math.exp(-S2):.6g}")
    if calib is not None:
        pred = calib.get("predicted", {})
        rb = pred.get("rho_baryon_model_SI", None)
        rd = pred.get("rho_dm_model_SI", None)
        if (rb is not None) and (rd is not None):
            try:
                print(f"[ratios] rho_dm0/rho_b0 = {float(rd)/float(rb):.6g}")
            except Exception:
                pass
        print("[cal] wrote densities_calibration_summary.csv and added 'calibration' to summary.json")
    print("[note] Totals come from background grid; density calibration is phenomenological.\n")

if __name__ == "__main__":
    main()
