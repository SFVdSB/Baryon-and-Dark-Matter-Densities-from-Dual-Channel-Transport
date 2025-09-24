#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 transport (quadrature) — v3d
Same physics (integrals on background grid) plus visibility fixes:
  • thetas_delta.png           (ThetaLSP-mu, ThetaB-mu; linear)
  • thetas_delta_semilog.png   (|Theta-mu|; semilogy)
  • cumulative_dualaxis.png    (N_LSP and N_B on separate y-axes)
  • cumulative_normalized.png  (each cumulative divided by its own final value)
  • cumulative_zoom.png        (auto-zoom around wall window)
Other outputs unchanged.
"""
import os, json, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def ensure_dir(d): os.makedirs(d, exist_ok=True)

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
    c_r   = pick(["r","R"]); c_Phi = pick(["Phi","Phi_bg"]); c_phi = pick(["phi","varphi","phi_bg"])
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

def compute_on_grid(r, Phi_s, phi_s, mu, S2D1D, r_core):
    Phi  = Phi_s(r);  phi  = phi_s(r)
    dPhi = Phi_s.derivative()(r); dphi = phi_s.derivative()(r)
    r_soft = np.sqrt(r*r + r_core*r_core)
    src_L = (r_soft**3) * (dPhi**2)
    src_B = math.exp(-S2D1D) * (r_soft**3) * (dphi**2)
    ChiL = cumtrapz_strict(src_L, r); ChiB = cumtrapz_strict(src_B, r)
    ThetaL_raw = cumtrapz_strict(ChiL/(r_soft**3), r); ThetaB_raw = cumtrapz_strict(ChiB/(r_soft**3), r)
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

def write_outputs(outdir, phys, vis, mu, S2D1D, bg_points, eval_points):
    ensure_dir(outdir)
    rV = vis["r"]
    pd.DataFrame({
        "r": rV, "ThetaLSP": vis["ThetaL"], "ThetaB": vis["ThetaB"],
        "ChiLSP": vis["ChiL"], "ChiB": vis["ChiB"], "Phi": vis["Phi"], "phi": vis["phi"]
    }).to_csv(os.path.join(outdir,"transport_profile.csv"), index=False)
    pd.DataFrame({
        "r": rV, "J_LSP": vis["J_L"], "J_B": vis["J_B"], "ratio_running": vis["ratio_run"]
    }).to_csv(os.path.join(outdir,"currents_profile.csv"), index=False)
    pd.DataFrame({"r": rV, "N_LSP": vis["ChiL"], "N_B": vis["ChiB"], "Phi":vis["Phi"], "phi":vis["phi"]}
    ).to_csv(os.path.join(outdir,"cumulative_numbers.csv"), index=False)

    # Plots
    import numpy as np
    # fields
    plt.figure(); plt.plot(rV, vis["Phi"], label="Phi"); plt.plot(rV, vis["phi"], label="phi")
    plt.xlabel("r"); plt.ylabel("fields"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"fields.png"), dpi=150); plt.close()

    # derivatives
    plt.figure(); plt.semilogy(rV, np.abs(vis["dPhi"])+1e-300, label="|dPhi/dr|")
    plt.semilogy(rV, np.abs(vis["dphi"])+1e-300, label="|dphi/dr|")
    plt.xlabel("r"); plt.ylabel("abs(derivatives)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"derivatives.png"), dpi=150); plt.close()

    # sources
    plt.figure(); plt.semilogy(rV, np.maximum(vis["src_L"],1e-300), label="r^3 (dPhi)^2")
    plt.semilogy(rV, np.maximum(vis["src_B"],1e-300), label="e^{-S2} r^3 (dphi)^2")
    plt.xlabel("r"); plt.ylabel("sources"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"sources.png"), dpi=150); plt.close()

    # thetas (original)
    plt.figure(); plt.plot(rV, vis["ThetaL"], label=r"$\Theta_{\rm LSP}$"); plt.plot(rV, vis["ThetaB"], label=r"$\Theta_B$")
    plt.xlabel("r"); plt.ylabel("Theta"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"thetas.png"), dpi=150); plt.close()

    # thetas delta
    dTL = vis["ThetaL"] - mu; dTB = vis["ThetaB"] - mu
    plt.figure(); plt.plot(rV, dTL, label=r"$\Theta_{\rm LSP}-\mu$"); plt.plot(rV, dTB, label=r"$\Theta_B-\mu$")
    plt.xlabel("r"); plt.ylabel("Theta - mu"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"thetas_delta.png"), dpi=150); plt.close()

    # thetas delta semilog
    plt.figure(); 
    plt.semilogy(rV, np.abs(dTL)+1e-300, label=r"$|\Theta_{\rm LSP}-\mu|$")
    plt.semilogy(rV, np.abs(dTB)+1e-300, label=r"$|\Theta_B-\mu|$")
    plt.xlabel("r"); plt.ylabel("|Theta - mu|"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"thetas_delta_semilog.png"), dpi=150); plt.close()

    # cumulative (original)
    plt.figure(); plt.plot(rV, vis["ChiL"], label=r"$N_{\rm LSP}(r)$"); plt.plot(rV, vis["ChiB"], label=r"$N_B(r)$")
    plt.xlabel("r"); plt.ylabel("Cumulative number"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cumulative_numbers.png"), dpi=150); plt.close()

    # cumulative dual-axis
    fig, ax1 = plt.subplots()
    ax1.plot(rV, vis["ChiL"], label=r"$N_{\rm LSP}(r)$")
    ax1.set_xlabel("r"); ax1.set_ylabel(r"$N_{\rm LSP}(r)$")
    ax2 = ax1.twinx()
    ax2.plot(rV, vis["ChiB"], label=r"$N_B(r)$")
    ax2.set_ylabel(r"$N_B(r)$")
    fig.tight_layout(); fig.savefig(os.path.join(outdir,"cumulative_dualaxis.png"), dpi=150); plt.close(fig)

    # cumulative normalized
    NL = vis["ChiL"]; NB = vis["ChiB"]
    NLn = NL / (NL[-1] if NL[-1]!=0 else 1.0)
    NBn = NB / (NB[-1] if NB[-1]!=0 else 1.0)
    plt.figure(); plt.plot(rV, NLn, label=r"$N_{\rm LSP}(r)/N_{\rm LSP}(\infty)$")
    plt.plot(rV, NBn, label=r"$N_B(r)/N_B(\infty)$")
    plt.xlabel("r"); plt.ylabel("Cumulative (normalized)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cumulative_normalized.png"), dpi=150); plt.close()

    # cumulative zoom near wall
    i1, i2 = auto_wall_window(rV, vis["dPhi"], vis["dphi"])
    plt.figure(); plt.plot(rV[i1:i2+1], NL[i1:i2+1], label=r"$N_{\rm LSP}(r)$")
    plt.plot(rV[i1:i2+1], NB[i1:i2+1], label=r"$N_B(r)$")
    plt.xlabel("r (zoom)"); plt.ylabel("Cumulative number"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cumulative_zoom.png"), dpi=150); plt.close()

    summary = {"mu":mu,"S2D1D":S2D1D,"N_LSP":phys["N_LSP"],"N_B":phys["N_B"],"ratio":phys["ratio"],
               "bg_points":int(bg_points),"eval_points":int(eval_points),
               "notes":"Integrals on background grid; visibility fixes (delta, normalized, zoom)."}
    with open(os.path.join(outdir,"summary.json"),"w") as f: json.dump(summary,f,indent=2)

def tune_S2_for_target(r_int, Phi_s, phi_s, mu, r_core, target_ratio, S2_lo=0.0, S2_hi=10.0, tol=1e-3, max_iter=30):
    def ratio_at(S2):
        return compute_on_grid(r_int, Phi_s, phi_s, mu, S2, r_core)["ratio"]
    r_lo, r_hi = ratio_at(S2_lo), ratio_at(S2_hi); tries=0
    while not (min(r_lo,r_hi) <= target_ratio <= max(r_lo,r_hi)) and tries<5:
        S2_lo, S2_hi = max(0.0,S2_lo-2.0), S2_hi+2.0
        r_lo, r_hi = ratio_at(S2_lo), ratio_at(S2_hi); tries+=1
    a,b = S2_lo,S2_hi
    for _ in range(max_iter):
        m = 0.5*(a+b); Rm = ratio_at(m)
        if abs(Rm-target_ratio)/max(target_ratio,1e-12) < tol: return m,Rm
        if Rm < target_ratio: a = m
        else: b = m
    return m,Rm

def main():
    p = argparse.ArgumentParser(description="Stage-2 transport — v3d (visibility fixes).")
    p.add_argument("--bg-csv", required=True)
    p.add_argument("--mu", type=float, default=0.03)
    p.add_argument("--S2D1D", type=float, default=4.2)
    p.add_argument("--target-ratio", type=float, default=None)
    p.add_argument("--r-core", type=float, default=0.02)
    p.add_argument("--eval-n", type=int, default=20000)
    p.add_argument("--downsample-keep", type=int, default=None)
    p.add_argument("--outdir", default="runs/transport_stage2")
    args = p.parse_args()

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
        print(f"[tune] S2D1D ≈ {S2:.6g} {tuned_note}; achieved ratio ≈ {Rm:.4f}")

    phys = compute_on_grid(r_int, Phi_s, phi_s, mu, S2, r_core)
    N_eval = int(args.eval_n or args.downsample_keep or 20000)
    r_vis = np.linspace(r_bg[0], r_bg[-1], N_eval)
    vis = compute_on_grid(r_vis, Phi_s, phi_s, mu, S2, r_core)

    write_outputs(args.outdir, phys, vis, mu, S2, len(r_bg), len(r_vis))

    print("\n--- Stage 2 (quadrature, v3d) ---")
    print(f"N_LSP = {phys['N_LSP']:.6e}, N_B = {phys['N_B']:.6e}, ratio = {phys['ratio']:.6g}, mu(rmax) = {mu:.6g}")
    print(f"S2D1D = {S2:.6g} {tuned_note}")
    print(f"[note] Totals from background grid; extra plots expose structure even if base plots look flat.\n")

if __name__ == "__main__":
    main()
