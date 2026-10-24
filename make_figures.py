"""make_figures.py - builds Figures 3-8 of the revised manuscript from results/*.json"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import case_study as cs

R = os.path.join(os.path.dirname(__file__), "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1f5aa6", "#e08214", "#2b8a3e", "#c0392b", "#7f7f7f", "#6a3d9a"
W = json.load(open(f"{R}/worked_example.json"))

# ---------------------------------------------------------------- Fig: case graph
def fig_graph():
    pos = {"WEB": (0, 5), "MAIL": (0, 4), "AD": (1.3, 4), "BILL": (1.3, 5.2), "VPN": (2.6, 4.6),
           "SUP": (2.6, 6.1), "CLOUD": (4.1, 6.1), "JUMP": (3.9, 4.0), "HIST": (5.2, 4.9),
           "EWS": (5.2, 3.0), "SCADA": (6.5, 4.0), "HMI": (7.8, 4.9), "RTU": (6.5, 2.0),
           "IED": (7.8, 1.0), "TEL": (5.0, 0.9), "RAD": (5.0, 0.0),
           "F1": (9.3, 2.6), "F2": (9.3, 4.6), "F3": (2.6, 6.9 - 0.6), "F4": (10.6, 1.4)}
    pos["F3"] = (0.1, 6.3)
    col = {"Operator": "#dbe8f7", "Supplier": "#fde3c8", "Telco": "#e5dcf3", "WaterCo": "#d8f0dc"}
    edg = {"Operator": BLUE, "Supplier": ORANGE, "Telco": PURPLE, "WaterCo": GREEN}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    def arrow(a, b, color, ls, lw, rad):
        ax.annotate("", xy=pos[b], xytext=pos[a],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=13, shrinkB=13,
                                    connectionstyle=f"arc3,rad={rad}", mutation_scale=8))
    for a, b, w in cs.ATT_ARCS:
        arrow(a, b, RED, "--", 0.6 + 1.6 * w, 0.12)
    for a, b, w in cs.CON_ARCS:
        arrow(a, b, BLUE, "-", 0.6 + 1.6 * w, -0.12)
    for (members, j, w) in cs.AND_GROUPS:
        for m in members:
            arrow(m, j, PURPLE, ":", 1.8, 0.0)
    for nid, lab, own, kind in cs.NODES:
        x, y = pos[nid]
        box = "round,pad=0.25" if kind != "Function" else "square,pad=0.3"
        ax.text(x, y, nid if kind != "Function" else nid.replace("F", "EF"), ha="center", va="center",
                fontsize=7.5, fontweight="bold" if kind == "Function" else "normal",
                bbox=dict(boxstyle=box, fc=col[own], ec=edg[own], lw=1.2 if kind == "Function" else 0.8))
    ax.text(5.55, 0.55, "AND", fontsize=6.5, color=PURPLE, ha="left", va="center")
    ax.set_xlim(-0.6, 11.2); ax.set_ylim(-0.6, 6.9); ax.axis("off")
    leg = [Line2D([0], [0], color=RED, ls="--", lw=1.5, label="Attack layer (reachability, identity, access)"),
           Line2D([0], [0], color=BLUE, lw=1.5, label="Consequence layer (service, control, data)"),
           Line2D([0], [0], color=PURPLE, ls=":", lw=1.8, label="Redundancy (AND) group"),
           Patch(fc=col["Operator"], ec=edg["Operator"], label="Operator"),
           Patch(fc=col["Supplier"], ec=edg["Supplier"], label="Supplier"),
           Patch(fc=col["Telco"], ec=edg["Telco"], label="Telecom provider"),
           Patch(fc=col["WaterCo"], ec=edg["WaterCo"], label="Dependent water utility")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(-0.02, -0.30), ncol=3, frameon=False, fontsize=7)
    fig.savefig(f"{F}/fig_case_graph.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: worked example ranking + robustness
def fig_case_ranking():
    sc = W["scenarios"]; ids = [s["id"] for s in sc]
    R0 = np.array([s["R0"] for s in sc])
    Rw5 = R0.copy(); Rw5[1] = W["p_now"][1] * sc[1]["C"]
    M = np.array([s["matrix_score"] for s in sc]); cv = np.array([s["cvss"] for s in sc])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x = np.arange(len(ids)); w = 0.38
    ax = axs[0]
    ax.bar(x - w / 2, R0, w, color=BLUE, label="CORA-OT, week 0")
    ax.bar(x + w / 2, Rw5, w, color=ORANGE, label="CORA-OT, week 5 (after evidence)")
    ax.set_xticks(x); ax.set_xticklabels(ids); ax.set_ylabel("Risk (points/year)")
    ax2 = ax.twinx(); ax2.plot(x, M, "kD", ms=4, label="5x5 matrix score"); ax2.plot(x, cv, "s", color=GREY, mfc="none", ms=5, label="CVSS")
    ax2.set_ylabel("Matrix score / CVSS"); ax2.set_ylim(0, 26); ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=7)
    ax.set_ylim(0, 14.5); ax.set_title("(a) Scenario risk and conventional scores")
    ax = axs[1]
    rk = np.load(f"{R}/rank_samples.npy")
    bp = ax.boxplot([rk[:, i] for i in range(rk.shape[1])], tick_labels=ids, whis=(5, 95), showfliers=False,
                    patch_artist=True, medianprops=dict(color="k"))
    for b in bp["boxes"]: b.set_facecolor("#dbe8f7")
    ax.invert_yaxis(); ax.set_ylabel("Rank (1 = highest)"); ax.set_yticks(range(1, 9))
    ax.set_title("(b) Rank stability, week 0 (2000 perturbations)")
    fig.tight_layout(); fig.savefig(f"{F}/fig_case_ranking.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: evidence update
def fig_evidence():
    T = W["timeline"]; E3 = json.load(open(f"{R}/E3_calibration.json"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axs[0]; wk = T["weeks"]
    ax.step(wk, T["naive"], where="post", color=RED, ls="--", label="Naive (duplicates counted)")
    ax.step(wk, T["geomean"], where="post", color=ORANGE, ls="-.", label="Geo-mean w, clustered")
    ax.step(wk, T["product"], where="post", color=BLUE, lw=2, label="CORA-OT")
    for w_, lab, yy in [(2, "TI", 0.97), (4, "CR-17 rec.", 0.97), (5, "CR-17 valid.", 0.89), (6, "2 duplicates", 0.97), (9, "Fix verified", 0.97)]:
        ax.axvline(w_, color=GREY, lw=0.5, ls=":"); ax.text(w_ + 0.08, yy, lab, fontsize=5.8, va="top", rotation=0,
                                                         bbox=dict(fc="white", ec="none", pad=0.3))
    ax.set_xlabel("Week"); ax.set_ylabel(r"$p_{S2}(t)$"); ax.set_ylim(0, 1); ax.set_xlim(0, 12)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), frameon=True, framealpha=0.95, edgecolor="none", fontsize=6)
    ax.set_title("(a) Scenario S2 probability over 12 weeks")
    ax = axs[1]
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8)
    sty = {"Static prior": (GREY, "o"), "Naive (w=1, duplicates)": (RED, "v"),
           "Product w, duplicates": (PURPLE, "^"), "Geo-mean w, clustered": (ORANGE, "s"),
           "Product w, clustered (CORA-OT)": (BLUE, "D")}
    for k, (c, m) in sty.items():
        rel = np.array(E3[k]["reliability"])
        ax.plot(rel[:, 0], rel[:, 1], marker=m, color=c, ms=3.5, lw=1,
                label=f"{k} (Brier {E3[k]['brier']:.3f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, fontsize=6, loc="upper left"); ax.set_title("(b) Reliability diagram (n = 6000)")
    fig.tight_layout(); fig.savefig(f"{F}/fig_evidence.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: ranking validity
def fig_ranking():
    E1 = json.load(open(f"{R}/E1_ranking.json")); E2 = json.load(open(f"{R}/E2_noise.json"))
    order = ["Matrix", "CVSS-only", "FAIR-style", "BAG-style", "CORA-OT static", "CORA-OT dynamic",
             "CORA-OT dyn. federated (3 bins)", "CORA-OT dyn. partner arcs omitted"]
    labels = ["5x5\nmatrix", "CVSS\nonly", "FAIR-\nstyle", "BAG-\nstyle", "CORA\nstatic", "CORA\ndynamic",
              "CORA\nfeder-\nated", "CORA\npartner\nomitted"]
    cols = [GREY, GREY, ORANGE, ORANGE, BLUE, BLUE, PURPLE, PURPLE]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    bp = ax.boxplot([E1["raw"][m] for m in order], tick_labels=labels, showfliers=False, patch_artist=True,
                    medianprops=dict(color="k"), widths=0.6)
    for b, c in zip(bp["boxes"], cols): b.set_facecolor(c); b.set_alpha(0.45)
    means = [E1["summary"][m]["tau"][0] for m in order]
    ax.plot(range(1, 9), means, "wD", mec="k", ms=4, label="mean")
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel(r"Kendall's $\tau$ vs. ground truth")
    ax.tick_params(axis="x", labelsize=6.5); ax.legend(frameon=False, loc="lower right")
    ax.set_title("(a) Ranking validity, 300 synthetic instances")
    ax = axs[1]
    sig = sorted(E2.keys(), key=float)
    for m, c, ls in [("Matrix", GREY, ":"), ("CVSS-only", GREY, "--"), ("FAIR-style", ORANGE, "-."),
                     ("BAG-style", ORANGE, ":"), ("CORA-OT static", BLUE, "--"), ("CORA-OT dynamic", BLUE, "-")]:
        ax.plot([float(s) for s in sig], [E2[s][m] for s in sig], color=c, ls=ls, marker="o", ms=3, label=m)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"Elicitation noise $\sigma$"); ax.set_ylabel(r"Mean Kendall's $\tau$")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="center left", bbox_to_anchor=(0.0, 0.42))
    ax.set_title("(b) Sensitivity to elicitation noise")
    fig.tight_layout(); fig.savefig(f"{F}/fig_ranking.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: control selection + scalability
def fig_controls():
    fr = W["frontier"]; E4 = json.load(open(f"{R}/E4_scalability.json"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axs[0]
    B = [f["B"] for f in fr]
    ax.plot(B, [f["exact"] for f in fr], "-o", color=BLUE, ms=3, label="Exact (Eqs. 14-16)")
    ax.plot(B, [f["greedy"] for f in fr], "--s", color=ORANGE, ms=3, label="Greedy + 1-swap")
    cvf = [f["cvss"] for f in fr]
    ax.plot(B, cvf, ":", color=GREY, label="CVSS-first (budget only)")
    for f in fr:
        ax.plot(f["B"], f["cvss"], "x" if not f["cvss_feasible"] else "o", color=RED if not f["cvss_feasible"] else GREY, ms=5 if not f["cvss_feasible"] else 3)
    ax.plot([], [], "x", color=RED, label="violates downtime limit")
    inh = W["controls"]["inherent"]
    ax.axhline(inh, color="k", lw=0.5, ls="--")
    ax.text(0.98, inh + 0.4, "inherent risk", fontsize=6.5, ha="right", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_ylim(top=inh + 3.2)
    ax.set_xlabel("Budget B (GBP k/yr)"); ax.set_ylabel("Residual risk (points/yr)")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right", bbox_to_anchor=(1.02, 0.84)); ax.set_title("(a) Residual risk vs. budget")
    ax = axs[1]
    pr = E4["propagation"]
    ax.loglog([p["nodes"] for p in pr], [p["seconds"] for p in pr], "-o", color=BLUE, ms=3)
    ax.set_xlabel("Graph nodes (3 arcs/node)"); ax.set_ylabel("Seconds"); ax.set_title("(b) Propagation, Eq. (6)")
    ax = axs[2]
    op = E4["optimisation"]; gl = E4["greedy_large"]
    ax.semilogy([o["m"] for o in op], [o["exact_s"] for o in op], "-o", color=BLUE, ms=3, label="Exact enumeration")
    ax.semilogy([o["m"] for o in op] + [g["m"] for g in gl], [o["greedy_s"] for o in op] + [g["greedy_s"] for g in gl],
                "--s", color=ORANGE, ms=3, label="Greedy")
    ax.set_xscale("log"); ax.set_xlabel("Candidate controls m"); ax.set_ylabel("Seconds")
    ax.legend(frameon=False, fontsize=6.5); ax.set_title("(c) Control selection")
    fig.tight_layout(); fig.savefig(f"{F}/fig_controls.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: LLM gate
def fig_llm():
    E5 = json.load(open(f"{R}/E5_llm_gate.json"))
    cats = list(E5.keys()); vals = [E5[c] * 100 for c in cats]
    cols = [GREEN if c == "Correct" else (BLUE if v >= 99 else (ORANGE if v > 0 else RED)) for c, v in zip(cats, vals)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(cats))[::-1]
    ax.barh(y, vals, color=cols)
    for yy, v, c in zip(y, vals, cats):
        ax.text(v + 1, yy, f"{v:.0f}%" + ("  (false-positive rate)" if c == "Correct" else ""), va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(cats); ax.set_xlim(0, 125)
    ax.set_xlabel("Outputs flagged by the deterministic gate (%), 100 synthetic outputs per category")
    fig.tight_layout(); fig.savefig(f"{F}/fig_llm_gate.pdf"); plt.close(fig)

if __name__ == "__main__":
    fig_graph(); fig_case_ranking(); fig_evidence(); fig_ranking(); fig_controls(); fig_llm()
    print("ok")
