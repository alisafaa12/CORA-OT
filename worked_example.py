"""
worked_example.py - deterministic worked example (Section 8.1-8.3).
Writes results/worked_example.json
"""
import json, os
import numpy as np
import cora_model as cm
import case_study as cs

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

GAMMA = 0.8
Wa, Wc = cs.matrices()
AG = cs.and_groups()
I = cs.impact_matrix()
d = cm.direct_consequence(I, cs.OMEGA, noncomp_idx=(0,))
d_comp = cm.direct_consequence(I, cs.OMEGA, noncomp_idx=())  # compensatory only

rows = []
for sid, title, entry, targets, rate, cvss, local in cs.SCENARIOS:
    e = [cs.IDX[x] for x in entry]; t = [cs.IDX[x] for x in targets]
    if sid in ("S5", "S8"):
        # non-malicious initiating events: entry already at target
        p0 = 1 - np.exp(-rate); reach = None
    else:
        p0, reach = cm.scenario_probability(rate, Wa, e, t)
    C, pi = cm.cascading_consequence(Wc, t, d, GAMMA, and_groups=AG)
    C_direct = float(sum(d[j] for j in t))
    q = np.zeros(cs.N); q[t] = 1
    C_lin = cm.linear_series(Wc, q, d, GAMMA, K=4)
    rows.append(dict(id=sid, title=title, rate=rate, cvss=cvss, p0=p0, C=C,
                     C_direct=C_direct, C_linear=C_lin,
                     R0=p0 * C, impaired={cs.NODES[j][0]: round(float(pi[j]), 3)
                                          for j in range(cs.N) if pi[j] > 0.05}))

# Baseline 5x5 matrix (local asset view)
def lik_bin(rate):
    return int(np.digitize(rate, [0.05, 0.1, 0.25, 0.5]) + 1)
for r, sc in zip(rows, cs.SCENARIOS):
    local = sc[6]
    imp = int(max(I[cs.IDX[local]])) if sc[0] not in cs.MATRIX_SEVERITY else cs.MATRIX_SEVERITY[sc[0]]
    imp = max(1, min(5, imp))
    sev = cs.MATRIX_SEVERITY.get(sc[0], None)
    lk = lik_bin(r["rate"])
    r["matrix_L"], r["matrix_I"], r["matrix_score"] = lk, imp, lk * imp

# ---------- Evidence update timeline for S2 (weeks 0..12)
p0_S2 = [r for r in rows if r["id"] == "S2"][0]["p0"]
events = [
  # week, label, cluster, r, q, age_days, half_life, alpha, strength, sign
  (2,  "TI: campaign targeting VPN credentials in energy sector", "TI-1", 0.8, 0.9, 7, 90, 0.9, "moderate", +1),
  (4,  "Crowd report CR-17 received (not yet validated)",          "CR-17", 0.7, 0.0, 0, 180, 1.0, "strong", +1),
  (5,  "CR-17 validated: MFA bypass on VPN portal (replica)",       "CR-17", 0.7, 0.9, 0, 180, 1.0, "strong", +1),
  (6,  "Duplicate crowd report CR-21 (same flaw)",                  "CR-17", 0.6, 0.9, 0, 180, 1.0, "strong", +1),
  (6,  "Duplicate crowd report CR-22 (same flaw)",                  "CR-17", 0.5, 0.9, 0, 180, 1.0, "strong", +1),
  (9,  "Assurance: MFA fix verified, retest negative",             "CTRL-1", 0.9, 1.0, 0, 365, 1.0, "strong", -1),
]
weeks = np.arange(0, 13)
def series(mode, eta, dedup):
    out = []
    for wk in weeks:
        clusters = {}
        for k, (w, lab, cl, r, q, age, hl, a, st, sg) in enumerate(events):
            if w > wk:
                continue
            delta = cm.freshness(age + 7 * (wk - w), hl)   # evidence ages over time
            c = cm.evidence_weight(r, q, delta, a, mode) * cm.ell(st, sg)
            key = cl if dedup else k                        # naive: every item independent
            clusters.setdefault(key, []).append(c)
        out.append(cm.update_probability(p0_S2, list(clusters.values()), eta))
    return out
timeline = dict(weeks=weeks.tolist(), p0=p0_S2,
                naive=series("product", 1.0, False),
                product=series("product", 0.0, True),
                geomean=series("geomean", 0.0, True),
                events=[(e[0], e[1]) for e in events])
# weights printout for the table
wtab = []
for (w, lab, cl, r, q, age, hl, a, st, sg) in events:
    delta = cm.freshness(age, hl)
    wtab.append(dict(week=w, label=lab, r=r, q=q, delta=round(delta, 3), alpha=a,
                     w_prod=round(cm.evidence_weight(r, q, delta, a), 3),
                     w_geo=round(cm.evidence_weight(r, q, delta, a, "geomean"), 3),
                     ell=round(cm.ell(st, sg), 3)))

# ---------- Control selection with week-5 probabilities (S2 raised by evidence)
p_now = np.array([r["p0"] for r in rows])
p_now[1] = timeline["product"][5]
C = np.array([r["C"] for r in rows])
eps, cost, nu, z, h, mand = cs.control_arrays()
args = (p_now, C, eps, cost, nu, h, z, cs.BUDGET, cs.H_MAX, cs.Z_MAX, mand, cs.LAM_C, cs.LAM_O)
x_opt, f_opt = cm.solve_exact(*args)
x_gr, f_gr = cm.solve_greedy(*args)
R_inh = cm.residual_risk(p_now, C, eps, np.zeros(len(cost)))

# CVSS-first heuristic: address scenarios in CVSS order, cheapest control first, budget only
order = np.argsort([-r["cvss"] for r in rows])
x_cv = np.zeros(len(cost), int)
for j in mand: x_cv[j] = 1
for s in order:
    for j in np.argsort(-eps[s]):
        if eps[s, j] > 0 and not x_cv[j] and cost @ x_cv + cost[j] <= cs.BUDGET:
            x_cv[j] = 1; break
def summarise(x):
    R = cm.residual_risk(p_now, C, eps, x)
    return dict(selected=[cs.CONTROLS[j][0] for j in range(len(x)) if x[j]],
                residual=float(R.sum()), cost=float(cost @ x), downtime=float(z @ x),
                change_risk=float(h @ x), objective=float(cm.objective(x, p_now, C, eps, cost, nu, cs.LAM_C, cs.LAM_O)),
                feasible=bool(cm.feasible(x, cost, cs.BUDGET, h, cs.H_MAX, z, cs.Z_MAX, mand)),
                per_scenario=R.round(4).tolist())
ctrl = dict(inherent=float(R_inh.sum()), inherent_per_scenario=R_inh.round(4).tolist(),
            exact=summarise(x_opt), greedy=summarise(x_gr), cvss_first=summarise(x_cv))

# budget frontier
frontier = []
for B in [50, 100, 150, 200, 250, 300, 400, 500, 700]:
    a2 = (p_now, C, eps, cost, nu, h, z, B, cs.H_MAX, cs.Z_MAX, mand, 0.0, 0.0)
    xo, _ = cm.solve_exact(*a2)
    xg, _ = cm.solve_greedy(*a2)
    # CVSS-first under the same budget
    xc = np.zeros(len(cost), int)
    for j in mand: xc[j] = 1
    for s in order:
        for j in np.argsort(-eps[s]):
            if eps[s, j] > 0 and not xc[j] and cost @ xc + cost[j] <= B:
                xc[j] = 1; break
    frontier.append(dict(B=B,
        exact=float(cm.residual_risk(p_now, C, eps, xo).sum()) if xo is not None else None,
        greedy=float(cm.residual_risk(p_now, C, eps, xg).sum()),
        cvss=float(cm.residual_risk(p_now, C, eps, xc).sum()),
        cvss_feasible=bool(cm.feasible(xc, cost, B, h, cs.H_MAX, z, cs.Z_MAX, mand))))

# budget-only optimum at B=300 (lambda_c = 0) and sensitivity of the anchor lambda_c
xb, _ = cm.solve_exact(p_now, C, eps, cost, nu, h, z, cs.BUDGET, cs.H_MAX, cs.Z_MAX, mand, 0.0, 0.0)
ctrl["exact_budget_only"] = summarise(xb)
# unconstrained-by-downtime variant to show what the downtime/safety limits remove
xu, _ = cm.solve_exact(p_now, C, eps, cost, nu, h, z, cs.BUDGET, 99, 999, mand, 0.0, 0.0)
ctrl["exact_no_ops_limits"] = summarise(xu)
lam_sens = []
for lc in [0.0, 0.005, 0.01, 0.02, 0.04]:
    xl, _ = cm.solve_exact(p_now, C, eps, cost, nu, h, z, cs.BUDGET, cs.H_MAX, cs.Z_MAX, mand, lc, 0.0)
    sm = summarise(xl); lam_sens.append(dict(lam_c=lc, selected=sm["selected"], cost=sm["cost"], residual=sm["residual"]))
ctrl["lambda_sensitivity"] = lam_sens
res = dict(d=dict(zip([n[0] for n in cs.NODES], d.round(3).tolist())),
           d_comp=dict(zip([n[0] for n in cs.NODES], d_comp.round(3).tolist())),
           scenarios=rows, timeline=timeline, weights=wtab, controls=ctrl, frontier=frontier,
           p_now=p_now.tolist())
json.dump(res, open(os.path.join(OUT, "worked_example.json"), "w"), indent=1, default=float)
if __name__ == "__main__":
    for r in rows:
        print(f"{r['id']} p0={r['p0']:.4f} Cdir={r['C_direct']:.2f} C={r['C']:.2f} Clin={r['C_linear']:.2f} R0={r['R0']:.3f} M={r['matrix_score']} cvss={r['cvss']}")
    print("d", res["d"])
    print("timeline product", np.round(timeline["product"], 4))
    print("timeline naive  ", np.round(timeline["naive"], 4))
    print("timeline geo    ", np.round(timeline["geomean"], 4))
    for k in ("exact", "greedy", "cvss_first"):
        print(k, ctrl[k])
    print("inherent", ctrl["inherent"])
    for f in frontier: print(f)
    print('budget-only', ctrl['exact_budget_only'])
    print('no ops limits', ctrl['exact_no_ops_limits'])
    for l in lam_sens: print(l)
    for w in wtab: print(w)

# ---------- Robustness: rank stability of the week-0 ranking under perturbation
def rank_stability(n=2000, seed=11):
    rng = np.random.default_rng(seed)
    ranks = np.zeros((n, len(rows)), int)
    for it in range(n):
        Wa_p = np.clip(Wa * np.exp(rng.normal(0, 0.3, Wa.shape)), 0, 0.95)
        Wc_p = np.clip(Wc * np.exp(rng.normal(0, 0.3, Wc.shape)), 0, 0.95)
        om = rng.dirichlet(cs.OMEGA * 40)
        ratio = rng.uniform(3, 8)
        gam = rng.uniform(0.6, 1.0)
        dd = cm.direct_consequence(I, om, (0,), ratio=ratio)
        R = []
        for sid, title, entry, targets, rate, cvss, local in cs.SCENARIOS:
            e = [cs.IDX[x] for x in entry]; t = [cs.IDX[x] for x in targets]
            rt = rate * np.exp(rng.normal(0, 0.3))
            p0 = 1 - np.exp(-rt) if sid in ("S5", "S8") else cm.scenario_probability(rt, Wa_p, e, t)[0]
            R.append(p0 * cm.cascading_consequence(Wc_p, t, dd, gam, and_groups=AG)[0])
        ranks[it] = np.argsort(np.argsort(-np.array(R))) + 1
    return ranks
rk = rank_stability()
res["rank_stability"] = dict(median=np.median(rk, 0).tolist(), p05=np.percentile(rk, 5, 0).tolist(),
                             p95=np.percentile(rk, 95, 0).tolist(),
                             top3_freq=(rk <= 3).mean(0).tolist())
np.save(os.path.join(OUT, "rank_samples.npy"), rk)
json.dump(res, open(os.path.join(OUT, "worked_example.json"), "w"), indent=1, default=float)
if __name__ == "__main__":
    print("rank stability", res["rank_stability"])
