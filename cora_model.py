"""
cora_model.py
Reference implementation of the CORA-OT formal model (revised Equations).
All functions are deterministic and use only numpy.

Notation follows the revised manuscript:
  W_att  : attack-layer arc weights  (conditional step-success probability)
  W_con  : consequence-layer arc weights (conditional impairment probability)
  and_groups : redundancy groups; target j is impaired through the group only
               if every member of the group is impaired.
"""
import numpy as np
from itertools import product

# ---------------------------------------------------------------- Eq. (3)
def evidence_weight(r, q, delta, alpha, mode="product"):
    """Evidence weight w_e. 'product' = joint validity (default),
    'geomean' = sensitivity alternative."""
    f = np.array([r, q, delta, alpha], dtype=float)
    if mode == "product":
        return float(np.prod(f))
    if mode == "geomean":
        return float(np.prod(f) ** 0.25)
    raise ValueError(mode)

def freshness(age_days, half_life_days):
    """delta_e = 2^(-age/h)."""
    return 2.0 ** (-age_days / half_life_days)

# Calibrated ordinal anchors for log-likelihood contributions (ell_e = ln LR)
LR_ANCHORS = {"weak": 2.0, "moderate": 5.0, "strong": 10.0, "very_strong": 20.0}

def ell(strength, sign=+1):
    return sign * np.log(LR_ANCHORS[strength])

# ---------------------------------------------------------------- Eq. (4)
def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cluster_contribution(contribs, eta=0.0):
    """Combine signed contributions w_e*ell_e that refer to the same claim
    (same weakness and same underlying observation). The strongest item counts
    fully; the others count with corroboration factor eta (eta=0: duplicates
    add nothing; eta=1: full independence)."""
    if len(contribs) == 0:
        return 0.0
    c = np.asarray(contribs, dtype=float)
    k = np.argmax(np.abs(c))
    return float(c[k] + eta * (c.sum() - c[k]))

def update_probability(p0, clusters, eta=0.0):
    """clusters: list of lists of signed contributions w_e*ell_e."""
    z = logit(p0) + sum(cluster_contribution(c, eta) for c in clusters)
    return float(sigmoid(z))

# ---------------------------------------------------------------- Eq. (5)
def value(level, ratio=5.0):
    """Governance value function v(.) mapping ordinal level 0..5 to a ratio
    scale with v(5)=1 and a constant ratio between adjacent levels; v(0)=0."""
    level = np.asarray(level, dtype=float)
    return np.where(level <= 0, 0.0, ratio ** (level - 5.0))

def direct_consequence(I, omega, noncomp_idx=(0,), ratio=5.0, scale=100.0):
    """Revised Eq. (5): additive value model with a non-compensatory floor.
    I: (n_nodes, D) ordinal impacts 0..5; omega sums to 1.
    d = scale * max( sum_k omega_k v(I_k), max_{k in N} v(I_k) )."""
    V = value(I, ratio)
    comp = V @ np.asarray(omega, dtype=float)
    floor = V[:, list(noncomp_idx)].max(axis=1) if len(noncomp_idx) else 0.0
    return scale * np.maximum(comp, floor)

# ---------------------------------------------------------------- Eq. (6)
def propagate(W, pi0, gamma=1.0, K=None, and_groups=None, tol=1e-10):
    """Noisy-OR (independent-cascade, mean-field) propagation.
    pi_j <- 1 - (1-pi0_j) * prod_i (1 - gamma*W_ij*pi_i) * prod_G (1 - gamma*W_Gj*prod_{i in G} pi_i)
    W: (n,n) with W[i,j] weight of arc i->j. and_groups: list of (members, j, w).
    Returns impairment probabilities pi in [0,1]."""
    n = W.shape[0]
    K = n if K is None else K
    pi = np.array(pi0, dtype=float)
    for _ in range(K):
        surv = np.prod(1.0 - gamma * W * pi[:, None], axis=0)
        if and_groups:
            for members, j, w in and_groups:
                surv[j] *= 1.0 - gamma * w * np.prod(pi[list(members)])
        new = 1.0 - (1.0 - np.asarray(pi0)) * surv
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    return pi

def linear_series(W, q, d, gamma, K):
    """Original Eq. (6) (truncated walk sum) kept for comparison."""
    M = np.eye(W.shape[0])
    Pk = np.eye(W.shape[0])
    for h in range(1, K + 1):
        Pk = Pk @ W
        M = M + (gamma ** h) * Pk
    return float(q @ M @ d)

def scenario_probability(rate, W_att, entry, targets, K=None):
    """Baseline annual probability p_s^0 = 1 - exp(-rate * rho_target)."""
    pi0 = np.zeros(W_att.shape[0]); pi0[list(entry)] = 1.0
    reach = propagate(W_att, pi0, 1.0, K)
    rho = 1.0 - np.prod(1.0 - reach[list(targets)])
    return 1.0 - np.exp(-rate * rho), reach

def cascading_consequence(W_con, targets, d, gamma=0.8, K=None, and_groups=None):
    pi0 = np.zeros(W_con.shape[0]); pi0[list(targets)] = 1.0
    pi = propagate(W_con, pi0, gamma, K, and_groups)
    return float(pi @ d), pi

# ---------------------------------------------------------------- Eq. (7)-(8)
def combined_effectiveness(eps_s, x):
    return 1.0 - np.prod(1.0 - np.asarray(eps_s) * np.asarray(x))

def residual_risk(p, C, eps, x):
    """eps: (S, m). Returns central residual risk vector (S,)."""
    surv = np.prod(1.0 - eps * np.asarray(x)[None, :], axis=1)
    return np.asarray(p) * np.asarray(C) * surv

def precautionary(R, u, lam_u):
    return np.asarray(R) * (1.0 + lam_u * np.asarray(u))

# ---------------------------------------------------------------- Eq. (9)-(12)
def objective(x, p, C, eps, cost, nu, lam_c, lam_o):
    return residual_risk(p, C, eps, x).sum() + lam_c * cost @ x + lam_o * nu @ x

def feasible(x, cost, B, h, H, z, Z, mandatory):
    x = np.asarray(x)
    if any(x[j] == 0 for j in mandatory):
        return False
    return (cost @ x <= B + 1e-9) and (h @ x <= H + 1e-9) and (z @ x <= Z + 1e-9)

def solve_exact(p, C, eps, cost, nu, h, z, B, H, Z, mandatory, lam_c, lam_o):
    m = len(cost)
    best, bx = np.inf, None
    for bits in product((0, 1), repeat=m):
        x = np.array(bits)
        if not feasible(x, cost, B, h, H, z, Z, mandatory):
            continue
        val = objective(x, p, C, eps, cost, nu, lam_c, lam_o)
        if val < best:
            best, bx = val, x
    return bx, best

def solve_greedy(p, C, eps, cost, nu, h, z, B, H, Z, mandatory, lam_c, lam_o,
                 local_search=True):
    """Greedy by marginal objective reduction per unit cost, then 1-swap search."""
    m = len(cost)
    x = np.zeros(m, dtype=int)
    for j in mandatory:
        x[j] = 1
    cur = objective(x, p, C, eps, cost, nu, lam_c, lam_o)
    while True:
        best_gain, bj = 0.0, None
        for j in range(m):
            if x[j]:
                continue
            x[j] = 1
            if feasible(x, cost, B, h, H, z, Z, mandatory):
                val = objective(x, p, C, eps, cost, nu, lam_c, lam_o)
                gain = (cur - val) / max(cost[j], 1e-9)
                if gain > best_gain:
                    best_gain, bj = gain, j
            x[j] = 0
        if bj is None:
            break
        x[bj] = 1
        cur = objective(x, p, C, eps, cost, nu, lam_c, lam_o)
    if local_search:
        improved = True
        while improved:
            improved = False
            for a in range(m):
                for b in range(m):
                    if x[a] == 1 and x[b] == 0 and a not in mandatory:
                        y = x.copy(); y[a] = 0; y[b] = 1
                        if feasible(y, cost, B, h, H, z, Z, mandatory):
                            v = objective(y, p, C, eps, cost, nu, lam_c, lam_o)
                            if v < cur - 1e-12:
                                x, cur, improved = y, v, True
                for b in range(m):
                    if x[b] == 0:
                        y = x.copy(); y[b] = 1
                        if feasible(y, cost, B, h, H, z, Z, mandatory):
                            v = objective(y, p, C, eps, cost, nu, lam_c, lam_o)
                            if v < cur - 1e-12:
                                x, cur, improved = y, v, True
    return x, cur
