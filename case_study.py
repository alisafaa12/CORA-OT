"""
case_study.py
Synthetic distribution-network case used in Section 8 of the revised manuscript.
All values are synthetic and illustrative. They do not describe a real operator.
"""
import numpy as np

NODES = [
    # id, label, owner, kind
    ("WEB",   "Public website",               "Operator", "IT"),
    ("MAIL",  "E-mail service",               "Operator", "IT"),
    ("AD",    "Corporate identity (AD)",      "Operator", "IT"),
    ("VPN",   "Remote-access VPN",            "Operator", "IT/OT"),
    ("JUMP",  "Jump host (DMZ)",              "Operator", "IT/OT"),
    ("HIST",  "Historian (DMZ)",              "Operator", "IT/OT"),
    ("EWS",   "Engineering workstation",      "Operator", "OT"),
    ("SCADA", "SCADA/DMS server",             "Operator", "OT"),
    ("HMI",   "Control-room HMI",             "Operator", "OT"),
    ("RTU",   "Substation gateway (61850/DNP3)", "Operator", "OT"),
    ("IED",   "Protection relays (IEDs)",     "Operator", "OT"),
    ("BILL",  "Billing/CRM",                  "Operator", "IT"),
    ("SUP",   "Supplier support portal",      "Supplier", "IT"),
    ("CLOUD", "Supplier cloud analytics",     "Supplier", "IT"),
    ("TEL",   "Telecom fibre WAN",            "Telco",    "Comms"),
    ("RAD",   "Backup radio link",            "Operator", "Comms"),
    ("F1",    "EF1 Distribution control",     "Operator", "Function"),
    ("F2",    "EF2 Network monitoring",       "Operator", "Function"),
    ("F3",    "EF3 Customer services",        "Operator", "Function"),
    ("F4",    "EF4 Water pumping (external)", "WaterCo",  "Function"),
]
IDX = {n[0]: i for i, n in enumerate(NODES)}
N = len(NODES)

# Attack layer: reachability, identity dependence, maintenance access
ATT_ARCS = [
    ("WEB", "BILL", 0.15), ("MAIL", "AD", 0.35), ("AD", "VPN", 0.60),
    ("AD", "HIST", 0.30), ("AD", "BILL", 0.50), ("VPN", "JUMP", 0.70),
    ("JUMP", "EWS", 0.60), ("EWS", "SCADA", 0.50), ("EWS", "RTU", 0.30),
    ("SCADA", "RTU", 0.60), ("RTU", "IED", 0.20), ("SUP", "VPN", 0.50),
    ("SUP", "JUMP", 0.40), ("HIST", "SCADA", 0.25), ("CLOUD", "HIST", 0.30),
    ("AD", "JUMP", 0.20),
]
# Consequence layer: service reliance, physical control, data flow, recovery
CON_ARCS = [
    ("RTU", "F1", 0.90), ("SCADA", "F1", 0.60), ("SCADA", "F2", 0.80),
    ("SCADA", "HMI", 0.80), ("HMI", "F1", 0.40), ("HIST", "F2", 0.50),
    ("CLOUD", "F2", 0.30), ("IED", "F1", 0.80), ("BILL", "F3", 0.90),
    ("WEB", "F3", 0.30), ("AD", "BILL", 0.60), ("AD", "JUMP", 0.30),
    ("AD", "MAIL", 0.70), ("TEL", "SCADA", 0.40), ("F1", "F4", 0.60),
    ("JUMP", "EWS", 0.20), ("HIST", "SCADA", 0.15),
]
# Redundancy (AND) group: substation comms lost only if fibre AND radio fail
AND_GROUPS = [(("TEL", "RAD"), "RTU", 0.85)]

DIMS = ["Safety", "Availability", "Financial", "Environmental", "Regulatory", "Societal"]
OMEGA = np.array([0.30, 0.25, 0.15, 0.10, 0.10, 0.10])

# Ordinal direct impacts 0..5 (synthetic, agreed scale)
IMPACT = {
    "WEB":  [0, 1, 1, 0, 1, 2], "MAIL": [0, 1, 1, 0, 1, 1],
    "AD":   [0, 2, 2, 0, 1, 1], "VPN":  [0, 1, 1, 0, 0, 0],
    "JUMP": [0, 1, 1, 0, 0, 0], "HIST": [0, 1, 1, 0, 1, 0],
    "EWS":  [0, 1, 1, 0, 0, 0], "SCADA":[1, 3, 2, 0, 1, 1],
    "HMI":  [1, 2, 1, 0, 0, 0], "RTU":  [1, 2, 2, 0, 1, 0],
    "IED":  [5, 3, 3, 1, 3, 2], "BILL": [0, 1, 2, 0, 3, 2],
    "SUP":  [0, 0, 1, 0, 0, 0], "CLOUD":[0, 1, 1, 0, 1, 0],
    "TEL":  [0, 1, 1, 0, 0, 0], "RAD":  [0, 0, 0, 0, 0, 0],
    "F1":   [3, 5, 4, 1, 4, 4], "F2":   [1, 3, 2, 0, 2, 1],
    "F3":   [0, 2, 3, 0, 3, 2], "F4":   [3, 4, 2, 3, 3, 5],
}

# Scenarios: entry nodes, target nodes, annual initiation rate, CVSS of key weakness
SCENARIOS = [
    # id, title, entry, targets, rate, cvss, local asset for matrix view
    ("S1", "Public website compromise (no OT path)", ["WEB"], ["WEB"], 0.60, 9.8, "WEB"),
    ("S2", "Remote engineering access misuse", ["AD"], ["RTU"], 0.30, 6.5, "VPN"),
    ("S3", "Supplier credential compromise", ["SUP"], ["RTU"], 0.25, 7.2, "SUP"),
    ("S4", "Monitoring-data manipulation", ["CLOUD"], ["HIST"], 0.30, 5.3, "HIST"),
    ("S5", "Accidental gateway configuration change", ["RTU"], ["RTU"], 0.10, 0.0, "RTU"),
    ("S6", "Ransomware on shared IT services", ["MAIL"], ["AD", "BILL"], 0.50, 8.8, "AD"),
    ("S7", "IT/OT integration component exploit", ["AD"], ["HIST", "SCADA"], 0.25, 9.1, "HIST"),
    ("S8", "Fibre WAN outage with radio failure", ["TEL", "RAD"], ["TEL", "RAD"], 0.08, 0.0, "TEL"),
]
# Severity label used by the matrix assessor for non-CVE scenarios
MATRIX_SEVERITY = {"S5": 3, "S8": 2}

# Candidate controls: cost (GBP k/yr annualised), burden nu, downtime z (h), change risk h
CONTROLS = [
    # id, name, cost, nu, z, h, {scenario: effectiveness}
    ("C1",  "MFA and conditional access on VPN", 40, 1, 0, 0.0, {"S2": .70, "S3": .40, "S6": .20, "S7": .20}),
    ("C2",  "Privileged access management on jump host", 60, 2, 2, 0.1, {"S2": .50, "S3": .50, "S7": .30}),
    ("C3",  "Brokered time-bound supplier access", 35, 1, 0, 0.0, {"S3": .70}),
    ("C4",  "Historian DMZ re-segmentation", 120, 2, 16, 0.3, {"S7": .60, "S6": .40, "S4": .30}),
    ("C5",  "Patch historian (critical CVE)", 15, 1, 8, 0.4, {"S7": .50}),
    ("C6",  "Patch website CMS", 5, 0, 0, 0.0, {"S1": .90}),
    ("C7",  "Passive OT network monitoring", 90, 1, 0, 0.0, {"S2": .30, "S3": .30, "S4": .40, "S5": .20, "S7": .30}),
    ("C8",  "Offline backups and tested recovery", 50, 1, 0, 0.0, {"S6": .60, "S8": .20, "S5": .30}),
    ("C9",  "Gateway change management and integrity check", 30, 1, 4, 0.1, {"S5": .70, "S2": .20}),
    ("C10", "Authenticated DNP3/IEC 62351 at substation", 200, 3, 24, 0.5, {"S2": .40, "S3": .40, "S5": .10}),
    ("C11", "Telecom path diversity", 150, 1, 0, 0.0, {"S8": .70}),
    ("C12", "Endpoint detection on IT estate", 45, 1, 0, 0.0, {"S6": .50, "S1": .20}),
]
MANDATORY = ["C8"]  # policy: tested offline recovery is mandatory
BUDGET, H_MAX, Z_MAX = 300.0, 0.8, 12.0   # one 12 h maintenance window
LAM_C = 0.01   # governance anchor: 1 risk point/yr is worth GBP 100k/yr
LAM_O = 0.0

def matrices(att=ATT_ARCS, con=CON_ARCS):
    Wa = np.zeros((N, N)); Wc = np.zeros((N, N))
    for a, b, w in att:
        Wa[IDX[a], IDX[b]] = w
    for a, b, w in con:
        Wc[IDX[a], IDX[b]] = w
    return Wa, Wc

def and_groups(ag=AND_GROUPS):
    return [([IDX[m] for m in members], IDX[j], w) for members, j, w in ag]

def impact_matrix():
    return np.array([IMPACT[n[0]] for n in NODES], dtype=float)

def control_arrays():
    S = [s[0] for s in SCENARIOS]
    eps = np.zeros((len(S), len(CONTROLS)))
    for j, c in enumerate(CONTROLS):
        for s, e in c[6].items():
            eps[S.index(s), j] = e
    cost = np.array([c[2] for c in CONTROLS], float)
    nu = np.array([c[3] for c in CONTROLS], float)
    z = np.array([c[4] for c in CONTROLS], float)
    h = np.array([c[5] for c in CONTROLS], float)
    mand = [i for i, c in enumerate(CONTROLS) if c[0] in MANDATORY]
    return eps, cost, nu, z, h, mand
