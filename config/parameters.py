"""This module contains the parameters for the CBR_NSGA2 test suite."""

# Lists are per noise std: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30].
# All three (cbr, mocba, sr) picked from the 3rd sweep (weighted rank HV>IGD+>GD+, 3:2:1).
# rtea (archive resamples k) picked from the rtea sweep + refinement (k=7/15/20) with the same weighted rank; ties broken by HV.
# NDTLZ1/NDTLZ3 rtea ranked on IGD+:GD+ (2:1) only -- HV can't separate configs there (ref point blown up by unconverged runs).
params = {
    "NZDT1": {
        "cbr": [0.85, 0.85, 0.7, 0.7, 0.7, 0.7],
        "mocba": [(400, 3), (250, 5), (400, 5), (400, 5), (400, 5), (400, 5)], # (Budget per generation, initial replications)
        "sr": [4, 10, 16, 16, 10, 16],
        "rtea": [5, 3, 5, 3, 3, 5],
    },
    "NZDT2": {
        "cbr": [0.7, 0.85, 0.55, 0.1, 0.4, 0.4],
        "mocba": [(250, 3), (250, 5), (400, 5), (400, 5), (600, 5), (400, 5)],
        "sr": [4, 4, 10, 10, 10, 10],
        "rtea": [3, 3, 5, 5, 3, 5],
    },
    "NZDT3": {
        "cbr": [0.7, 0.25, 0.4, 0.7, 0.7, 0.7],
        "mocba": [(250, 5), (400, 5), (400, 5), (400, 5), (400, 8), (600, 5)],
        "sr": [6, 6, 16, 16, 16, 16],
        "rtea": [3, 3, 5, 7, 5, 5],
    },
    "NZDT4": {
        "cbr": [0.1, 0.1, 0.1, 0.85, 0.25, 0.85],
        "mocba": [(20, 3), (20, 3), (20, 3), (20, 3), (20, 3), (60, 3)],
        "sr": [1, 1, 2, 1, 1, 1],
        "rtea": [1, 1, 1, 1, 2, 1],
    },
    "NZDT6": {
        "cbr": [0.25, 0.1, 0.1, 0.55, 0.1, 0.4],
        "mocba": [(20, 3), (120, 3), (250, 3), (250, 3), (250, 3), (250, 5)],
        "sr": [2, 2, 4, 6, 4, 6],
        "rtea": [1, 2, 2, 2, 2, 2],
    },
    "NDTLZ1": {
        "rtea": [1, 2, 3, 5, 5, 2],
    },
    "NDTLZ2": {
        "rtea": [2, 5, 5, 5, 10, 10],
    },
    "NDTLZ3": {
        "rtea": [1, 1, 2, 1, 2, 1],
    },
    "NDTLZ7": {
        "rtea": [2, 2, 3, 7, 5, 7],
    },
}
