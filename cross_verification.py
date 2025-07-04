import numpy as np

def match_score(A, B):
    A_rounded = np.round(A).astype(int)
    B_rounded = np.round(B).astype(int)

    conditions = {
        "dx0 == dx1": (A_rounded[0, 0] - B_rounded[0, 0]) == (A_rounded[1, 0] - B_rounded[1, 0]),
        "dy0 == dy1": (A_rounded[0, 1] - B_rounded[0, 1]) == (A_rounded[0, 1] - B_rounded[1, 1]),
        "z0 match":   A_rounded[0, 2] == B_rounded[0, 2],
        "z1 match":   A_rounded[1, 2] == B_rounded[1, 2],
    }

    score = sum(conditions.values())  # Count of True values

    return conditions, score
