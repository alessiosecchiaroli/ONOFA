import numpy as np

def normres(U, V, Thr, b=1, eps=0.1):
    """
    Normalized fluctuation residual detection with correction.

    Simply translated the original code from J. Westerweel and F. Scarano
    From Matlab into python.
    Check the paper for more info.

    DOI 10.1007/s00348-005-0016-6
   

    Parameters
    ----------
    U, V : 2D numpy arrays
        Displacement vector field components (shape J x I).
    Thr : float
        Detection threshold.
    b : int, optional
        Data-point neighborhood radius (default=1).
    eps : float, optional
        Estimated noise level (default=0.1).

    Returns
    -------
    Info1 : 2D boolean numpy array
        Outlier indicator (True = outlier).
    U_new, V_new : 2D numpy arrays
        Corrected displacement fields after median test.
    """
    J, I = U.shape

    # Initialise arrays
    NormFluct = np.zeros((J, I, 2))
    U_new = U.copy()
    V_new = V.copy()

    # Loop over the two velocity components
    for c, VelComp in enumerate([U, V]):
        # Loop over all data-points (excluding border)
        for i in range(b, I - b):
            for j in range(b, J - b):
                # Data neighborhood including center point
                Neigh = VelComp[j - b:j + b + 1, i - b:i + b + 1]
                NeighCol = Neigh.flatten()

                # Exclude center point
                center_idx = (2 * b + 1) ** 2 // 2
                NeighCol2 = np.delete(NeighCol, center_idx)

                # Median of neighborhood
                Median = np.median(NeighCol2)

                # Fluctuation w.r.t median
                Fluct = VelComp[j, i] - Median

                # Residual: neighborhood fluctuation w.r.t. median
                Res = NeighCol2 - Median

                # Median (absolute) value of residual
                MedianRes = np.median(np.abs(Res))

                # Normalised fluctuation
                NormFluct[j, i, c] = np.abs(Fluct / (MedianRes + eps))

                # --- Correction step ---
                if c == 0:  # U component
                    if NormFluct[j, i, c] > Thr:
                        U_new[j, i] = Median
                else:  # V component
                    if NormFluct[j, i, c] > Thr:
                        V_new[j, i] = Median

    # Combine fluctuations from U and V and apply detection criterion
    Info1 = np.sqrt(NormFluct[:, :, 0] ** 2 + NormFluct[:, :, 1] ** 2) > Thr

    return Info1, U_new, V_new
