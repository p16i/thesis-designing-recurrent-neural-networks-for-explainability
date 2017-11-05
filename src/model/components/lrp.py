import numpy as np


def z_plus_prop(X, W, R):
    V = np.maximum(0, W)

    Z = np.dot(X, V) + 1e-9
    S = R / Z
    C = np.dot(S, V.T);
    return X * C

def z_beta_prop(X, W, R, lowest=-1, highest=1):
    W, V, U = W, np.maximum(0, W), np.minimum(0, W)
    X, L, H = X, X * 0 + lowest, X * 0 + highest

    Z = np.dot(X, W) - np.dot(L, V) - np.dot(H, U) + 1e-9
    S = R / Z
    return X * np.dot(S, W.T) - L * np.dot(S, V.T) - H * np.dot(S, U.T)

def z_plus_beta_prop(X_p, W_p, X_b, W_b, R, lowest=-1, highest=1):
    V_p = np.maximum(0, W_p)
    Z_p = np.dot(X_p, V_p) + 1e-9

    W_b, V_b, U_b = W_b, np.maximum(0, W_b), np.minimum(0, W_b)
    L_b, H_b = X_b * 0 + lowest, X_b * 0 + highest
    Z_b = np.dot(X_b, W_b) - np.dot(L_b, V_b) - np.dot(H_b, U_b) + 1e-9

    Z = Z_p + Z_b

    # z-plus
    S_p = R / Z
    C_p = np.dot(S_p, V_p.T);
    R_p = X_p * C_p

    # z-beta
    S_b = R / Z
    R_b = X_b * np.dot(S_b, W_b.T) - L_b * np.dot(S_b, V_b.T) - H_b * np.dot(S_b, U_b.T)

    return R_p, R_b