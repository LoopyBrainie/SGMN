import numpy as np
def jac_scalar(c, lam, r=1, d=1, Pmat=None):
    D, R = d, r
    P = np.eye(D)[:R, :] if Pmat is None else Pmat
    P = P @ P.T if P.shape[0] != P.shape[1] else P
    Ws = P / (c + lam)
    U0, s, Vt = np.linalg.svd(Ws)
    U = U0[:, :R] * np.sqrt(s[:R]); V = Vt[:R, :].T * np.sqrt(s[:R])
    keys = [np.sqrt(c) * np.eye(D)[i] for i in range(D)]
    C = sum(np.outer(k, k) for k in keys) / D
    B = sum(np.outer(P @ k, k) for k in keys) / D
    n = 2 * D * R; h = 1e-6; A = np.eye(n)
    for j in range(n):
        dU = np.zeros((D, R)); dV = np.zeros((D, R))
        if j < D * R: dU.flat[j] = h
        else: dV.flat[j - D * R] = h
        outs = []
        for sg in (+1, -1):
            Ux, Vx = U + sg * dU, V + sg * dV
            Gx = (Ux @ Vx.T @ C - B) + lam * (Ux @ Vx.T)
            outs.append(np.concatenate([(Ux - (Gx @ Vx)).ravel(), (Vx - (Gx.T @ Ux)).ravel()]))
        d_ = (outs[0] - outs[1]) / (2 * h)
        A[:, j] = np.eye(n)[j] - d_
    return A, U, V, Ws
for (d, r, c, lam) in ((1,1,1.0,0.0), (1,1,1.0,0.05), (2,2,1.0,0.05)):
    A, U, V, Ws = jac_scalar(c, lam, r, d)
    print("d=%d r=%d c=%.2f λ=%.2f  eig(A)=%s" % (d, r, c, lam,
          np.round(np.sort_complex(np.linalg.eigvals(A)), 6)))
# 手推标量：A=[[c+λ, c+λ],[c+λ, c+λ]] ⇒ 0, 2(c+λ)
