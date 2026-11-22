"""DIR-77 共用：48 维因子空间展平 + 前向差分 Jacobian + 模最大特征对。不改装置任何部分。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, step72

SH = ((6, 2), (6, 2), (6, 2), (6, 2))

def flat(x):
    return np.concatenate([np.asarray(X, float).ravel(order="C") for X in x])

def unflat(v):
    out, i = [], 0
    for r, c in SH:
        n = r * c; out.append(np.array(v[i:i + n], float).reshape(r, c)); i += n
    return tuple(out)

def jac(x, eta, mode, g, h=1e-9, c=6.0):
    x = tuple(np.array(X, float) for X in x)
    f0 = flat(step72.step(*x, eta, mode, g, c))
    J = np.zeros((48, 48))
    for j in range(48):
        d = np.zeros(48); d[j] = h
        J[:, j] = (flat(step72.step(*unflat(flat(x) + d), eta, mode, g, c)) - f0) / h
    return J

def top_pair(J):
    w, V = np.linalg.eig(J)
    k = int(np.argmax(np.abs(w)))
    return w[k], V[:, k]
