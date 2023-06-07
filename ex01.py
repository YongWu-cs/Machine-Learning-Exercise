import numpy as np
from matplotlib import pyplot as plt
from numpy import radians
from scipy.sparse import coo_matrix


def construct_X(M, alphas, Np=None):
    D = M * M  # Tomogram size
    No = len(alphas)  # Number of measurement angles

    if Np is None:
        Np = int(np.floor(np.sqrt(2) * M))  # Default sensor resolution

    sensor_center = (Np - 1) // 2  # Center of sensor array

    # Create coordinate grid for tomogram pixels
    ja, jb = np.mgrid[0:M, 0:M]
    # j = ja + M * jb

    # Create coordinate grid for sensor elements
    ip, io = np.mgrid[0:Np, 0:No]

    alpha_rad = radians(alphas)
    n = np.array([np.cos(alpha_rad), np.sin(alpha_rad)])

    # Compute pixel centers and projection on the sensor
    C = np.array([jb - (M - 1) // 2, (M - 1) // 2 - ja], dtype=np.float64)
    p = np.dot(n.T, C.reshape(2, D)) + sensor_center

    # Find the nearest sensor elements influenced by each pixel
    p_floor = np.floor(p).astype(int)
    p_floor = p_floor.T
    p_ceil = p_floor + 1

    # Calculate weights for the sensor elements
    weights_floor = (p_ceil - p.T)
    weights_ceil = (p.T - p_floor)

    # Create lists for sparse matrix construction
    rows = []
    cols = []
    data = []

    for io_idx, alpha in enumerate(alphas):
        for j_idx in range(D):
            if np.any(weights_floor[j_idx]):
                rows.append(p_floor[j_idx][io_idx] + Np * io_idx)
                cols.append(j_idx)
                data.append(weights_floor[j_idx][io_idx])

            if np.any(weights_ceil[j_idx]):
                if p_ceil[j_idx][io_idx] != Np:
                    rows.append(p_ceil[j_idx][io_idx] + Np * io_idx)
                    cols.append(j_idx)
                    data.append(weights_ceil[j_idx][io_idx])

    # Create sparse matrix X
    X = coo_matrix((data, (rows, cols)), shape=(Np * No, D)).tocsc().toarray()

    return X


res = construct_X(10, [-33, 1, 42])
print(res)
im = plt.imshow(res)
plt.colorbar(im, orientation='horizontal')
plt.show()
