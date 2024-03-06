import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Here we are in the setting of a uniform mesh of [0,1]^2 with N^2 points, and we consider
this mesh as an undirected graph with edges that link points in the x and y directions.

We study a model of heat diffusion that follows Fourier's law, and we denote k as the thermal conductivity.
Here we consider that heat can only diffuse between two points that are adjacent. This leads to the following "PDE" on the graph:
    ∂T/∂t = -κΔT

Where Δ = D - A, the discrete Laplacian, with D the degree matrix and A the adjacency matrix.
"""

def laplacian_grid(N): #Not on Torus e.g. no periodic bounduary condition 
    """
    Grid numerotation of size N=3 (Not the adjacency matrix)
    6 7 8
    3 4 5
    0 1 2
    """

    A = np.zeros((N**2,N**2))

    for i in range(0,N):
        for j in range(0,N):
            if 0 < i < N-1 and 0 < j < N-1:
                A[i*N+j, i*N+j-1] = 1
                A[i*N+j, i*N+j+1] = 1
                A[i*N+j, (i-1)*N+j] = 1
                A[i*N+j, (i+1)*N+j] = 1
            elif i == 0 and 0 < j < N-1:
                A[i*N+j, i*N+j-1] = 1
                A[i*N+j, i*N+j+1] = 1
                A[i*N+j, i*N+j+N] = 1
            elif i == N-1 and 0 < j < N-1:
                A[i*N+j, i*N+j-1] = 1
                A[i*N+j, i*N+j+1] = 1
                A[i*N+j, i*N+j-N] = 1
            elif i == 0 and j == 0:
                A[i*N+j, 1] = 1
                A[i*N+j, N] = 1
            elif i == 0 and j == N-1:
                A[i*N+j, N-2] = 1
                A[i*N+j, 2*N-1] = 1
            elif i == N-1 and j == 0:
                A[i*N+j, i*N+j+1] = 1
                A[i*N+j, i*N+j-N] = 1
            elif i == N-1 and j == N-1:
                A[i*N+j, i*N+j-1] = 1
                A[i*N+j, i*N+j-N] = 1
            elif 0 < i < N-1 and j == 0:
                A[i*N+j, (i+1)*N+j] = 1
                A[i*N+j, (i-1)*N+j] = 1
                A[i*N+j, i*N+j+1] = 1
            elif 0 < i < N-1 and j == N-1:
                A[i*N+j, (i+1)*N+j] = 1
                A[i*N+j, (i-1)*N+j] = 1
                A[i*N+j, i*N+j-1] = 1

    D = np.zeros((N**2,N**2))

    for i in range(0,N):
        for j in range(0,N):
            D[i*N+j, i*N+j] = sum([A[i*N+j, k] for k in range(0,N**2)])

    return D-A

N = 20
κ = 1.0

L = laplacian_grid(N)

eigenvalues, eigenvectors = np.linalg.eigh(L)
P = eigenvectors
#Diag = P.T @ L @ P
ker_dim = np.sum(np.abs(eigenvalues) < 1e-10)
diagonal_values = np.concatenate((np.ones(ker_dim), np.zeros(N**2 - ker_dim)))
P_diag_ker = np.diag(diagonal_values)

# This operator is an orthogonal projector, physically it assign to each initial temparute distribution T_0 its final one T_infinity
# Note that T_infinity is uniform. This is coherent with the fact that we are in a setting equivalent to a Neumann boundary condition for the continuous case.
P_ker = P @ P_diag_ker @ P.T  


T_0 = np.random.uniform(low=0.0, high=1.0, size=N**2)
c = [np.dot(T_0,eigenvectors.T[k]) for k in range(0,N**2)]


def T(t):
    global eigenvectors
    global eigenvalues
    global c
    global κ

    return np.sum(np.array([ c[j]*np.exp(-κ*eigenvalues[j]*t)*eigenvectors.T[j]  for j in range(0,N**2)]), axis=0)

"""
Animated heatmap plot
"""

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

num_frames = 1000
time = np.linspace(0,10, num_frames)
T_all = np.array([T(t) for t in time]) 


fig, ax = plt.subplots(figsize=(6, 6))

scat = ax.scatter([], [], s=100, c=[], cmap='plasma', edgecolors='k', zorder=2)

for i in range(N):
    for j in range(N):
        if i < N - 1:
            ax.plot([X[i, j], X[i + 1, j]], [Y[i, j], Y[i + 1, j]], 'k-', zorder=1)  # vertical edges
        if j < N - 1:
            ax.plot([X[i, j], X[i, j + 1]], [Y[i, j], Y[i, j + 1]], 'k-', zorder=1)  # horizontal edges

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Evolution of temperature over time')
ax.set_aspect('equal', adjustable='box')

cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(T_all.min(), T_all.max()))
sm.set_array([])
cbar = plt.colorbar(sm, label='T value', cax=cax)


def animate(frame):
    T = T_all[frame]
    colors = plt.cm.plasma((T - T.min()) / (T.max() - T.min()))
    scat.set_offsets(np.column_stack((X.flatten(), Y.flatten())))
    scat.set_array(T)  


ani = FuncAnimation(fig, animate, frames=num_frames, interval=25, blit=False)

plt.show()


"""
# Because we have neumann boundary condition, we can show that the mean temperature in space is constant

print(np.mean(T(0.5)))
print(np.mean(T(10)))
time = np.linspace(0,1, 100)
Temp = [T(t) for t in time]
#print(time)
plt.figure()
plt.plot(time,[np.mean(Temp[k]) for k in range(0,len(Temp))])
plt.show()
"""