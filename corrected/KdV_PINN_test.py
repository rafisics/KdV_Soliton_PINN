# Import Packages
import numpy as np
import tensorflow as tf
import deepxde as dde
from scipy.interpolate import griddata

# Define the domain
x_lower, x_upper = -20.0, 20.0
t_lower, t_upper = 0.0, 20.0  # t_lower is non-negative

# Create the 2D grid (for plotting and input)
x = np.linspace(x_lower, x_upper, 160)  # Reduced resolution
t = np.linspace(t_lower, t_upper, 250)  # Reduced resolution
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Space and time domains/geometry (for the deepxde model)
space_domain = dde.geometry.Interval(x_lower, x_upper)
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

# Physics-Informed Part
def pde(x, y):
    u = y[:, 0:1]
    # In 'jacobian', i is the output component and j is the input component
    u_t = dde.grad.jacobian(y, x, j=1)  # Derivative of u with respect to t
    u_x = dde.grad.jacobian(y, x, j=0)  # Derivative of u with respect to x
    u_xx = dde.grad.jacobian(u_x, x, j=0)  # Second derivative of u with respect to x
    u_xxx = dde.grad.jacobian(u_xx, x, j=0)  # Third derivative of u with respect to x

    # KdV equation: u_t + 6*u*u_x + u_xxx = 0
    f_u = u_t + 6 * u * u_x + u_xxx

    return f_u

# Initial and Boundary conditions using Dirichlet BC
def boun_1(x):
    T = x[:, 1:2]
    return 0.5 * (1 / np.cosh(0.5 * (x_lower - T)))**2

def boun_2(x):
    T = x[:, 1:2]
    return 0.5 * (1 / np.cosh(0.5 * (x_upper - T)))**2

def init_1(x):
    X = x[:, 0:1]
    return 0.5 * (1 / np.cosh(0.5 * X))**2  # Initial condition at t = 0

# Boundary and initial conditions
bc_1 = dde.DirichletBC(geomtime, boun_1, lambda _, on_boundary: on_boundary, component=0)
bc_2 = dde.DirichletBC(geomtime, boun_2, lambda _, on_boundary: on_boundary, component=0)
ic_1 = dde.IC(geomtime, init_1, lambda _, on_initial: on_initial, component=0)

# Create data for NN
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_1, bc_2, ic_1],
    num_domain=20000,
    num_boundary=150,
    num_initial=150,
    train_distribution="pseudo",
)

# Network architecture
net = dde.maps.FNN([2] + [25] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Compile and train with Adam optimizer
model.compile("adam", lr=1e-3, loss="MSE")
model.train(iterations=10000, display_every=1000)

# L-BFGS Optimization
dde.optimizers.config.set_LBFGS_options(
    maxcor=50,
    ftol=1.0 * np.finfo(float).eps,
    gtol=1e-08,
    maxiter=5000,
    maxfun=5000,
    maxls=50,
)
model.compile("L-BFGS")
model.train()

# Predict and compare with exact solution
prediction = model.predict(X_star, operator=None)
u = griddata(X_star, prediction[:, 0], (X, T), method="cubic")

# Exact solution for the KdV equation (soliton solution)
exact = 0.5 * (1 / np.cosh(0.5 * (X - T)))**2

# Save data and plot
np.savetxt("pred_test.txt", u)
np.savetxt("exact_test.txt", exact)
