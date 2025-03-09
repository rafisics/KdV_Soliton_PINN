'''
Program to predict solitary wave in Korteweg-de Vries(KdV)
equation using Physics-Informed Neural Network(PINN)

R. Anand, M.Sc Physics, Department of Physics, 
Bharathidasan University, Trichy

Email : anandphy0@gamil.com
Date : 06/03/2024
'''




# Import Packages
import numpy as np
import tensorflow as tf
import deepxde as dde
from scipy.interpolate import griddata

# Define the domain
x_lower, x_upper = -20.0, 20.0
t_lower, t_upper = -20.0, 20.0

# Create the 2D grid (for plotting and input)
x = np.linspace(x_lower, x_upper, 128)  # Reduced resolution
t = np.linspace(t_lower, t_upper, 200)  # Reduced resolution
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
	u_t = dde.grad.jacobian(y, x, j=1)
	u_x = dde.grad.jacobian(y, x, j=0)
	u_xx = dde.grad.jacobian(u_x, x, j=0)
	u_xxx = dde.grad.jacobian(u_xx, x, j=0)

	f_u = u_t + 6*u*u_x + u_xxx

	return f_u

#Initial and Boundary conditions using Dirichlet BC

def boun_1(x):
	T = x[:,1:2]
	
	return 0.125*(1/np.cosh(4.5 +0.0625*T))**2

def boun_2(x):
	T = x[:,1:2]

	return 0.125*(1/np.cosh(5.5 -0.0625*T))**2

def init_1(x):
	X = x[:,0:1]
	return 0.125*(1/np.cosh(1.75 +0.25*X))**2

bc_1 = dde.DirichletBC(geomtime, boun_1, lambda _, 
			on_boundary: on_boundary, component=0)
bc_2 = dde.DirichletBC(geomtime, boun_2, lambda _, 
			on_boundary: on_boundary, component=0)

ic_1 = dde.IC(geomtime, init_1, lambda _, 
			on_initial: on_initial,component=0)

# Create data for NN

data = dde.data.TimePDE(
	geomtime,
	pde,
	[bc_1, bc_2, ic_1],
	num_domain=10000,
	num_boundary=100,
	num_initial=100,
	train_distribution="pseudo",
)

#Network architecture

net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss="MSE")

model.train(epochs=5000, display_every=500)

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

prediction = model.predict(X_star, operator=None)
u = griddata(X_star, prediction[:, 0], (X, T), method="cubic")

k1 = 0.5
c = 1
exact = k1/2 * (1/np.cosh(1/2 * (k1 * X + 0.5 * T)) + c)**2


# Save data and plot
np.savetxt("pred.txt", u)
np.savetxt("exact.txt", exact)
