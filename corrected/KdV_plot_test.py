# Import Packages
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
import matplotlib.transforms as mtransforms

# Set font and style
fn = [12]
plt.rcParams.update({'font.size': fn[0], 'text.color': 'black', 'font.weight': 'bold'})
csfont = {'fontname': 'serif'}
hfont = {'fontname': 'serif'}
plt.rcParams['text.usetex'] = True

# Load Data
u = np.loadtxt("pred_test.txt")
ext = np.loadtxt("exact_test.txt")

# Create the 2D grid (for plotting and input)
x_lower, x_upper = -20.0, 20.0  # Match the data generation domain
t_lower, t_upper = 0.0, 20.0  # Match the data generation domain
x_res = 160  # Match the data generation resolution
t_res = 250  # Match the data generation resolution

# n_t, n_x = u.shape
# x = np.linspace(x_lower, x_upper, n_x)
# t = np.linspace(t_lower, t_upper, n_t)
x = np.linspace(x_lower, x_upper, x_res)
t = np.linspace(t_lower, t_upper, t_res)
X, T = np.meshgrid(x, t)

# Calculate error
err = np.abs(u - ext)**2
rel_err = np.abs(u - ext)/ext

# Create the figure
fig, ax = plt.subplots(figsize=(6, 10))
ax.axis('off')

cmap_name = "cool"

####### Row 0: Predicted Solution ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=0.8, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(u.T, interpolation='nearest', cmap=cmap_name,
              extent=[t_lower, t_upper, x_lower, x_upper],  # Adjusted time domain
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.set_title(r'Predicted $\psi(x,t)$')
ax.set_ylabel(r'$x$')
# ax.set_xlabel(r'$t$')

####### Row 1: Exact Solution ##################
gs1 = gridspec.GridSpec(1, 2)
gs1.update(top=0.72, bottom=0.58, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs1[:, :])

h = ax.imshow(ext.T, interpolation='nearest', cmap=cmap_name,
              extent=[t_lower, t_upper, x_lower, x_upper],  # Adjusted time domain
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_title(r'Exact $\psi(x,t)$')
ax.set_ylabel(r'$x$')
# ax.set_xlabel(r'$t$')

####### Row 2: Error ##################
gs2 = gridspec.GridSpec(1, 2)
gs2.update(top=0.5, bottom=0.36, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs2[:, :])

h = ax.imshow(err.T, interpolation='nearest', cmap=cmap_name,
              extent=[t_lower, t_upper, x_lower, x_upper],  # Adjusted time domain
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_title(r'Error $|\psi_{\rm{ex}}(x,t) - \psi_{\rm{pr}}(x,t)|^2$')
# ax.set_title(r'Relative error $|\psi_{\rm{ex}}(x,t) - \psi_{\rm{pr}}(x,t)|/\psi_{\rm{ex}}(x,t)$')
ax.set_ylabel(r'$x$')
# ax.set_xlabel(r'$t$')

####### Row 3: Line Plots ##################
gs3 = gridspec.GridSpec(1, 3)
gs3.update(top=0.28, bottom=0.13, left=0.13, right=0.9, wspace=0.5)

# Define the time points and corresponding indices
fractions = [0.25, 0.5, 0.75]

# Compute indices for desired time points
time_points = [f * t_upper for f in fractions]  # Adjusted for t = [0, 200]
indices = [int(round(t * (t_res -1) / t_upper)) for t in time_points]  # Map t to index: i = t * 1023 / 200


# Loop through the time points and plot
for i, (t_val, idx) in enumerate(zip(time_points, indices)):
    ax = plt.subplot(gs3[0, i])
    ax.plot(x, ext[idx], c='cyan', ls='-', linewidth=3, label='Exact')
    ax.plot(x, u[idx], c="purple", ls='--', linewidth=3, label='Prediction')
    ax.set_ylabel(r'$\psi(x,t)$')
    ax.set_xlabel(r'$x$')
    ax.set_title(f'$t = {t_val}$')

# Add a single legend for all subplots of the loop
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=5, frameon=True)

# Save and show the figure
plt.savefig("KdV_Soliton_2D_test.jpg", dpi=600)
plt.show()

####### 3D Plots ##################
fig = plt.figure(figsize=(13, 8), constrained_layout=True)

# Exact Solution 3D Plot
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X, T, ext, linewidth=0, cmap=cmap_name, antialiased=False, alpha=0.6, rcount=200, ccount=200)
ax.view_init(20, 70)
ax.set_xticks([x_lower, 0, x_upper])
ax.set_xlabel(r'$x$')
ax.set_xlim(x_lower, x_upper)
ax.set_yticks([0, t_upper/2, t_upper])
ax.set_ylabel(r'$t$')
ax.set_zticks([0.00, 0.06, 0.12])
# disable auto rotation
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'$\psi_{\rm{ex}}(x,t)$', rotation=90)
ax.set_title(r'Exact $\psi(x,t)$')

# Predicted Solution 3D Plot
ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(X, T, u, linewidth=0, cmap=cmap_name, antialiased=False, alpha=0.6, rcount=200, ccount=200)
ax.view_init(20, 70)
ax.set_xticks([x_lower, 0, x_upper])
ax.set_xlabel(r'$x$')
ax.set_xlim(x_lower, x_upper)
ax.set_yticks([0, t_upper/2, t_upper])
ax.set_ylabel(r'$t$')
ax.set_zticks([0.00, 0.06, 0.12])
# disable auto rotation
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'$\psi_{\rm{pr}}(x,t)$', rotation=90)
ax.set_title(r'Predicted $\psi(x,t)$')

# Add colorbar
fig.colorbar(surf, shrink=0.5, aspect=8)

# Save and show the figure
plt.savefig("KdV_Soliton_3D_test.jpg", dpi=600)
plt.show()
