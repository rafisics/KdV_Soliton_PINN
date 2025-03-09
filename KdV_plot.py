'''
Program to plot solitary wave in Korteweg-de Vries(KdV)
equation using Physics-Informed Neural Network(PINN)

R. Anand, M.Sc Physics, Department of Physics, 
Bharathidasan University, Trichy

Email : anandphy0@gmail.com
Date : 06/03/2024
'''



# Import Packages

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
import matplotlib.transforms as mtransforms
fn = [12]
plt.rcParams.update({'font.size': fn[0], 'text.color': 'black', 'font.weight': 'bold'})
csfont = {'fontname':'serif'}
hfont = {'fontname':'serif'}
plt.rcParams['text.usetex'] = True

# Load Data
u = np.loadtxt("pred.txt")
ext = np.loadtxt("exact.txt")
x = np.linspace(-20, 20, 128)  # Match spatial dimension
t = np.linspace(-20, 20, 200)  # Match time dimension
X, T = np.meshgrid(x, t)       # Create meshgrid for x and t

# Compute error
err = (np.abs(u - ext))**2

# Create figure
fig, ax = plt.subplots(figsize=(6, 10))
ax.axis('off')

cmap_name = "cool"

# ====================== Row 0: Predicted ======================

gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=0.8, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(u.T, interpolation='nearest', cmap=cmap_name,
              extent=[-20, 20, -20, 20], origin='lower', aspect='auto')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_title(r'$Pred\;|\psi(x,t)|$')
ax.set_ylabel(r'$x$')

# ====================== Row 1: Exact ======================

gs1 = gridspec.GridSpec(1, 2)
gs1.update(top=0.72, bottom=0.58, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs1[:, :])

h = ax.imshow(ext.T, interpolation='nearest', cmap=cmap_name,
              extent=[-20, 20, -20, 20], origin='lower', aspect='auto')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_title(r'$Exact\;|\psi(x,t)|$')
ax.set_ylabel(r'$x$')

# ====================== Row 2: Error ======================

gs2 = gridspec.GridSpec(1, 2)
gs2.update(top=0.5, bottom=0.36, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs2[:, :])

h = ax.imshow(err.T, interpolation='nearest', cmap=cmap_name,
              extent=[-20, 20, -20, 20], origin='lower', aspect='auto')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_title(r'$({|\psi(x,t)|_{Exact}-|\psi(x,t)|_{Pre}})^2$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')

# ====================== Row 3: Cross-sections ======================

gs3 = gridspec.GridSpec(1, 3)
gs3.update(top=0.28, bottom=0.13, left=0.13, right=0.9, wspace=0.5)

for i, idx in enumerate([100, 150, 199]):  # Make sure idx is within bounds (0 to 199)
    ax = plt.subplot(gs3[0, i])
    ax.plot(x, ext[idx], c='cyan', ls='-', linewidth=3, label='Exact')
    ax.plot(x, u[idx], c='purple', ls='--', linewidth=3, label='Prediction')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=5, frameon=True)

# plt.savefig("KdV_Soliton.jpg", dpi = 600)
plt.show()

# ====================== 3D Surface Plots ======================

fig = plt.figure(figsize=(13, 8), constrained_layout=True)

# ----------- Exact 3D Surface -----------
ax = fig.add_subplot(121, projection='3d')
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor(ax.yaxis.pane.set_edgecolor(ax.zaxis.pane.set_edgecolor('w')))
ax.grid(False)

# Use ext directly without transposing, as X and T are already transposed correctly
surf = ax.plot_surface(X, T, ext, linewidth=0, cmap=cmap_name, antialiased=False, alpha=0.6, rcount=200, ccount=200)

ax.view_init(20, 70)
ax.set_xticks([-20, 0, 20])
ax.set_xlim(-20, 20)
ax.set_yticks([-20, 0, 20])
ax.set_zticks([0.00, 0.06, 0.12])

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$\psi(x, t)$', rotation=90)
ax.set_title(r'$Exact\;|\psi(x,t)|$')

# ----------- Prediction 3D Surface -----------
ax = fig.add_subplot(122, projection='3d')
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor(ax.yaxis.pane.set_edgecolor(ax.zaxis.pane.set_edgecolor('w')))
ax.grid(False)

# Use u directly without transposing, as X and T are already transposed correctly
surf = ax.plot_surface(X, T, u, linewidth=0, cmap=cmap_name, antialiased=False, alpha=0.6, rcount=200, ccount=200)

ax.view_init(20, 70)
ax.set_xticks([-20, 0, 20])
ax.set_xlim(-20, 20)
ax.set_yticks([-20, 0, 20])
ax.set_zticks([0.00, 0.06, 0.12])

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_title(r'$Pred\;|\psi(x,t)|$')

fig.colorbar(surf, shrink=0.5, aspect=8)

# Save figure
# plt.savefig("KdV_Soliton_3d.jpg", dpi=600)
plt.show()
