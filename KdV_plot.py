'''
Program to plot solitary wave in Korteweg-de Vries(KdV)
equation using Physics-Informed Neural Network(PINN)

R. Anand, M.Sc Physics, Department of Physics, 
Bharathidasan University, Trichy

Email : anandphy0@gamil.com
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
plt.rcParams.update({'font.size': fn[0],'text.color':'black','font.weight':'bold'})
csfont = {'fontname':'serif'}
hfont = {'fontname':'serif'}
plt.rcParams['text.usetex'] = True

#Load Data 
u = np.loadtxt("pred.txt")
ext = np.loadtxt("exact.txt")
x = np.linspace(-20, 20, 400)
t = np.linspace(-20, 20, 400)
X, T = np.meshgrid(x, t)

err = abs(u - ext)**2
fig, ax = plt.subplots(figsize=(6, 10))
ax.axis('off')


cmap_name = "cool"
####### Row 0: h(t,x) ##################    


#------------------------------------------------------------------
#-----------------------------predicted----------------------------
#------------------------------------------------------------------
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=0.8, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(u.T, interpolation='nearest', cmap="cool",
			extent = [-20, 20, -20, 20],
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.set_title(' $Pred\;|\psi(x,t)|$')
#ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
#------------------------------------------------------------------
#-----------------------------Exact--------------------------------
#------------------------------------------------------------------
gs1 = gridspec.GridSpec(1, 2)
gs1.update(top=0.72, bottom=0.58, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs1[:, :])

h = ax.imshow(ext.T, interpolation='nearest', cmap="cool",
			extent = [-20, 20, -20, 20],
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.set_title(' $Exact\;|\psi(x,t)|$')
#ax.set_xlabel("$t$")
ax.set_ylabel("$x$")

#------------------------------------------------------------------
#-----------------------------Error--------------------------------
#------------------------------------------------------------------

gs2 = gridspec.GridSpec(1, 2)
gs2.update(top=0.5, bottom=0.36, left=0.15, right=0.9, wspace=0.2)
ax = plt.subplot(gs2[:, :])

h = ax.imshow(err.T, interpolation='nearest', cmap="cool",
			extent = [-20, 20, -20, 20],
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.set_title('$({|\psi(x,t)|_{Exact}-|\psi(x,t)|_{Pre}})^2$')
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")


gs3 = gridspec.GridSpec(1, 3)
gs3.update(top=0.28, bottom=0.13, left=0.13, right=0.9, wspace=0.5)

ax = plt.subplot(gs3[0, 0])
ax.plot(x,ext[100], c='cyan', ls='-', linewidth = 3, label = 'Exact')       
ax.plot(x,u[100], c = "purple",ls = '--', linewidth = 3, label = 'Prediction')
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")

ax = plt.subplot(gs3[0, 1])
ax.plot(x,ext[200], c='cyan', ls='-', linewidth = 3, label = 'Exact')       
ax.plot(x,u[200], c = "purple",ls = '--', linewidth = 3, label = 'Prediction')
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")

ax = plt.subplot(gs3[0, 2])
ax.plot(x,ext[300], c='cyan', ls='-', linewidth = 3, label = 'Exact')       
ax.plot(x,u[300], c = "purple",ls = '--', linewidth = 3, label = 'Prediction')
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=5, frameon=True)

#plt.savefig("KdV_Soliton.jpg", dpi = 600)
plt.show()



fig = plt.figure(figsize=(13, 8), constrained_layout=True)
ax = fig.add_subplot(121, projection='3d')




ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Bonus: To get rid of the grid as well:
ax.grid(False)
surf = ax.plot_surface(X, T, ext.T,  linewidth=0, cmap="cool", antialiased=False, alpha=0.6,rcount=200, ccount=200)

#cset = ax.contourf(X, T, q1_exact, zdir='z', offset=-0.5, cmap="cool")
#cset = ax.contour(X, T, q1_exact, zdir='x', offset=-15, cmap="jet")
#cset = ax.contour(X, T, q1_exact, zdir='y', offset=15, cmap="jet")

#fig.colorbar(surf, shrink=0.5, aspect=8)

ax.view_init(20,70)

ax.set_xticks([-20,0,20])
ax.set_xlabel('$x$')
ax.set_xlim(-20, 20)

ax.set_ylabel('$t$')
ax.set_yticks([-20,0,20])
ax.set_zticks([0.00,0.06, 0.12])
ax.set_zlabel('$\psi(x, t)$', rotation = 90)
ax.set_title(' $Exact\;|\psi(x,t)|$')

#fig.colorbar(surf, shrink=0.5, aspect=8)

ax = fig.add_subplot(122, projection='3d')




ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Bonus: To get rid of the grid as well:
ax.grid(False)
surf = ax.plot_surface(X, T, u.T,  linewidth=0, cmap="cool", antialiased=False, alpha=0.6,rcount=200, ccount=200)

#cset = ax.contourf(X, T, q1_exact, zdir='z', offset=-0.5, cmap="cool")
#cset = ax.contour(X, T, q1_exact, zdir='x', offset=-15, cmap="jet")
#cset = ax.contour(X, T, q1_exact, zdir='y', offset=15, cmap="jet")

#fig.colorbar(surf, shrink=0.5, aspect=8)

ax.view_init(20,70)

ax.set_xticks([-20,0,20])
ax.set_xlabel('$x$')
ax.set_xlim(-20, 20)

ax.set_ylabel('$t$')
ax.set_yticks([-20,0,20])
ax.set_zticks([0.00,0.06, 0.12])
#ax.set_zlabel('$\psi(x, t)$', rotation = 90)
ax.set_title(' $Pred\;|\psi(x,t)|$')

fig.colorbar(surf, shrink=0.5, aspect=8)
plt.savefig("KdV_Soliton_3d.jpg", dpi = 600)
plt.show()
