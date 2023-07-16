from read_file import read_waypoints_from_file
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))

# file_name = "../input/heart_path_c.txt"
# waypoints, width, height = read_waypoints_from_file(file_name)
# waypoints = waypoints[:100]
# x, y = waypoints[:,0], waypoints[:,1]
# plt.plot(x, y, linewidth=1, color='c')
#
# file_name = "../input/heart_path_m.txt"
# waypoints, width, height = read_waypoints_from_file(file_name)
# waypoints = waypoints[:100]
# x, y = waypoints[:,0], waypoints[:,1]
# plt.plot(x, y, linewidth=1, color='m')
#
# file_name = "../input/heart_path_y.txt"
# waypoints, width, height = read_waypoints_from_file(file_name)
# waypoints = waypoints[:100]
# x, y = waypoints[:,0], waypoints[:,1]
# plt.plot(x, y, linewidth=1, color='y')
#
# file_name = "../input/heart_path_k.txt"
# waypoints, width, height = read_waypoints_from_file(file_name)
# waypoints = waypoints[:100]
# x, y = waypoints[:,0], waypoints[:,1]
# plt.plot(x, y, linewidth=1, color='k')

# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('heart_tsp.pdf')
# plt.show(block=True)


#### video figure

EPSILON_d = 0.5
EPSILON_k = 2.0
color = 'k'

file_name = "../input/heart_path_k.txt"
waypoints, width, height = read_waypoints_from_file(file_name)
waypoints = waypoints[:100]
x, y = waypoints[:,0], waypoints[:,1]
plt.plot(x, y, linewidth=1, color=color)
plt.plot(x, y, 'bo', markersize=2, color=color)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('heart_1.pdf')
plt.show(block=False)

from rdp import douglasPeucker, obj, const
import numpy as np

fig = plt.figure(figsize=(10,10))
result = douglasPeucker(list(waypoints))
result = np.array(result)

x, y = result[:,0], result[:,1]
plt.plot(x, y, linewidth=1, color=color)
plt.plot(x, y, 'bo', markersize=2, color=color)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('heart_2.pdf')
plt.show(block=False)

from bezier_curve import Curves
from plot_bezier import plot_bezier
from scipy.optimize import minimize

c = Curves(np.array(result))
A = []; B = []
A.append(c.curves[0].A)

fig = plt.figure(figsize=(10,10))
plot_bezier(c, color=color, fig=fig, plot_details=True)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('heart_3.pdf')
plt.show(block=False)

fig = plt.figure(figsize=(10,10))

for i in range(c.num_curves -1):
    curve_i = c.curves[i]
    curve_i1 = c.curves[i+1]
    kappa = c.curves[i].get_curvature_at_the_end()
    if (kappa > EPSILON_k):
        scale = minimize(obj, 1.0, tol=1e-8, bounds=(), constraints={'type':'ineq', 'fun': const, 'args': (curve_i.A, curve_i.B, curve_i.Pi_1)})
        scale = scale.x[0]
        curve_i.B = (curve_i.B - curve_i.Pi_1) * scale + curve_i.Pi_1
        B.append(curve_i.B)
        curve_i1.A = (curve_i1.A - curve_i1.Pi) * scale + curve_i1.Pi
        A.append(curve_i1.A)
    else :
        A.append(curve_i1.A)
        B.append(curve_i.B)
B.append(c.curves[-1].B)


c.A = np.array(A)
c.B = np.array(B)
c.reset_bezeir()
plot_bezier(c, color=color, fig=fig, plot_details=True)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('heart_4.pdf')
plt.show(block=False)

