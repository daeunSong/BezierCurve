import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

VEL_THRESHOLD = 0.01
ACC_THRESHOLD = 0.04

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(T, n = 20):
    return np.array([fun(t) for fun in T for t in np.linspace(0, 1, n)])

def evaluate_vel(c,n = 20):
    vel = []
    for curve in c.curves:
        for t in range(n):
            vel.append(np.linalg.norm(curve.tp(t)))

    # normalize to [0,1]
    vel = np.array(vel)
    vel = (vel - vel.min()) / (vel.max() - vel.min())
    return vel

def update(num, px, py, line):
    line.set_data(px[:num], py[:num])
    return line,

def plot_bezier(c, width = 0, height = 0, plot_details=True, animation=True):
    path = evaluate_bezier(c.T)
    waypoints = c.waypoints

    # extract x & y coordinates of points
    x, y = waypoints[:,0], waypoints[:,1]
    px, py = path[:,0], path[:,1]

    # plot
    if width != 0:
        fig = plt.figure(figsize=(width,height))
    else: fig = plt.figure(figsize=(10,8))

    # plt.xlim([0,1])
    # plt.ylim([-1,0])
    # plt.xlim([0.2,0.7])
    # plt.ylim([-0.85,-0.5])

    vel = evaluate_vel(c)
    # print(vel)

    plt.plot(x, y, 'bo', markersize=3)

    if animation:
        line,= plt.axes().plot(px, py)
        ani = FuncAnimation(fig, update, frames=len(vel), fargs=(px, py, line),interval=1, repeat=False)
        ani.save('animation.gif', fps=20)
    else :
        plt.plot(px, py, 'b-', linewidth=1)
        if plot_details:
            plot_control_points(c)
            plot_control_vector(c)
            # plot_threshold(c)
            # plot_curvature(c)
            # plot_test(c)
    plt.show()

def plot_control_points(c):
    x, y = c.A[:,0], c.A[:,1]
    plt.plot(x, y, 'go', markersize=2)
    x, y = c.B[:,0], c.B[:,1]
    plt.plot(x, y, 'go', markersize=2)

def plot_control_vector(c):
    for curve in c.curves:
        px1, py1 = curve.Pi
        px2, py2 = curve.A
        plt.plot([px1,px2], [py1,py2], 'g-', linewidth=1)

        px1, py1 = curve.B
        px2, py2 = curve.Pi_1
        plt.plot([px1,px2], [py1,py2], 'g-', linewidth=1)

def plot_threshold(c):
    vel_over = []; acc_over = []
    for curve in c.curves:
        vel = np.linalg.norm(curve.tp(1))
        acc = np.linalg.norm(curve.tpp(1))
        # print(speed, acc)
        if (vel > VEL_THRESHOLD):
            vel_over.append(curve.Pi_1)
        if (acc > ACC_THRESHOLD):
            acc_over.append(curve.Pi_1)
    vel_over = np.array(vel_over)
    plt.plot(vel_over[:,0], vel_over[:,1], 'ro', markersize=2)
    acc_over = np.array(acc_over[:-1]); # remove the last one
    plt.plot(acc_over[:,0], acc_over[:,1], 'yo', markersize=2)

def plot_curvature(c):
    curvatures = []
    p = []
    for curve in c.curves:
        curvature = np.cross(curve.tp(1), curve.tpp(1))
        if curvature < 0 : curvature = - curvature
        curvatures.append(curvature)

    curvatures = np.array(curvatures)
    curvatures = (curvatures - curvatures.min()) / (curvatures.max()-curvatures.min())
    # print(curvatures.argmax(), curvatures.max())
    # print(curvatures.argmin(), curvatures.min())

    for i in range(c.num_curves):
        if curvatures[i] > 0.4:
            p.append(c.curves[i].Pi)
    p = np.array(p)
    plt.plot(p[:,0], p[:,1], 'ro', markersize=2)

def plot_test (c):
    p = []
    for i in range(c.num_curves -1):
        curve_i = c.curves[i]
        curve_i1 = c.curves[i+1]
        vec = np.linalg.norm(curve_i.B - curve_i.Pi_1)
        if (vec < 1.0):
            p.append(c.curves[i].Pi_1)
    p = np.array(p)
    plt.plot(p[:,0], p[:,1], 'ro', markersize=2)

if __name__ == '__main__':
    # generate 5 (or any number that you want) random points that we want to fit (or set them youreself)
    waypoints = np.random.rand(5, 2)

    from bezier_curve import Curves
    c = Curves(waypoints)
    plot_bezier(c)
    plt.show()


