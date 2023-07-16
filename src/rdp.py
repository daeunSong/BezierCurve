import numpy as np
from read_file import read_waypoints_from_file
from scipy.optimize import minimize
import time

EPSILON_d = 0.5
EPSILON_k = 2.0

def perpendicularDist (point, line):
    p1, p2 = line
    # line
    a = p2[1] - p1[1]
    b = - (p2[0] - p1[0])
    c = p1[1]*(p2[0]-p1[0]) + p1[0]*(p1[1]-p2[1])
    # point
    m,n = point
    # dist
    d = np.abs(a*m + b*n + c)/np.sqrt(np.power(a,2)+np.power(b,2))

    return d

def douglasPeucker (points):
    # find the point with the maximum dist
    max_dist = 0
    index = 0
    end = len(points)
    for i in list(range(1,end-1)):
        d = perpendicularDist(points[i], (points[0], points[-1]))
        if d > max_dist:
            index = i
            max_dist = d
    result = []
    # if max dist is greater than epsilon, recursively simplify
    if max_dist > EPSILON_d :
        # recursive call
        result1 = douglasPeucker(points[:index+1])
        result2 = douglasPeucker(points[index:])
        result = result1 + result2[1:]
    else :
        result = [points[0], points[-1]]

    return result

def obj (s):
    return s
def const (s, A, B, P_1):
    d = np.absolute(np.cross(P_1 - A, P_1 - B)/np.linalg.norm(P_1 - B))
    c = np.linalg.norm((B - P_1)*s)
    kappa = (2*d)/(3*c**2)
    return EPSILON_k - kappa

if __name__ == '__main__':
    file_name = "../input/heart_path_c.txt"
    waypoints, width, height = read_waypoints_from_file(file_name)
    # test_num = 100
    # result = douglasPeucker(list(waypoints[:test_num]))
    result = douglasPeucker(list(waypoints))

    from plot_bezier import plot_bezier
    from bezier_curve import Curves
    import matplotlib.pyplot as plt

    # c = Curves(waypoints[:test_num])
    # plot_bezier(c, plot_details=True)
    c = Curves(np.array(result))
    print("curves made")
    # plot_bezier(c, plot_details=True)
    #
    # from plot_bezier import parameterize_bezier
    # # waypoints = parameterize_bezier(c)
    # # c = Curves(waypoints)
    # # plot_bezier(c)
    #
    # # c = Curves(np.array(result))
    # # plot_bezier(c)
    #
    #
    ## if the control point vector is too small - scale up
    A = []; B = []
    A.append(c.curves[0].A)

    fig = plt.figure(figsize=(10,10))

    start = time.time()
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
    end = time.time()
    B.append(c.curves[-1].B)
    print(c.num_points)
    print(f"{end - start:.5f} sec")

    c.A = np.array(A)
    c.B = np.array(B)
    c.reset_bezeir()
    plot_bezier(c, color='c', fig=fig)
    print("cyan done")


    file_name = "../input/heart_path_m.txt"
    waypoints, width, height = read_waypoints_from_file(file_name)
    result = douglasPeucker(list(waypoints))

    c = Curves(np.array(result))

    A = []; B = []
    A.append(c.curves[0].A)

    start = time.time()
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
    end = time.time()
    B.append(c.curves[-1].B)
    print(c.num_points)
    print(f"{end - start:.5f} sec")

    c.A = np.array(A)
    c.B = np.array(B)
    c.reset_bezeir()
    plot_bezier(c, color='m', fig=fig)
    print("magenta done")

    file_name = "../input/heart_path_y.txt"
    waypoints, width, height = read_waypoints_from_file(file_name)
    result = douglasPeucker(list(waypoints))

    c = Curves(np.array(result))

    A = []; B = []
    A.append(c.curves[0].A)

    start = time.time()
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
    end = time.time()
    B.append(c.curves[-1].B)
    print(c.num_points)
    print(f"{end - start:.5f} sec")

    c.A = np.array(A)
    c.B = np.array(B)
    c.reset_bezeir()
    plot_bezier(c, color='y', fig=fig)
    print("yellow done")

    file_name = "../input/heart_path_k.txt"
    waypoints, width, height = read_waypoints_from_file(file_name)
    result = douglasPeucker(list(waypoints))

    c = Curves(np.array(result))

    A = []; B = []
    A.append(c.curves[0].A)

    start = time.time()
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
    end = time.time()
    B.append(c.curves[-1].B)
    print(c.num_points)
    print(f"{end - start:.5f} sec")

    c.A = np.array(A)
    c.B = np.array(B)
    c.reset_bezeir()
    plot_bezier(c, color='k', fig=fig)
    print("black done")


    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('heart_bezier.pdf')
    plt.show(block=True)
