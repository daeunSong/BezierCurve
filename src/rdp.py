import numpy as np
from read_file import read_waypoints_from_file

EPSILON = 1.0

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
    if max_dist > EPSILON :
        # recursive call
        result1 = douglasPeucker(points[:index+1])
        result2 = douglasPeucker(points[index:])
        result = result1 + result2[1:]
    else :
        result = [points[0], points[-1]]

    return result


if __name__ == '__main__':
    file_name = "../input/heart_path_c.txt"
    waypoints, width, height = read_waypoints_from_file(file_name)
    test_num = 100
    result = douglasPeucker(list(waypoints[:test_num]))

    from plot_bezier import plot_bezier
    from bezier_curve import Curves

    # c = Curves(waypoints[:test_num])
    # plot_bezier(c, plot_details=True)
    c = Curves(np.array(result))
    plot_bezier(c, plot_details=True)
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
    # ## if the control point vector is too small - scale up
    # scale = 2.0
    # A = []; B = []
    # A.append(c.curves[0].A)
    # for i in range(c.num_curves -1):
    #     curve_i = c.curves[i]
    #     curve_i1 = c.curves[i+1]
    #     vec = np.linalg.norm(curve_i.B - curve_i.Pi_1)
    #     # print(vec)
    #     if (vec < 0.8):
    #         curve_i.B = (curve_i.B - curve_i.Pi_1) * scale + curve_i.Pi_1
    #         B.append(curve_i.B)
    #         curve_i1.A = (curve_i1.A - curve_i1.Pi) * scale + curve_i1.Pi
    #         A.append(curve_i1.A)
    #     else :
    #         A.append(curve_i1.A)
    #         B.append(curve_i.B)
    # B.append(c.curves[-1].B)
    #
    # c.A = np.array(A)
    # c.B = np.array(B)
    # c.reset_bezeir()
    #
    # plot_bezier(c, plot_details=True)
