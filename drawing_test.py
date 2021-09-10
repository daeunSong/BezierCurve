import numpy as np
import matplotlib.pyplot as plt

# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# returns the first derivative
def get_first_deriv(a, b, c, d):
    return lambda t: (-3) * np.power(1 - t, 2) * a + 3 * (1 - 3*t) * (1 - t) * b + 3 * t * (2 - 3*t) * c + 3 * np.power(t, 2) * d

# returns the second derivative
def get_second_deriv(a, b, c, d):
    return lambda t: 6 * (1 - t) * a + 6 * (3*t - 2) * b + 6 * (1 - 3*t) * c + 6 * t * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return np.array([
        np.array([get_cubic(points[i], A[i], B[i], points[i + 1]), 
         get_first_deriv(points[i], A[i], B[i], points[i + 1]),
         get_second_deriv(points[i], A[i], B[i], points[i + 1])])
        for i in range(len(points) - 1)
    ])

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)[:,0]
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])


# generate 5 (or any number that you want) random points that we want to fit (or set them youreself)
points = np.random.rand(5, 2)

# read way points
f = open("input/heart_path_c.txt")

line = list(map(int, f.readline().split(" ")))
width = line[0]
height = line[1]

waypoints = []

while True:
  line = f.readline()
  if not line: break
  line = line.split(" ")
  y = float(line[0])# * width
  z = float(line[1].split("\n")[0]) * (-1)# * height 
  x = 0
  waypoints.append([y, z])

#waypoints = waypoints[:1000]
waypoints = np.array(waypoints)
curves = get_bezier_cubic(waypoints)

# fit the points with Bezier interpolation
# use 50 points between each consecutive points to draw the curve
path = evaluate_bezier(waypoints, 10)

# extract x & y coordinates of points
x, y = waypoints[:,0], waypoints[:,1]
px, py = path[:,0], path[:,1]

# plot
plt.figure(figsize=(width/50,height/50))
plt.xlim([0,1])
plt.ylim([-1,0])
plt.plot(px, py, 'b-', linewidth=1)
plt.plot(x, y, 'ro', markersize=2)
#plt.scatter(x, y, s=5, color='r')
plt.show()

for i in range(len(waypoints)-1):
    print(curves[i][2](1))

