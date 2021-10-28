import numpy as np

def read_waypoints_from_file (file_name):
    waypoints = []
    try:
        f = open(file_name, 'r')
        lines = f.readlines()
        for line in lines:
            x, y = list(map(float,line.split()))
            waypoints.append([x, y])
    except:
        print("Error opening the file")
    return np.array(waypoints)

def save_waypoints_to_file (waypoints, file_name="../input/fixed_heart_k.txt"):
    f = open(file_name, 'w')
    for waypoint in waypoints:
        f.write("%f " %waypoint.x)
        f.write("%f\n" %waypoint.y)
    f.close()

if __name__ == '__main__':
    file_name = "../input/heart_c.txt"
    waypoints = read_waypoints_from_file(file_name)
    # save_waypoints_to_file(waypoints)
