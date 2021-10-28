import numpy as np

def read_waypoints_from_file (file_name):
    f = open(file_name)
    line = list(map(int, f.readline().split(" ")))
    width = line[0]
    height = line[1]

    waypoints = []

    while True:
        line = f.readline()

        if not line: break
        if line == 'End\n' :
            line = f.readline()
            if line == '': break

        x, y = list(map(float,line.split(" ")))
        waypoints.append([(x - 0.5) * width, (-y + 0.5) * height])
        # waypoints.append([float(x), float(y)*(-1)])

        #y = float(line[0])# * width
        #z = float(line[1].split("\n")[0]) * (-1)# * height
        #waypoints.append([y, z])

    return np.array(waypoints), width, height

def save_waypoints_to_file (waypoints, file_name="../input/heart_k.txt"):
    f = open(file_name, 'w')
    for waypoint in waypoints:
        f.write("%f " %waypoint[0])
        f.write("%f \n" %waypoint[1])
    f.close()

if __name__ == '__main__':
    file_name = "../input/heart_path_k.txt"
    waypoints, width, height = read_waypoints_from_file(file_name)
    save_waypoints_to_file(waypoints)
