import numpy as np

def Circle_waypoints(n, Tmax=10*np.pi):
    t = np.linspace(0, Tmax, n)
    x = 0.5 * np.cos(t) - 0.5
    y = np.sin(t)
    z = 0*t # One full sine wave, starts and ends at 0
    return t, np.stack((x, y, z), axis=-1)

def Helix_waypoints(n, Tmax=4*np.pi):
    t = np.linspace(0, Tmax, n)
    x = 0.5 * np.cos(t) - 0.5
    y = 0.5 * np.sin(t)
    z = 0.5 * t / Tmax * 2
    return t, np.stack((x, y, z), axis=-1)

# def Infinity_waypoints(n, Tmax=2*np.pi):
#     t = np.linspace(0, Tmax, n)
#     t_shift = t + np.pi/2
#     x = 0.5*np.cos(t_shift)
#     y = 0.5*np.sin(2*t_shift)
#     z = 0*t
#     return t, np.stack((x, y, z), axis=-1)

def write_waypoints_csv(filename, t, waypoints, yaw=0):
    with open(filename, "w") as f:
        f.write("t,x,y,z,yaw\n")
        for ti, (x, y, z) in zip(t, waypoints):
            f.write("{},{},{},{},{}\n".format(ti, x, y, z, yaw))
