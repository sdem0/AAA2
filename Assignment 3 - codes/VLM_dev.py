import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
s = 0.451 # m semi-span
c = 0.051 # m chord

# biot_savart
# Calculate the contribution to the induced velocity from a vortex segment
def dlxr_point(x1, y1, x2, y2, x, y):
    dL = np.array([x2 - x1, y2 - y1])
    R = np.array([x - (x1+x2)/2, y - (y1+y2)/2])
    R_mag = np.linalg.norm(R)
    if R_mag == 0:
        return 0  # Avoid division by zero
    dL_cross_R = np.cross(dL, R)  # Cross product in 2D (returns a scalar)
    return dL_cross_R / R_mag**3

# Integrate along the vortex line
def biot_savart_vort_line(x1, y1, x2, y2, x, y):
    integration_N = 10
    ys = np.linspace(y1, y2, integration_N + 1)
    xs = np.linspace(x1, x2, integration_N + 1)
    duind = 0
    for i in range(integration_N):
        duind += dlxr_point(xs[i], ys[i], xs[i + 1], ys[i + 1], x, y)
    u_ind = 1 / (4 * np.pi) * duind
    return u_ind

# whole wing discretisation
def VLM2Dmesh_even(N_pannels_chordwise, N_pannels_spanwise, l_wake, s, c):
    s = 2*s
    x_coords = np.linspace(0, c, N_pannels_chordwise + 1)
    y_coords = np.linspace(0, s, N_pannels_spanwise + 1)

    x_coords_wake = np.append(x_coords, c + l_wake)

    X, Y = np.meshgrid(x_coords, y_coords)
    X_wake, Y_wake = np.meshgrid(x_coords_wake, y_coords)
    grid_points = np.stack((X_wake, Y_wake), axis=-1)

    # Calculate center points for the original grid excluding the last row
    X_centers = (X[:-1, :-1] + X[1:, 1:]) / 2
    Y_centers = (Y[:-1, :-1] + Y[1:, 1:]) / 2
    
    # Combine the center points
    center_points = np.dstack((X_centers, Y_centers))

    #wake grid:
    return grid_points,center_points

# wake discretisation
# one single extra row of points further back at a distance l_wake

l_wake = 1 # m
N_pannels_chordwise = 10
N_pannels_spanwise = 20

grid_points, center_points = VLM2Dmesh_even(N_pannels_chordwise, N_pannels_spanwise, l_wake, s, c)
grid_points_plot = grid_points.reshape(-1, 2)
center_points_plot = center_points.reshape(-1, 2)

def plot_VLMgrid():
    plt.figure(figsize=(10, 5))
    plt.scatter(grid_points_plot[:, 0], grid_points_plot[:, 1], c='blue', marker='o', label='Grid Points')
    plt.scatter(center_points_plot[:, 0], center_points_plot[:, 1], c='red', marker='x', label='Center Points')
    for i in range(grid_points.shape[0]):
        plt.plot(grid_points[i, :, 0], grid_points[i, :, 1], 'blue')
    for j in range(grid_points.shape[1]):
        plt.plot(grid_points[:, j, 0], grid_points[:, j, 1], 'blue')
    plt.xlabel('Chordwise (x)')
    plt.ylabel('Spanwise (y)')
    plt.title('Grid Points and Center Points with Wake Region')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_VLMgrid()

# Flatten the arrays for plotting
# sum (contribution * vortex strength) + Vinf * sin(alpha) = 0 
# for all the centerpoints except for the wake ones
# setting up Ax = b system:
grid_points_flat = grid_points.reshape(-1, 2)
center_points_flat = center_points.reshape(-1, 2)

print("center_points",center_points)
print("grid_points",grid_points)

print("center_points_flat",center_points_flat)
print("grid_points_flat",grid_points_flat)

# all lines constituting each vortex ring:
vortex_ring_lines = []
for i in range(N_pannels_spanwise):
    for j in range(N_pannels_chordwise):
        #for amount of contributors:
        contribution_ij = []
        if (j+1)%N_pannels_chordwise == 0:
            # for trailing edge vortex ring cases (account for wake vorticity)
            line_top_to_wake = np.array([grid_points[i,-1],grid_points[i,j]])
            contribution_ij.append(line_top_to_wake)
            line_bottom_to_wake = np.array([grid_points[i+1,j],grid_points[i+1,-1]])
            contribution_ij.append(line_bottom_to_wake)
            line_left = np.array([grid_points[i,j],grid_points[i+1,j]])
            contribution_ij.append(line_left)
        else:
            # for non-trailing edge vortex ring cases
            line_top = np.array([grid_points[i,j+1],grid_points[i,j]])
            contribution_ij.append(line_top)
            line_bottom = np.array([grid_points[i+1,j],grid_points[i+1,j+1]])
            contribution_ij.append(line_bottom)
            line_left = np.array([grid_points[i,j],grid_points[i+1,j]])
            contribution_ij.append(line_left)
            line_right = np.array([grid_points[i+1,j+1],grid_points[i,j+1]])
            contribution_ij.append(line_right)

        vortex_ring_lines.append(contribution_ij)

print(vortex_ring_lines)

A =np.zeros((N_pannels_chordwise*N_pannels_spanwise,N_pannels_chordwise*N_pannels_spanwise))

for i in range(N_pannels_spanwise*N_pannels_chordwise):
    # center point i
    xc,yc = center_points_flat[i,0], center_points_flat[i,1]
    for j in range(N_pannels_spanwise*N_pannels_chordwise):
        #contributor j
        #into A[i,j] 
        A[i,j] = 0
        for k in vortex_ring_lines[j]:
            # for each line contributing to the jth column of the materix at row i
            x1,y1 = k[0,0],k[0,1]
            x2,y2 = k[1,0],k[1,1]
            A[i,j] += biot_savart_vort_line(x1,y1,x2,y2,xc,yc)

print(A)

alpha = np.deg2rad(10)
Vinf = 1 # m/s
b = -np.ones_like(A.T[0]) * np.sin(alpha) * Vinf
x = np.linalg.inv(A)@b
print(x)
x_2D = x.reshape(N_pannels_spanwise, N_pannels_chordwise)
print(x_2D)

#plot circulation distribution:
def plot_VLMresults():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot vectors A at positions B
    for i in range(center_points.shape[0]):
        for j in range(center_points.shape[1]):
            x, y = center_points[i, j]
            z = 0
            u, v, w = 0, 0, x_2D[i, j]
            ax.quiver(x, y, z, u, v, w, color='r')

    # Overlay the 2D plot on the xy plane
    ax.scatter(grid_points_plot[:, 0], grid_points_plot[:, 1], c='blue', marker='o', label='Grid Points')

    #for i in range(grid_points.shape[0]):
        #ax.plot(grid_points[i, :, 0], grid_points[i, :, 1], 'blue')

    #for j in range(grid_points.shape[1]):
        ax.plot(grid_points[:, j, 0], grid_points[:, j, 1], 'blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Vectors and 2D Grid Plot')

    plt.legend()
    plt.show()

plot_VLMresults()

