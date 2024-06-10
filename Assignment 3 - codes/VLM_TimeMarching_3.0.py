import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.linalg as ln
import json

import FEM.fem_linear as fe
import FEM.fem_loads as fl
import FEM.fem_validation as val

def cross_product_2d(a, b):
    #Calculate the cross product of two 2D vectors.
    return a[0] * b[1] - a[1] * b[0]

#B-S function
def biot_savart_vortex_line(x1, y1, x2, y2, gamma, xr, yr):
    # Compute the vortex line vector
    dx = x2 - x1
    dy = y2 - y1

    # Compute the length of the vortex line
    line_length = np.sqrt(dx ** 2 + dy ** 2)

    # direction vector
    el = np.array([dx, dy])
    el = el * (1 / (line_length))

    # Parametric representation of the vortex line
    t = np.linspace(0, 1, 100)  # Parametric variable
    x_line = x1 + t * dx
    y_line = y1 + t * dy

    # Compute the induced velocity at the reference point

    vz = 0
    for i in range(len(t) - 1):
        # Midpoint of the line segment
        xm = (x_line[i] + x_line[i + 1]) / 2
        ym = (y_line[i] + y_line[i + 1]) / 2

        # Distance vector from midpoint to reference point
        rx = xr - xm
        ry = yr - ym

        # Distance from midpoint to reference point
        r = np.sqrt(rx ** 2 + ry ** 2)

        # Differential length of the vortex line
        dl = line_length / len(t)

        vector_dl = dl * el
        vector_r = np.array([rx, ry])

        cross = cross_product_2d(vector_dl, vector_r)

        # Induced velocity components (using cross product in 2D)
        dvz = (gamma / (4 * np.pi)) * ( cross / (r*r*r))

        vz += dvz

    return vz

def estimate_lift(U, alpha_mid, Ab, Aw, x_b_mid, x_w_mid, cell_c, cell_span, n_w, span, c, sim_time, rho=1.225):
    # Initial Condition Estimation
    U_normal_col = U * np.sin(np.radians(alpha)) * np.ones((x_b_mid.size, 1))
    gamma_b_col = - np.linalg.inv(Ab) @ U_normal_col
    gamma_w_col = np.zeros(x_w_mid.size).reshape(-1, 1)

    gamma_b_matrix = gamma_b_col.reshape(cell_c, cell_span)
    gamma_w_matrix = gamma_w_col.reshape(n_w * cell_c, cell_span)
    d_gamma_b = np.zeros_like(gamma_b_matrix)
    gamma_b_matrix_old = np.zeros_like(gamma_b_matrix)
    gamma_for_lift = np.zeros_like(gamma_b_matrix)
    lift = np.zeros_like(gamma_b_matrix)

    # Time Marching
    cell_length = c / cell_c
    time_step = cell_length / U
    sim_step = int(sim_time / time_step)
    time = np.linspace(0, sim_time, sim_step)
    cl_time = np.zeros_like(time)

    print(f"Time Step: {time_step} s")

    # Iteration Process
    for i in tqdm(range(sim_step), desc="Processing"):
        gamma_b_matrix_old = gamma_b_matrix

        gamma_w_matrix[1:] = gamma_w_matrix[:-1]
        gamma_w_matrix[0] = -gamma_b_matrix[-1]

        gamma_w_col = gamma_w_matrix.reshape(-1, 1)

        gamma_b_col = - np.linalg.inv(Ab) @ (U_normal_col - Aw @ gamma_w_col)
        gamma_b_matrix = gamma_b_col.reshape(cell_c, cell_span)
        gamma_w_matrix = gamma_w_col.reshape(n_w * cell_c, cell_span)

        d_gamma_b = (gamma_b_matrix - gamma_b_matrix_old) / time_step

        # Lift Estimation
        gamma_for_lift[0] = gamma_b_matrix[0]
        for j in range(1, gamma_b_matrix.shape[0]):
            gamma_for_lift[j] = gamma_b_matrix[j] - gamma_b_matrix[j - 1]

        cell_dy = span / cell_span
        cell_area = cell_dy * cell_length
        lift = - (rho * U * np.multiply(np.cos(np.radians(alpha_mid)), gamma_for_lift) * cell_dy + rho * d_gamma_b * cell_area)

        cl_time[i] = np.sum(lift) / (0.5 * rho * U * U * span * c)

    return lift, time_step, cl_time

def compute_fem_response(config_path, moment_str_flatten, lift_str_flatten, d=-0.2, KBT=-0.2, bc='CF'):
    # Load configuration
    with open(config_path) as f:
        constant = json.load(f)

    # Decouple torsion from bending
    constant['fem']['d'] = d
    constant['fem']['KBT'] = KBT

    # Boundary condition
    constant['fem']['bc'] = bc

    # Initialise FEM properties
    fem = fe.initialise_fem(constant['fem'])
    n_nd = fem['n_nd']
    n_dof = fem['n_dof']
    b_u = fem['b_u']

    # Applied load values at nodes
    r = moment_str_flatten  # torque distribution, [Nm/m]
    f = lift_str_flatten    # shear force distribution, [N/m]
    q = 0.0 * np.ones((n_nd,))  # bending moment distribution, [Nm/m]

    # 2nd order system approach
    KK = fe.mat_stiffness(fem)
    KKinv = ln.inv(KK)

    DD = fl.mat_force_dst(fem)
    u_red = fl.generate_load_vector(fem, r, f, q)

    # Solution
    resp1_red = KKinv.dot(DD.dot(u_red))

    # Add the known DOFs to the eigen vectors
    resp1 = np.zeros((n_dof,))
    resp1[b_u] = resp1_red

    # Extract specific responses
    str_twist = resp1[0::3]
    str_bending = resp1[1::3]
    str_angle = resp1[2::3]

    return str_twist, str_bending, str_angle

def coupling_mapping(d_span, lift, y_b_mid, c_elastic, semi_span, n_span, n_nd, cell_c):
    # Calculate midpoint aero span
    midpoint_aero_span = (d_span[:-1] + d_span[1:]) / 2
    midpoint_aero_span = midpoint_aero_span[(n_span - 1) // 2:] - semi_span

    # Calculate lift_half
    lift_half = lift[:, ((n_span - 1) // 2):]

    # Generate node_fem_span
    node_fem_span = np.linspace(0, semi_span, n_nd)

    # Find the closest node
    closest_node = np.zeros_like(node_fem_span)
    for i in range(len(node_fem_span)):
        closest_node[i] = np.argmin(np.abs(midpoint_aero_span - node_fem_span[i]))

    # Calculate lift_beam_flatten
    lift_beam_flatten = np.sum(lift_half, axis=0)

    # Initialize moment_beam_flatten
    moment_beam_flatten = np.zeros((n_span - 1) // 2)

    # Calculate moment_beam_flatten
    for i in range((n_span - 1) // 2):
        for j in range(cell_c):
            moment_beam_flatten[i] -= lift_half[j, i] * (y_b_mid[j, i] - c_elastic)

    # Initialize structure input arrays
    lift_str_flatten = np.zeros(n_nd)
    moment_str_flatten = np.zeros(n_nd)

    # Map lift and moment to structure nodes
    for i in range(n_nd):
        index = int(closest_node[i])
        lift_str_flatten[i] = lift_beam_flatten[index]
        moment_str_flatten[i] = moment_beam_flatten[index]

    return lift_str_flatten, moment_str_flatten

def map_inverse(node_fem_span, midpoint_aero_span, str_twist):
    # Find the closest node
    closest_node = np.zeros_like(midpoint_aero_span, dtype=int)
    for i in range(len(midpoint_aero_span)):
        closest_node[i] = np.argmin(np.abs(node_fem_span - midpoint_aero_span[i]))

    # Initialize twist_bem_flatten
    twist_bem_flatten = np.zeros_like(midpoint_aero_span)

    # Map str_twist to twist_bem_flatten
    for i in range(len(midpoint_aero_span)):
        index = int(closest_node[i])
        twist_bem_flatten[i] = str_twist[index]

    return twist_bem_flatten

#Condition defination
span = 2 * 0.451
semi_span = span / 2
c = 0.051
c_elastic = 0.5 * c
U = 20
alpha = 10
sim_time = 1
rho = 1.225

#discretization defination
n_span = 11
cell_span = n_span - 1
n_c = 8
cell_c = n_c - 1
cell_length = c / cell_c
n_w = 2

d_span = np.linspace(0 , span, n_span)
d_c = np.linspace(0 , c , n_c)
d_w = np.linspace(c , c + n_w * c, n_c * n_w - (n_w - 1))

x_b_node = np.tile(d_span, (n_c , 1))
y_b_node = np.tile(d_c.reshape(n_c, 1), (1, n_span))

x_w_node = np.tile(d_span, (n_c * n_w - (n_w - 1) , 1))
y_w_node = np.tile(d_w.reshape(n_c * n_w - (n_w - 1), 1), (1, n_span))

#offset 1/4 chord
y_b_node = y_b_node + 0.25 * (c / cell_c)
y_w_node = y_w_node + 0.25 * (c / cell_c)

# Compute Cell midpoints
x_b_mid = (x_b_node[:, :-1] + x_b_node[:, 1:]) / 2
y_b_mid = (y_b_node[:-1, :] + y_b_node[1:, :]) / 2

x_w_mid = (x_w_node[:, :-1] + x_w_node[:, 1:]) / 2
y_w_mid = (y_w_node[:-1, :] + y_w_node[1:, :]) / 2

# Adjust shapes to match
x_b_mid = (x_b_node[:-1, :-1] + x_b_node[:-1, 1:] + x_b_node[1:, :-1] + x_b_node[1:, 1:]) / 4
y_b_mid = (y_b_node[:-1, :-1] + y_b_node[:-1, 1:] + y_b_node[1:, :-1] + y_b_node[1:, 1:]) / 4

x_w_mid = (x_w_node[:-1, :-1] + x_w_node[:-1, 1:] + x_w_node[1:, :-1] + x_w_node[1:, 1:]) / 4
y_w_mid = (y_w_node[:-1, :-1] + y_w_node[:-1, 1:] + y_w_node[1:, :-1] + y_w_node[1:, 1:]) / 4

# Calculate midpoint aero span
midpoint_aero_span = (d_span[:-1] + d_span[1:]) / 2
midpoint_aero_span = midpoint_aero_span[(n_span - 1) // 2:] - semi_span

# Estimate Cell span-wise length and area
cell_area = np.zeros_like(x_b_mid)
cell_dy = np.zeros_like(x_b_mid)
for i in range(cell_c):
    for j in range(cell_span):
        cell_area[i,j] = (cell_length * (x_b_node[i,j+1] - x_b_node[i,j]))
        cell_dy[i,j] = x_b_node[i,j+1] - x_b_node[i,j]


# Plotting the node lattice
plt.figure(figsize=(8, 6))
plt.plot(x_b_node, y_b_node, 'bo', label='Nodes')  # 'bo' stands for blue color and circle marker
plt.plot(x_w_node, y_w_node, 'go', label='Nodes')  # 'bo' stands for blue color and circle marker
plt.plot(x_b_mid, y_b_mid, 'ro', label='Midpoints')
plt.plot(x_w_mid, y_w_mid, 'ko', label='Midpoints')
plt.title('Node Lattice Plot')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()

# Structural Discretization
if __name__ == "__main__":

    with open('../Configurations/wing_TangDowell.json') as f:
        constant = json.load(f)

    # Decouple torsion from bending:
    constant['fem']['d'] = 0
    constant['fem']['KBT'] = 0

    # Boundary condition:
    constant['fem']['bc'] = 'CF'

    # Initialise FEM properties
    fem = fe.initialise_fem(constant['fem'])
    n_nd = fem['n_nd']
    n_dof = fem['n_dof']
    b_u = fem['b_u']
    y_nd = fem['y_nd']

# Generate node_fem_span
node_fem_span = np.linspace(0, semi_span, n_nd)

# flatten the position matrixs for the input of BS function

# bound midpoint position matrix flattened
x_b_mid_flat = x_b_mid.flatten()
y_b_mid_flat = y_b_mid.flatten()

# bound node point matrix flattened, 1-2-3-4 indicating four nodes for a cell
x_1 = x_b_node[0:n_c-1, 0:n_span-1]
x_b_1_flat = x_1.flatten()
x_2 = x_b_node[1:n_c, 0:n_span-1]
x_b_2_flat = x_2.flatten()
x_3 = x_b_node[1:n_c ,1:n_span]
x_b_3_flat = x_3.flatten()
x_4 = x_b_node[0:n_c-1, 1:n_span,]
x_b_4_flat = x_4.flatten()
y_1 = y_b_node[0:n_c-1, 0:n_span-1]
y_b_1_flat = y_1.flatten()
y_2 = y_b_node[1:n_c, 0:n_span-1]
y_b_2_flat = y_2.flatten()
y_3 = y_b_node[1:n_c, 1:n_span]
y_b_3_flat = y_3.flatten()
y_4 = y_b_node[0:n_c-1, 1:n_span]
y_b_4_flat = y_4.flatten()

# wake flattened, as the same way
x_w_mid_flat = x_w_mid.flatten()
y_w_mid_flat = y_w_mid.flatten()

x_1 = x_w_node[0:n_c * n_w - (n_w - 1)-1, 0:n_span-1]
x_w_1_flat = x_1.flatten()
x_2 = x_w_node[1:n_c * n_w - (n_w - 1), 0:n_span-1]
x_w_2_flat = x_2.flatten()
x_3 = x_w_node[1:n_c * n_w - (n_w - 1), 1:n_span]
x_w_3_flat = x_3.flatten()
x_4 = x_w_node[0:n_c * n_w - (n_w - 1)-1, 1:n_span]
x_w_4_flat = x_4.flatten()
y_1 = y_w_node[0:n_c * n_w - (n_w - 1)-1, 0:n_span-1]
y_w_1_flat = y_1.flatten()
y_2 = y_w_node[1:n_c * n_w - (n_w - 1), 0:n_span-1]
y_w_2_flat = y_2.flatten()
y_3 = y_w_node[1:n_c * n_w - (n_w - 1), 1:n_span]
y_w_3_flat = y_3.flatten()
y_4 = y_w_node[0:n_c * n_w - (n_w - 1)-1, 1:n_span]
y_w_4_flat = y_4.flatten()

# Building AIC matrix
Ab = np.zeros((x_b_mid.size, x_b_mid.size))
Aw = np.zeros((x_b_mid.size, x_w_mid.size))

for i in range(x_b_mid.size):
    for j in range(x_b_mid.size):
        Ab[i, j] = -(biot_savart_vortex_line(x_b_1_flat[j], y_b_1_flat[j], x_b_2_flat[j], y_b_2_flat[j], 1, x_b_mid_flat[i], y_b_mid_flat[i])
                    +biot_savart_vortex_line(x_b_2_flat[j], y_b_2_flat[j], x_b_3_flat[j], y_b_3_flat[j], 1, x_b_mid_flat[i], y_b_mid_flat[i])
                    +biot_savart_vortex_line(x_b_3_flat[j], y_b_3_flat[j], x_b_4_flat[j], y_b_4_flat[j], 1, x_b_mid_flat[i], y_b_mid_flat[i])
                    +biot_savart_vortex_line(x_b_4_flat[j], y_b_4_flat[j], x_b_1_flat[j], y_b_1_flat[j], 1, x_b_mid_flat[i], y_b_mid_flat[i]))

for i in range(x_b_mid.size):
    for j in range(x_w_mid.size):
        Aw[i, j] = -(biot_savart_vortex_line(x_w_1_flat[j], y_w_1_flat[j], x_w_2_flat[j], y_w_2_flat[j], 1, x_b_mid_flat[i], y_b_mid_flat[i])
                    +biot_savart_vortex_line(x_w_2_flat[j], y_w_2_flat[j], x_w_3_flat[j], y_w_3_flat[j], 1, x_b_mid_flat[i], y_b_mid_flat[i])
                    +biot_savart_vortex_line(x_w_3_flat[j], y_w_3_flat[j], x_w_4_flat[j], y_w_4_flat[j], 1, x_b_mid_flat[i], y_b_mid_flat[i])
                    +biot_savart_vortex_line(x_w_4_flat[j], y_w_4_flat[j], x_w_1_flat[j], y_w_1_flat[j], 1, x_b_mid_flat[i], y_b_mid_flat[i]))

# Define AOA
alpha_mid = alpha * np.ones((cell_c, cell_span))

# lift estimation
lift, time_step, cl_time = estimate_lift(U, alpha_mid, Ab, Aw, x_b_mid, x_w_mid, cell_c, cell_span, n_w, span, c, sim_time)

# Plot Contour of Wing
plt.figure(figsize=(8, 8))
plt.contourf(x_b_mid, y_b_mid, lift, cmap='viridis')
plt.colorbar(label='Lift')
plt.title('Lift Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)

plt.gca().set_aspect('equal', adjustable='box')

plt.show()

lift_str_flatten, moment_str_flatten = coupling_mapping(d_span, lift, y_b_mid, c_elastic, semi_span, n_span, n_nd, cell_c)

print(f"First Lift:")
print(np.sum(lift))

# Solve Structure
np.set_printoptions(precision=4, linewidth=400)

if __name__ == "__main__":

    config_path = '../Configurations/wing_TangDowell.json'

    str_twist, str_bending, str_angle = compute_fem_response(config_path, moment_str_flatten, lift_str_flatten)

# region Plot
lbl_y = ['theta, [rad]', 'v, [m]', 'beta, [rad]']

fig, ax = plt.subplots(3, 1, sharex=True, num=1)
ax[0].plot(y_nd, str_twist)
ax[1].plot(y_nd, str_bending)
ax[2].plot(y_nd, str_angle)
for j in range(3):
    ax[j].set_ylabel(lbl_y[j])

plt.show()
# endregion

twist_bem_flatten = map_inverse(node_fem_span, midpoint_aero_span, str_twist)

# iteration process
iteration = 5
alpha_mid_origin = alpha_mid

###
for i in range(iteration):
    lift_old = np.sum(lift)
    mirrored = twist_bem_flatten[::-1]
    twist_bem_flatten = np.concatenate((mirrored, twist_bem_flatten))
    twist_bem_flatten = twist_bem_flatten * 180 / np.pi
    alpha_mid = alpha_mid_origin + twist_bem_flatten

    lift, time_step, cl_time = estimate_lift(U, alpha_mid, Ab, Aw, x_b_mid, x_w_mid, cell_c, cell_span, n_w, span, c, sim_time)

    lift_str_flatten, moment_str_flatten = coupling_mapping(d_span, lift, y_b_mid, c_elastic, semi_span, n_span, n_nd, cell_c)

    if __name__ == "__main__":

        config_path = '../Configurations/wing_TangDowell.json'

        str_twist, str_bending, str_angle = compute_fem_response(config_path, moment_str_flatten, lift_str_flatten)

    twist_bem_flatten = map_inverse(node_fem_span, midpoint_aero_span, str_twist)
    lift_res = np.sum(lift) - lift_old
    print(f"Lift Res:")
    print(lift_res)
###

# Plot Contour of Wing
plt.figure(figsize=(8, 8))
plt.contourf(x_b_mid, y_b_mid, lift, cmap='viridis')
plt.colorbar(label='Lift')
plt.title('Lift Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)

plt.gca().set_aspect('equal', adjustable='box')

plt.show()

lbl_y = ['theta, [rad]', 'v, [m]', 'beta, [rad]']

fig, ax = plt.subplots(3, 1, sharex=True, num=1)
ax[0].plot(y_nd, str_twist)
ax[1].plot(y_nd, str_bending)
ax[2].plot(y_nd, str_angle)
for j in range(3):
    ax[j].set_ylabel(lbl_y[j])

plt.show()

