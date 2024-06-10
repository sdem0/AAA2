import numpy as np
import matplotlib.pyplot as plt

class CantileverBeamFEM:
    def __init__(self, N, L, EI, GJ):
        self.N = N  # Number of elements
        self.L = L  # Length of the beam (m)
        self.EI = EI  # Bending stiffness (Nm^2)
        self.GJ = GJ  # Torsional stiffness (Nm^2)
        self.dof = 3  # Degrees of freedom per node: twist, bending angle, vertical position
        self.total_dof = self.dof * (N + 1)  # Total degrees of freedom

    def element_stiffness_matrix(self):
        L = self.L / self.N
        
        # Bending stiffness matrix
        k_bending = self.EI / L**3 * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        
        # Torsional stiffness matrix
        k_torsion = self.GJ / L * np.array([
            [1, -1],
            [-1, 1]
        ])
        
        return k_bending, k_torsion

    def assemble_global_stiffness(self):
        K_global = np.zeros((self.total_dof, self.total_dof))
        L_element = self.L / self.N
        
        for i in range(self.N):
            k_bending, k_torsion = self.element_stiffness_matrix()
            
            # Bending DOFs: [phi_i, w_i, phi_{i+1}, w_{i+1}]
            bending_dofs = [
                self.dof * i + 1,
                self.dof * i + 2,
                self.dof * (i + 1) + 1,
                self.dof * (i + 1) + 2
            ]
            
            # Torsion DOFs: [theta_i, theta_{i+1}]
            torsion_dofs = [
                self.dof * i,
                self.dof * (i + 1)
            ]
            
            # Add bending stiffness matrix to global stiffness matrix
            for a in range(4):
                for b in range(4):
                    K_global[bending_dofs[a], bending_dofs[b]] += k_bending[a, b]
            
            # Add torsion stiffness matrix to global stiffness matrix
            for a in range(2):
                for b in range(2):
                    K_global[torsion_dofs[a], torsion_dofs[b]] += k_torsion[a, b]
                    
        return K_global

    def apply_boundary_conditions(self, K_global, F_global):
        # Constrain the first node (encastered end) in twist, vertical position, and bending angle
        constrained_dofs = [
            0,    # Twist at node 0
            1,    # Bending angle at node 0
            2     # Vertical position at node 0
        ]
        
        # Apply boundary conditions by modifying the stiffness matrix and force vector
        for dof in constrained_dofs:
            K_global[dof, :] = 0
            K_global[:, dof] = 0
            K_global[dof, dof] = 1
            F_global[dof] = 0
        
        return K_global, F_global

    def solve(self, F_global):
        K_global = self.assemble_global_stiffness()
        K_global, F_global = self.apply_boundary_conditions(K_global, F_global)
        
        displacements = np.linalg.solve(K_global, F_global)
        
        return displacements

    def plot_results(self, displacements):
        # Extract twist, bending angle, and vertical position
        twist = displacements[0::3]
        bending_angle = displacements[1::3]
        vertical_position = displacements[2::3]
        
        # Node positions along the span
        node_positions = np.linspace(0, self.L, self.N + 1)
        
        # Plot twist
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(node_positions, twist, 'b-o')
        plt.xlabel('Spanwise position (m)')
        plt.ylabel('Twist (radians)')
        plt.title('Twist Distribution')
        plt.grid(True)

        # Plot deflection
        plt.subplot(1, 2, 2)
        plt.plot(node_positions, vertical_position, 'r-o')
        plt.xlabel('Spanwise position (m)')
        plt.ylabel('Vertical Position (m)')
        plt.title('Deflection Distribution')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Example parameters
N = 20  # Number of elements
L = 10.0  # Length of the beam (m)
EI = 0.4186  # Bending stiffness (Nm^2)
GJ = 0.9539  # Torsional stiffness (Nm^2)

# Initialize FEM model
beam_fem = CantileverBeamFEM(N, L, EI, GJ)

# External forces and moments (example: shear forces and torques at each node)
F_global = np.zeros(beam_fem.total_dof)
F_global[2::3] = 100.0  # Shear force (N) at each node
F_global[0::3] = 10.0   # Torque moment (Nm) at each node

# Solve for displacements
displacements = beam_fem.solve(F_global)

# Plot the results
beam_fem.plot_results(displacements)
