import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

class aero_model():
    """
    Steady vlm with fixed wake for a given wing configuration with twist distribution
    
    Example usage
    model = aero_model()
    model.VLM2Dmesh_even(N_pannels_chordwise=10, N_pannels_spanwise=10, l_wake=0.2)
    model.plot_VLMgrid()
    model.VLM()
    cl, cm = model.compute_cl_cm()
    model.plot_cl_cm(cl, cm)
    model.plot_lift_distribution()
    model.plot_VLMresults()

    """
    def __init__(self, s = 0.451,c = 0.051, AoA_distribution=False):
        self.c = c #chord m
        self.s = s #span m
        self.AoA = AoA_distribution 

    
    def VLM2Dmesh_even(self, N_pannels_chordwise, N_pannels_spanwise, l_wake):
        """ 
        generate center point and mesh matrices containing info on XY coordinates of each point
        given a mesh size and wake length desired, the mesh is an even grid.
        """
        self.N_pannels_chordwise = N_pannels_chordwise
        self.N_pannels_spanwise = N_pannels_spanwise
        s = 2 * self.s

        x_coords = np.linspace(0, self.c, N_pannels_chordwise + 1)
        y_coords = np.linspace(0, s, N_pannels_spanwise + 1)

        x_coords_wake = np.append(x_coords, self.c + l_wake)

        X, Y = np.meshgrid(x_coords, y_coords)
        X_wake, Y_wake = np.meshgrid(x_coords_wake, y_coords)
        self.grid_points = np.stack((X_wake, Y_wake), axis=-1)

        # Calculate center points for the original grid excluding the last row
        X_centers = (X[:-1, :-1] + X[1:, 1:]) / 2
        Y_centers = (Y[:-1, :-1] + Y[1:, 1:]) / 2
    
        # Combine the center points
        self.center_points = np.dstack((X_centers, Y_centers))
        print("mesh complete")

    def VLM2Dmesh_cos(self, N_pannels_chordwise, N_pannels_spanwise, l_wake):
        """ 
        Generate center point and mesh matrices containing info on XY coordinates of each point
        given a mesh size and wake length desired. The mesh is cosine spaced in the spanwise direction
        and evenly spaced in the chordwise direction.
        """
        self.N_pannels_chordwise = N_pannels_chordwise
        self.N_pannels_spanwise = N_pannels_spanwise
        s = 2 * self.s

        # Even distribution along the chord
        x_coords = np.linspace(0, self.c, N_pannels_chordwise + 1)
        
        # Cosine distribution along the span
        y_cos = 0.5 * (1 - np.cos(np.linspace(0, np.pi, N_pannels_spanwise + 1))) * s
        
        x_coords_wake = np.append(x_coords, self.c + l_wake)

        X, Y = np.meshgrid(x_coords, y_cos)
        X_wake, Y_wake = np.meshgrid(x_coords_wake, y_cos)
        self.grid_points = np.stack((X_wake, Y_wake), axis=-1)

        # Calculate center points for the original grid excluding the last row
        X_centers = (X[:-1, :-1] + X[1:, 1:]) / 2
        Y_centers = (Y[:-1, :-1] + Y[1:, 1:]) / 2

        # Combine the center points
        self.center_points = np.dstack((X_centers, Y_centers))
        #print("Cosine mesh complete")

    def plot_VLMgrid(self):

        grid_points_plot = self.grid_points.reshape(-1, 2)
        center_points_plot = self.center_points.reshape(-1, 2)

        plt.figure(figsize=(10, 5))
        plt.scatter(grid_points_plot[:, 0], grid_points_plot[:, 1], c='blue', marker='o', label='Grid Points')
        plt.scatter(center_points_plot[:, 0], center_points_plot[:, 1], c='red', marker='x', label='Center Points')
        for i in range(self.grid_points.shape[0]):
            plt.plot(self.grid_points[i, :, 0], self.grid_points[i, :, 1], 'blue')
        for j in range(self.grid_points.shape[1]):
            plt.plot(self.grid_points[:, j, 0], self.grid_points[:, j, 1], 'blue')
        plt.xlabel('Chordwise (x)')
        plt.ylabel('Spanwise (y)')
        plt.title('Grid Points and Center Points with Wake Region')
        plt.legend()
        plt.grid(True)
        plt.show()

    def dlxr_point(self, x1, y1, x2, y2, x, y):
        """ 
        used in biot_savart for integration
        """
        dL = np.array([x2 - x1, y2 - y1])
        R = np.array([x - (x1+x2)/2, y - (y1+y2)/2])
        R_mag = np.linalg.norm(R)
        if R_mag == 0:
            return 0  # Avoid division by zero
        dL_cross_R = np.cross(dL, R)  # Cross product in 2D (returns a scalar)
        return dL_cross_R / R_mag**3

    def biot_savart_vort_line(self, x1, y1, x2, y2, x, y):
        """ 
        induced vertical velocity in a point xx,y from a filament x1,y1 - x2,y2
        """
        integration_N = 10
        ys = np.linspace(y1, y2, integration_N + 1)
        xs = np.linspace(x1, x2, integration_N + 1)
        duind = 0
        for i in range(integration_N):
            duind += self.dlxr_point(xs[i], ys[i], xs[i + 1], ys[i + 1], x, y)
        u_ind = 1 / (4 * np.pi) * duind
        return u_ind

    def VLM(self):

        vortex_ring_lines = []
        for i in range(self.N_pannels_spanwise):
            for j in range(self.N_pannels_chordwise):
                #for amount of contributors:
                contribution_ij = []
                if (j+1)%self.N_pannels_chordwise == 0:
                    # for trailing edge vortex ring cases (account for wake vorticity)
                    line_top_to_wake = np.array([self.grid_points[i,-1],self.grid_points[i,j]])
                    contribution_ij.append(line_top_to_wake)
                    line_bottom_to_wake = np.array([self.grid_points[i+1,j],self.grid_points[i+1,-1]])
                    contribution_ij.append(line_bottom_to_wake)
                    line_left = np.array([self.grid_points[i,j],self.grid_points[i+1,j]])
                    contribution_ij.append(line_left)
                else:
                    # for non-trailing edge vortex ring cases
                    line_top = np.array([self.grid_points[i,j+1],self.grid_points[i,j]])
                    contribution_ij.append(line_top)
                    line_bottom = np.array([self.grid_points[i+1,j],self.grid_points[i+1,j+1]])
                    contribution_ij.append(line_bottom)
                    line_left = np.array([self.grid_points[i,j],self.grid_points[i+1,j]])
                    contribution_ij.append(line_left)
                    line_right = np.array([self.grid_points[i+1,j+1],self.grid_points[i,j+1]])
                    contribution_ij.append(line_right)

                vortex_ring_lines.append(contribution_ij)

        A =np.zeros((self.N_pannels_chordwise*self.N_pannels_spanwise,self.N_pannels_chordwise*self.N_pannels_spanwise))

        grid_points_flat = self.grid_points.reshape(-1, 2)
        center_points_flat = self.center_points.reshape(-1, 2)

        for i in range(self.N_pannels_spanwise*self.N_pannels_chordwise):
            # center point i
            xc,yc = center_points_flat[i,0], center_points_flat[i,1]
            for j in range(self.N_pannels_spanwise*self.N_pannels_chordwise):
                #contributor j
                #into A[i,j] 
                A[i,j] = 0
                for k in vortex_ring_lines[j]:
                    # for each line contributing to the jth column of the materix at row i
                    x1,y1 = k[0,0],k[0,1]
                    x2,y2 = k[1,0],k[1,1]
                    A[i,j] += self.biot_savart_vort_line(x1,y1,x2,y2,xc,yc)
        #print("Contribution matrix constructed")
        
        if self.AoA == 0:
            alpha = np.deg2rad(10)
        else:
            print("variable alpha distribution not implemented")
        
        Vinf = 1 # m/s
        b = -np.ones_like(A.T[0]) * np.sin(alpha) * Vinf
        self.vorticity_map = np.linalg.inv(A)@b
        self.vorticity_map_2D = np.fliplr(self.vorticity_map.reshape(self.N_pannels_spanwise, self.N_pannels_chordwise))
        #print("VLM solved")

    def plot_VLMresults(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot vectors A at positions B
        for i in range(self.center_points.shape[0]):
            for j in range(self.center_points.shape[1]):
                x, y = self.center_points[i, j]
                z = 0
                u, v, w = 0, 0, self.vorticity_map_2D[i, j]
                ax.quiver(x, y, z, u, v, w, color='r')

        # Overlay the 2D plot on the xy plane
        #ax.scatter(grid_points_plot[:, 0], grid_points_plot[:, 1], c='blue', marker='o', label='Grid Points')

        for i in range(self.grid_points.shape[0]):
            ax.plot(self.grid_points[i, :, 0], self.grid_points[i, :, 1], 'blue')

        for j in range(self.grid_points.shape[1]):
            ax.plot(self.grid_points[:, j, 0], self.grid_points[:, j, 1], 'blue')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Vectors and 2D Grid Plot')

        plt.legend()
        plt.show()
    
    def compute_cl_cd_cm(self, rho=1.225, Vinf=1):
        """
        Compute the induced drag coefficient (Cd), lift coefficient (Cl),
        and moment coefficient (Cm) along the span.
        """
        dL = self.vorticity_map_2D * rho * Vinf * self.c # lift per unit span with Kutta-Joukowskui theorem
        Cl = np.sum(dL, axis=1) / (0.5 * rho * Vinf**2 * self.c) # sectional lift coefficient


        Cm = np.zeros(self.N_pannels_spanwise)
        # Assuming aerodynamic center at c/4 for simplicity
        for i in range(self.N_pannels_spanwise):
            Cm[i] =  np.sum(dL[i, :]*(self.c / 4))/ (0.5 * rho * Vinf**2 * self.c)

        """
        # Induced drag calculation
        Gamma = np.sum(self.vorticity_map_2D, axis=1)
        dy = np.diff(self.grid_points[0, :, 1])
        dD = np.zeros_like(Cl)
        for i in range(len(Gamma) + 1):
            dD[i] = (Gamma[i] * Gamma[i+1]) / (4 * np.pi * self.s) * dy[i]
        
        # Sectional drag coefficient (Cd)
        Cd = dD / (0.5 * rho * Vinf**2 * self.c)

        return Cl, Cd, Cm
        """
        return Cl, np.ones_like(Cl), Cm

    def compute_cm_about_point(self, x_cp, rho=1.225, Vinf=1):
        """
        Compute sectional moment coefficient (Cm) about a specified point x_cp along the chord.
        """
        cm = np.zeros(self.N_pannels_spanwise)
        
        for i in range(self.N_pannels_spanwise):
            circulation_sum = np.sum(self.vorticity_map_2D[i, :])
            cm[i] = ((x_cp - 0.25 * self.c) * rho * Vinf * circulation_sum * self.c) / (0.5 * rho * Vinf**2 * self.c**2)
        
        return cm

    def plot_cl_cd_cm(self, Cl, Cd, Cm):
        y_coords = np.linspace(-self.s, self.s, self.N_pannels_spanwise)

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(y_coords, Cl, 'r-')
        plt.xlabel('Spanwise position (y)')
        plt.ylabel('Sectional Lift Coefficient (Cl)')
        plt.title('Cl Distribution')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(y_coords, Cd, 'g-')
        plt.xlabel('Spanwise position (y)')
        plt.ylabel('Sectional Drag Coefficient (Cd)')
        plt.title('Cd Distribution')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(y_coords, Cm, 'b-')
        plt.xlabel('Spanwise position (y)')
        plt.ylabel('Sectional Moment Coefficient (Cm)')
        plt.title('Cm Distribution')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    def plot_lift_distribution(self, Vinf=1):
        X, Y = self.center_points[:, :, 0], self.center_points[:, :, 1]
        Z = self.vorticity_map_2D * (2 * self.c) / Vinf  # Scaling factor for lift coefficient
        fig, ax = plt.subplots(figsize=(12, 8))
        c = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        fig.colorbar(c, ax=ax)

        ax.set_xlabel('Chordwise (x)')
        ax.set_ylabel('Spanwise (y)')
        ax.set_title('Lift Distribution over the Surface')
        
        plt.show()

    def integrate_CL_CD(self, Cl, Cd):
        """
        Integrate the sectional lift coefficient (Cl) and drag coefficient (Cd)
        spanwise to get the total lift coefficient (CL) and total drag coefficient (CD).
        """
        y_coords = self.center_points[:,0,1]
        CL = np.trapz(Cl, y_coords)
        #CD = np.trapz(Cd, y_coords)
        return CL, 1 # CD

#Example usage


model = aero_model(0.451, 0.051)
model.VLM2Dmesh_cos(N_pannels_chordwise=2, N_pannels_spanwise=9, l_wake=0.51)
#model.plot_VLMgrid()
model.VLM()

Cl, Cd, Cm = model.compute_cl_cd_cm()

# Integrate CL and CD
CL, CD = model.integrate_CL_CD(Cl, Cd)
print(f"Total Lift Coefficient (CL): {CL}")
print(f"Total Drag Coefficient (CD): {CD}")
x_cp = 0.3 * model.c  # Example point at 30% chord
# Cm = model.compute_cm_about_point(x_cp)
#model.plot_cl_cd_cm(Cl, Cd, Cm)
#model.plot_lift_distribution()
#model.plot_VLMresults()
