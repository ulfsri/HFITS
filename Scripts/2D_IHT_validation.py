import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
stefan_boltzmann_constant = 5.67e-8 # (W/m^2K^4)
initial_temperature = 293 # Initial temperature of the plate sensor (K)

# User-defined variables
emission_time = 20 * 60  # Exposure time (s)
wall_thickness = 0.001 # Thickness of the plate sensor (m)
element_width = 0.05 # Size of a discrete element in the x dimension (m)
element_height = 0.10 # Size of a discrete element in the y dimension (m)
emissivity = 0.94 # Emissivity of the plate sensor
steel_density = 8050 # Density of the plate sensor material (kg/m^3)
cp_steel = 490 # Specific heat capacity of the plate sensor material (J/kgK)
k_steel = 45  # Thermal conductivity of the plate sensor material (W/mK)
convective_hfc_exposed = 10 # Convective heat transfer coefficient on the exposed side of the plate sensor (W/m^2K)
convective_hfc_unexposed = 8 # Convective heat transfer coefficient on the unexposed side of the plate sensor (W/m^2K)

# Initializing grid heat flux for the forward model
grid_size = 3  # 3x3 grid --> the simplest 2D case possible
central_heat_flux = 50000  # Central cell
surrounding_heat_flux = 10000  # neighboring cells
heat_flux_grid = np.full((grid_size, grid_size), surrounding_heat_flux)
heat_flux_grid[1, 1] = central_heat_flux  # Set central cell heat flux

# Apply conservation of energy over the 2D grid
def energy_balance_2d(t, T_flat):
    T = T_flat.reshape((grid_size, grid_size))
    dTdt = np.zeros_like(T)
    for i in range(grid_size):
        for j in range(grid_size):
            # conduction in 2D
            q_cond = 0
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    if abs(ni-i) == 1 and abs(nj-j) == 1:
                        q_cond = 0 
                    edge_length = element_height if di != 0 else element_width
                    path_length = element_height if di == 0 else element_width
                    q_cond += k_steel * (T[ni, nj] - T[i, j]) * edge_length * wall_thickness / path_length

            # Heat flux contribution
            area = element_width * element_height
            q_in = heat_flux_grid[i, j] * area if t < emission_time else 0
            q_conv_exp = convective_hfc_exposed * area * (T[i, j] - initial_temperature)
            q_conv_unexp = convective_hfc_unexposed * area * (T[i, j] - initial_temperature)
            q_rad = 2*emissivity * stefan_boltzmann_constant * area * (T[i, j]**4 - initial_temperature**4)
            # print(q_cond)
            # Energy balance
            dTdt[i, j] = (emissivity * q_in - q_conv_exp - q_conv_unexp - q_rad + q_cond) / (area * cp_steel * steel_density * wall_thickness)

    return dTdt.ravel()

# Initial condition for the 2D grid
T0_flat = np.full(grid_size**2, initial_temperature)

# Time duration 
t_span = (0, emission_time)

# Generate time steps
t_eval = np.linspace(t_span[0], t_span[1],6000)  # Creates an array from 0 to emission_time with a specified time step
# print(t_eval)

# Solve the 2D differential equation with specified time steps
solution = solve_ivp(energy_balance_2d, t_span, T0_flat, method='RK45', t_eval=t_eval, dense_output=True)
num_time_steps = len(solution.t)
# print((solution.y).shape)
T_values_2d = solution.y.reshape((grid_size, grid_size, num_time_steps))
# print(T_values_2d.shape)

# Plotting temperature distribution over time
fig , ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(14, 8) )
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
k = 0
for i in range(grid_size):
    for j in range(grid_size):
        axes[k].plot(solution.t / 60, T_values_2d[i, j, :], label=f'Cell ({i},{j})')
        axes[k].set_xscale('log')
        axes[k].set_ylabel('T (K)')
        # axes[k].set_title(f'Element ({i},{j})')
        axxx = axes[k].twinx()
        plt.axhline((heat_flux_grid[i , j])/1000, label='Incident Heat Flux kW/m2', color='red' )
        axxx.set_ylabel('Inc. HF (kW/m2)')
        axxx.legend(loc='upper right')

        k += 1
plt.tight_layout()

# IHT Model Function
def inverse_heat_transfer(T_values_2d, solution, k_steel, cp_steel, steel_density, element_width, element_height, wall_thickness, convective_hfc_exposed, convective_hfc_unexposed, emissivity):
    num_time_steps = T_values_2d.shape[2]
    # print(num_time_steps)
    estimated_flux = np.zeros((grid_size, grid_size, num_time_steps))
    
    temp_grad = np.gradient(T_values_2d, axis=2) / np.gradient(solution.t)
    area = element_height * element_width
    for time_step in range(num_time_steps):
        for i in range(grid_size):
            for j in range(grid_size):
                grad_T = np.zeros(2)  # [grad_T_x, grad_T_y]
                if i > 0:
                    grad_T[0] += k_steel *(-T_values_2d[i, j, time_step] + T_values_2d[i-1, j, time_step]) * element_height / element_width
                if i < grid_size - 1:
                    grad_T[0] += k_steel *(T_values_2d[i+1, j, time_step] - T_values_2d[i, j, time_step]) * element_height / element_width
                if j > 0:
                    grad_T[1] += k_steel *(-T_values_2d[i, j, time_step] + T_values_2d[i, j-1, time_step]) * element_width / element_height
                if j < grid_size - 1:
                    grad_T[1] += k_steel *(T_values_2d[i, j+1, time_step] - T_values_2d[i, j, time_step]) * element_width / element_height

                # Calculate heat flux
                q_cond =  np.sum(grad_T) * wall_thickness
                q_conv = (convective_hfc_exposed + convective_hfc_unexposed) * area * (T_values_2d[i, j, time_step] - initial_temperature)
                q_rad = 2*emissivity * stefan_boltzmann_constant * area * (T_values_2d[i, j, time_step]**4 - initial_temperature**4)
                q_storage = cp_steel * steel_density * area * wall_thickness * temp_grad[i, j, time_step]

                # Calculate the estimated incident heat flux
                estimated_flux[i, j, time_step] = (q_storage + q_conv + q_rad - q_cond)  / (area * emissivity)

    return estimated_flux

# Calculate the estimated heat flux using the inverse model
estimated_heat_flux = inverse_heat_transfer(T_values_2d, solution, k_steel, cp_steel, steel_density, element_width, element_height, wall_thickness, convective_hfc_exposed, convective_hfc_unexposed, emissivity)

#Plotts
fig , ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(14, 8) , sharex = True, sharey=True)
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
k = 0
for i in range(grid_size):
    for j in range(grid_size):
        ax = axes[k]
        ax.plot(solution.t / 60, estimated_heat_flux[i, j, :])
        # ax.plot(T_values_2d[i, j, :])
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Heat Flux (W/m²)')
        ax.set_title(f'Element ({i},{j})')
        ax.set_ylim([0,70000])
        ax.legend()

        # Create a secondary y-axis for temperature
        axx = ax.twinx()
        axx.plot(solution.t / 60, T_values_2d[i, j, :], label='Temperature', color='red')
        axx.set_ylabel('Temperature (K)')
        axx.legend(loc='upper right')
        # ax.grid(True)
        k += 1
# print(T_values_2d[1, 1, :])
plt.tight_layout()
plt.show()