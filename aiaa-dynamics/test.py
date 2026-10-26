import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 1. Define Physical Constants and System Parameters ---
# Gravitational Constant (m^3 kg^-1 s^-2)
G = 6.67430e-11 
# Mass of Earth (kg)
M_EARTH = 5.972e24 
# Standard Gravitational Parameter (mu = G * M)
MU_EARTH = G * M_EARTH

# --- 2. Define the Equations of Motion (The Physics Engine) ---
def ode_system(t, Y):
    """
    Defines the system of first-order differential equations (ODEs).
    State vector Y: [x, y, vx, vy]
    Returns derivatives dY/dt: [vx, vy, ax, ay]
    """
    x, y, vx, vy = Y

    # Calculate distance from the center
    r = np.sqrt(x**2 + y**2)

    # Avoid division by zero
    if r == 0:
        return np.zeros_like(Y)

    # Calculate acceleration: a = (-mu / r^3) * r_vector
    accel_factor = -MU_EARTH / (r**3)
    ax = accel_factor * x
    ay = accel_factor * y

    return [vx, vy, ax, ay]

# --- 3. Define Initial Conditions and Time Span ---
# Initial altitude: 400 km (Low Earth Orbit)
ALTITUDE_M = 400e3 
R_EARTH = 6371e3 
R0 = R_EARTH + ALTITUDE_M 

# Initial position: [R0, 0]
x0, y0 = R0, 0 

# Required velocity for circular orbit: V = sqrt(mu / r)
V_circular = np.sqrt(MU_EARTH / R0)

# Initial velocity: [0, V_circular] (Tangential)
vx0, vy0 = 0, V_circular
Y0 = [x0, y0, vx0, vy0]

# Time span: Calculate orbital period to simulate 2 full orbits
T_PERIOD = 2 * np.pi * np.sqrt(R0**3 / MU_EARTH)
T_SPAN = [0, 2 * T_PERIOD]

# --- 4. Perform Numerical Integration (Runge-Kutta) ---
# solve_ivp uses RK45 (Runge-Kutta 4/5) by default
solution = solve_ivp(
    fun=ode_system,
    t_span=T_SPAN,
    y0=Y0,
    rtol=1e-12, # High precision
    atol=1e-12,
    dense_output=True
)

# --- 5. Visualize the Orbit ---
x_traj = solution.y[0, :]
y_traj = solution.y[1, :]

plt.figure(figsize=(8, 8))
plt.plot(x_traj / 1e3, y_traj / 1e3, label='Satellite Trajectory')
plt.plot(0, 0, 'o', color='blue', markersize=10, label='Earth')
plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.title('2D Circular Orbit Simulation (Two-Body Problem)')
plt.axis('equal')
plt.grid(True)
plt.show()

print(f"Simulation Complete.")
print(f"Orbital Period: {T_PERIOD/60:.2f} minutes")
