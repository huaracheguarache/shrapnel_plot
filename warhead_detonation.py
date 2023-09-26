import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Parameters
height = 5  # Height of the warhead above the ground [m].
warhead_speed = 300  # Warhead speed in the direction of movement of the missile [ms-1].
angle = 30  # Angle relative to the x-axis (ground) [degrees].
shrapnel_speed = 2520  # Lateral/radial speed of shrapnel [ms-1].
N_shrapnel = 360  # Number of shrapnel particles in the simulation.
timesteps = 5000  # Number of timesteps in the simulation.
duration = 0.01  # Duration of the simulation [s].
street_width = 8  # Width of the street (used in 2D-plot) [m].

angle_rad = np.deg2rad(angle)
warhead_velocity = np.array((-np.cos(angle_rad) * warhead_speed, -np.sin(angle_rad) * warhead_speed, 0))

# Start out with shrapnel velocities in the y-z-plane.
full_circle = 360  # [degrees]
shrapnel_angles = np.deg2rad(np.arange(0, full_circle, full_circle / N_shrapnel))
shrapnel_velocity = np.zeros((N_shrapnel, 3))

# Calculate the shrapnel velocities component-wise.
for i, angle in enumerate(shrapnel_angles):
    shrapnel_velocity[i, :] = (0, np.cos(angle) * shrapnel_speed, np.sin(angle) * shrapnel_speed)

# Rotate the velocities so that they are perpendicular to the line the missile travels along.
R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0], [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]])
for i in range(N_shrapnel):
    # Also add the warhead velocity to get total velocity for the shrapnel.
    shrapnel_velocity[i, :] = np.matmul(R, shrapnel_velocity[i, :]) + warhead_velocity

# Simulating the movement of the shrapnel.
shrapnel_positions = np.zeros((N_shrapnel, 3, timesteps))
shrapnel_positions[:, 1, 0] = height
t, dt = np.linspace(0, duration, timesteps - 1, retstep=True)

for i, _ in enumerate(t):
    shrapnel_positions[:, :, i + 1] = shrapnel_positions[:, :, i] + shrapnel_velocity * dt
    # Check which shrapnel particles have gone through the x-z-plane (ground) and set their velocity to zero.
    escaped_index = np.argwhere(shrapnel_positions[:, 1, i + 1] <= 0)
    if escaped_index.any():
        for index in escaped_index:
            shrapnel_velocity[index[0]] = [0, 0, 0]

# Plot a 3D-plot of the shrapnel fan.
ax = plt.figure().add_subplot(projection='3d')
for i in range(N_shrapnel):
    ax.plot(shrapnel_positions[i, 0, :],
            shrapnel_positions[i, 1, :],
            shrapnel_positions[i, 2, :],
            color='g',
            linewidth=0.5,
            alpha=0.5)

# Also plot the impact points.
ax.scatter(shrapnel_positions[escaped_index, 0, -1],
           shrapnel_positions[escaped_index, 1, -1],
           shrapnel_positions[escaped_index, 2, -1],
           marker='s',
           s=2)

ax.set_title('Shrapnel fan and impacts on ground')
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10, offset=0))
ax.set_xlabel('Length [m]')
ax.yaxis.set_major_locator(ticker.MultipleLocator(base=10, offset=0))
ax.set_ylabel('Height [m]')
ax.zaxis.set_major_locator(ticker.MultipleLocator(base=10, offset=0))
ax.set_zlabel('Width [m]')

# Make sure the aspect ratio is set to equal so that the shrapnel fan looks like a proper circle.
ax.set_aspect('equal')

# Set the initial viewing angle and show the plot.
ax.view_init(140, -10, 90)
plt.show()

# A 2D-plot of the resulting shrapnel pattern on the ground.
fig, ax = plt.subplots(layout="constrained")
# Mask for only plotting the width of the street.
street_mask = np.abs(shrapnel_positions[escaped_index, 2, -1]) <= street_width / 2
ax.plot(shrapnel_positions[escaped_index, 2, -1][street_mask],
        shrapnel_positions[escaped_index, 0, -1][street_mask])
fig.suptitle('Shrapnel pattern on the ground seen from above', fontsize=15)
ax.set_title('Missile traveling in negative y-direction, \nzero of x-axis in the middle of the street.', fontsize=11)
ax.set_xlabel('Width [m]')
ax.set_ylabel('Length [m]')
plt.show()
