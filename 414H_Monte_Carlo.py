import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#setting random seed:

rng = np.random.default_rng(seed=99)

#Physical Parameters:
m_e = 9.109*10**-31           #electron mass (kg)
e = -1.60217663*10**-19       #electron charge (C)
E = 1e6                       #Electric Field (N/C)
v_th = 1.57e6                 #Thermal velocity of electron (m/s)
lam = 3.9e-8                  #Mean free path (m)
tau = lam/v_th                #Mean free time (s)

a = -e*E/m_e                  #Accleration due to electric field

#Initial velocity (|v|=v_therm)
v_0 = rng.normal(0,1,3)
v = v_th * v_0 /np.linalg.norm(v_0)

#Free flight acceleration
@njit
def free_flight(v,a, t):
    v[0] += a*t
    return v


#Rotation due to scattering
@njit
def apply_rotation(v, cos_theta, phi):

    speed = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    u = v.flatten() /speed

    #phi RV, uniform on (0,2pi)
    theta = np.arccos(cos_theta)     #theta calculated from cos(theta) RV so that cos(theta) is not skewed

    if abs(u[0]) >= 0.9:             #Build orthonormal basis to apply rotation
        r = np.array([0.0,1.0,0.0])
    else:
        r = np.array([1.0,0.0,0.0])

    dot_ru = r[0]*u[0] + r[1]*u[1] + r[2]*u[2]
    p = r - dot_ru * u
    size_p = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    p = p/size_p
    q = np.cross(u,p)
    
    n = np.sin(theta)*np.cos(phi)*p + np.sin(theta)*np.sin(phi)*q + cos_theta*u
    v = speed * n

    return v


#Proabilistic Parameters & Intializing
M = 1000
N = 10**5

x_results = np.zeros(M)
y_results = np.zeros(M)
z_results = np.zeros(M)

#Simulation
@njit
def Single_Particle_MC(N, v, ts, cos_thetas, phis, a):
    sum_vx, sum_vy, sum_vz = 0.0, 0.0, 0.0
    for i in range(N):
        # mean velocity during this flight in x is v[0] + 0.5*a*ts[i]
        sum_vx += v[0] + 0.5 * a * ts[i]
        # in y and z there is no acceleration, so mean velocity is just v[1], v[2]
        sum_vy += v[1]
        sum_vz += v[2]

        # advance velocity to end of flight and scatter
        v = free_flight(v, a, ts[i])
        v = apply_rotation(v, cos_thetas[i], phis[i])

    return [sum_vx / N, sum_vy / N, sum_vz / N]

for j in range(M):
    v_0 = rng.random(3)
    v = v_th * v_0 / np.linalg.norm(v_0)
    cos_thetas = rng.uniform(-1, 1, size = N)
    phis = rng. uniform(0, 2*np.pi, size=N)
    ts = rng.exponential(tau, size=N)
    x_results[j], y_results[j], z_results[j] = Single_Particle_MC(N, v, ts, cos_thetas, phis, a)

print(f'\nMean drift velocity in x: {np.mean(x_results):.4f} m/s')
print(f'Std dev in x: {np.std(x_results):.4f} m/s')
print(f'Analytical: {a*tau:.4f} m/s\n')

print(f'\nMean drift velocity in y: {np.mean(y_results):.4f} m/s')
print(f'Std dev in y: {np.std(y_results):.4f} m/s')

print(f'\nMean drift velocity in z: {np.mean(z_results):.4f} m/s')
print(f'Std dev in z: {np.std(z_results):.4f} m/s')



#Creating Gaussian approximations of each data

plt.figure(figsize=(12,5))

# Subplot 1: V_x Distribution
plt.subplot(1,3,1)
plt.hist(x_results, bins=30, color='red', edgecolor='black')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.xlabel('Speed (m/s)')
plt.ylabel('Number of simulations')
plt.title('V_x Distribution')

# Subplot 2: V_y Distribution
plt.subplot(1,3,2)
plt.hist(y_results, bins=30, color='blue',  edgecolor='black')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.xlabel('Speed (m/s)')
plt.ylabel('Number of simulations')
plt.title('V_y Distribution')

# Subplot 3: V_z Distribution
plt.subplot(1,3,3)
plt.hist(z_results, bins=30, color='green', edgecolor='black')
plt.grid(True)
plt.gca().set_axisbelow(True)
plt.xlabel('Speed (m/s)')
plt.ylabel('Number of simulations')
plt.title('V_z Distribution')


def gaussian(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2/(2*sigma**2))

# Compute means and stds
mx, sx = np.mean(x_results), np.std(x_results)
my, sy = np.mean(y_results), np.std(y_results)
mz, sz = np.mean(z_results), np.std(z_results)

# Create smooth x-axis for Gaussian curves
x_axis = np.linspace(mx-3*sx, mx+3*sx, 500)
y_axis = np.linspace(my-3*sy, my+3*sy, 500)
z_axis = np.linspace(mz-3*sz, mz+3*sz, 500)

# Evaluate Gaussian PDFs
gauss_x = gaussian(x_axis, mx, sx)
gauss_y = gaussian(y_axis, my, sy)
gauss_z = gaussian(z_axis, mz, sz)

# Scale PDFs to match histogram height
gauss_x *= len(x_results) * ( (np.max(x_results)-np.min(x_results)) / 30 )
gauss_y *= len(y_results) * ( (np.max(y_results)-np.min(y_results)) / 30 )
gauss_z *= len(z_results) * ( (np.max(z_results)-np.min(z_results)) / 30 )


plt.subplot(1,3,1)
plt.plot(x_axis, gauss_x, 'k', linewidth=2, label='Gaussian Fit')
plt.legend()


plt.subplot(1,3,2)
plt.plot(y_axis, gauss_y, 'k', linewidth=2, label='Gaussian Fit')
plt.legend()

plt.subplot(1,3,3)
plt.plot(z_axis, gauss_z, 'k', linewidth=2, label='Gaussian Fit')
plt.legend()

plt.tight_layout() 
plt.show()

import pandas as pd
import seaborn as sns

# Prepare data for bar plot
data = {
    'Direction': ['X-Drift', 'Y-Drift', 'Z-Drift'],
    'Mean Velocity (m/s)': [mx, my, mz],
    'Standard Deviation (m/s)': [sx, sy, sz]
}
df_summary = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
barplot = sns.barplot(
    x='Direction',
    y='Mean Velocity (m/s)',
    data=df_summary,
    hue='Direction', # Assign x to hue as suggested by warning
    palette='viridis',
    legend=False # Set legend to False as suggested by warning
)

# Error Bars
x_coords = [p.get_x() + p.get_width() / 2 for p in barplot.patches]
plt.errorbar(x_coords, df_summary['Mean Velocity (m/s)'],
             yerr=df_summary['Standard Deviation (m/s)'],
             fmt='none', c='black', capsize=5)

plt.title('Average Drift Velocity in X, Y, and Z Directions')
plt.ylabel('Mean Velocity (m/s)')
plt.xlabel('Direction')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for index, row in df_summary.iterrows():
    barplot.text(
        index,
        row['Mean Velocity (m/s)'],
        f'{row["Mean Velocity (m/s)"]:.2f}',
        color='black', ha="center", va='bottom'
        )

plt.tight_layout()
plt.show()





