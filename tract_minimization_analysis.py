import numpy as np
from scipy.optimize import minimize_scalar, minimize

# constants
h = 6.62607004 * (1/np.power(10,34, dtype=np.longdouble))    # Plank's
mu_0 = 1.25663706 * (1/np.power(10,6, dtype=np.longdouble))  # vacuum permeability
gamma_H = 267.52218744 * np.power(10,6, dtype=np.longdouble) # proton gyromagnetic ratio
gamma_N = -27.116 * np.power(10,6, dtype=np.longdouble)      # 15N gyromagnetic ratio
r = 1.02 * (1/np.power(10,10, dtype=np.longdouble))          # internuclear distance
delta_dN = 160 * (1/np.power(10, 6))                         # diff in axially symetric 15N CS tensor
theta = 17*np.pi/180                                         # angle between CSA axis and N-H bond

# field in MHz
field = 750

# derived field value in Tesla
B_0 = field * np.power(10, 6, dtype=np.longdouble) * 2 * np.pi / gamma_H # in Tesla

# equation (5)
p = mu_0*gamma_H*gamma_N*h/(16*np.pi*np.pi*np.sqrt(2)*np.power(r,3))     # DD 1H-15N bond

# equation (6)
dN = gamma_N*B_0*delta_dN/(3*np.sqrt(2)) # 15N CSA

# equation (7)
w_N = B_0 * gamma_N                      # 15N frequency (radians/s)

# measured relaxation rates
Rb = 80 #64 # Hz
Ra = 22 #13 # Hz

args = (w_N, Rb, Ra, p, dN, theta)

# Equation 12 function
def objective_function(tau_c, *args):
    
    #spectral density function
    def J(w_N, tau_c):
        return 0.4*tau_c/(1+(w_N**2*tau_c**2))
    #print(*args)
    (w_N, Rb, Ra, p, dN, theta) = args # unpact these constants inside the function
    return np.abs((4*J(0, tau_c) + 3*J(w_N, tau_c)) - ((Rb - Ra)/(2*p*dN*(3*np.cos(theta)**2-1))))

# guess tau_c 
t = 10 * (1/np.power(10,8, dtype=np.longdouble)) # guess 10 ns
res = minimize_scalar(objective_function, args=args, method='Brent')
print(f'Given: field = {field} MHz, Rb = {Rb} Hz and Ra = {Ra} Hz')
print(f'tau_c: {res.x} seconds -- Using Brent Minimization')
print()
res = minimize(objective_function, t, args=args, method='BFGS')
print(f'Given: field = {field} MHz, Rb = {Rb} Hz and Ra = {Ra} Hz')
print(f'tau_c: {res.x} seconds -- Using BFGS Minimization')
print()
res = minimize(objective_function, t, args=args, method='Powell')
print(f'Given: field = {field} MHz, Rb = {Rb} Hz and Ra = {Ra} Hz')
print(f'tau_c: {res.x} seconds -- Using Powell Minimization')
print()
res = minimize(objective_function, t, args=args, method='TNC')
print(f'Given: field = {field} MHz, Rb = {Rb} Hz and Ra = {Ra} Hz')
print(f'tau_c: {res.x} seconds -- Using TNC Minimization')
print()