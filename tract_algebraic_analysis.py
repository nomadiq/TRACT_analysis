import numpy as np

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
dN = gamma_N*B_0*delta_dN/(3*np.sqrt(2))                                 # 15N CSA

# equation (7)
w_N = B_0 * gamma_N                                                      # 15N frequency (radians/s)

# measured relaxation rates
# Panel A Figure 4
Rb = 64  # Hz
Ra = 13  # Hz

#Panel B Figure 4
Rb = 80  # Hz
Ra = 22  # Hz

# from equivalence between equation (8) RHS and equation (9) RHS
c = (Rb - Ra)/(2*dN*p*(3*np.cos(theta)**2-1))

# function that returns evaluation of equation (10) with w_N and constant 'c' above as inputs
def tc(w_N, c):
    
    t1 = (5*c)/24 
    t2 = (336*(w_N**2) - 25*(c**2)*(w_N**4)) / (24*(w_N**2) * (1800*c*(w_N**4) + 125*(c**3)*(w_N**6) + 24*np.sqrt(3)*np.sqrt(21952*(w_N**6) - 3025*(c**2)*(w_N**8) + 625*(c**4)*(w_N**10)))**(1/3))
    t3 = (1800*c*(w_N**4) + 125*(c**3)*(w_N**6) + 24*np.sqrt(3)*np.sqrt(21952*(w_N**6) - 3025*(c**2)*(w_N**8) + 625*(c**4)*(w_N**10)))**(1/3)/(24*w_N**2) 
    
    return t1 - t2 + t3

# computation of result from panel A of figure 4 in Lee et al. 
tau_c = tc(w_N, c)
print(f'Given: field = {field} MHz, Rb = {Rb} Hz and Ra = {Ra} Hz')
print(f'tau_c: {tau_c} seconds -- using algebraic solution')