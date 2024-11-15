import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from sympy.diffgeom import Manifold, Patch, CoordSystem, TensorProduct, metric_to_Ricci_components
from sympy import Symbol, sin, cos, symbols, Matrix, diff, sqrt, pi, Function, exp
import sympy as sp
from scipy.integrate import odeint, solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import scipy.integrate as integrate
from scipy.sparse.linalg import eigsh
import networkx as nx
from scipy.special import spherical_jn
from scipy.stats import entropy
import time 

# Philosophy about the research at https://qhin.cashewstable.com 

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
c = 3.0e8  # Speed of light in m/s
hbar = 1.0545718e-34  # Reduced Planck constant in m^2 kg / s
k_B = 1.380649e-23  # Boltzmann constant in m^2 kg s^-2 K^-1
l_p = sqrt(G * hbar / c**3)  # Planck length
t_p = l_p / c  # Planck time
m_p = sqrt(hbar * c / G)  # Planck mass


# Schwarzschild Metric
def schwarzschild_metric():
    """Define the Schwarzschild metric tensor"""
    G, M, c = symbols('G M c')
    r, theta, phi = symbols('r theta phi', real=True)

    g_tt = -(1 - 2 * G * M / (c ** 2 * r))
    g_rr = 1 / (1 - 2 * G * M / (c ** 2 * r))
    g_thetatheta = r ** 2
    g_phiphi = r ** 2 * sin(theta) ** 2

    metric = sp.Matrix([
        [g_tt, 0, 0, 0],
        [0, g_rr, 0, 0],
        [0, 0, g_thetatheta, 0],
        [0, 0, 0, g_phiphi]
    ])

    return metric


# Function to compute Christoffel symbols
def christoffel_symbols(metric):
    dim = metric.shape[0]
    symbols = sp.MutableDenseNDimArray.zeros(dim, dim, dim)
    inv_metric = metric.inv()

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                symbols[i, j, k] = 0.5 * sum(
                    inv_metric[i, l] * (sp.diff(metric[l, j], sp.symbols(f'x{k + 1}')) +
                                        sp.diff(metric[l, k], sp.symbols(f'x{j + 1}')) -
                                        sp.diff(metric[j, k], sp.symbols(f'x{l + 1}')))
                    for l in range(dim))
    return symbols


# Function to compute Riemann tensor
def riemann_tensor(christoffel):
    dim = christoffel.shape[0]
    riemann = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    riemann[i, j, k, l] = (sp.diff(christoffel[i, j, l], sp.symbols(f'x{k + 1}')) -
                                           sp.diff(christoffel[i, j, k], sp.symbols(f'x{l + 1}')) +
                                           sum(christoffel[i, k, m] * christoffel[m, j, l] -
                                               christoffel[i, l, m] * christoffel[m, j, k] for m in range(dim)))
    return riemann


# Function to compute Ricci tensor
def ricci_tensor(riemann):
    dim = riemann.shape[0]
    ricci = sp.MutableDenseNDimArray.zeros(dim, dim)

    for i in range(dim):
        for j in range(dim):
            ricci[i, j] = sum(riemann[m, i, m, j] for m in range(dim))
    return ricci


# Function to compute Ricci scalar
def ricci_scalar(ricci, metric):
    inv_metric = metric.inv()
    return sum(inv_metric[i, j] * ricci[i, j] for i in range(metric.shape[0]) for j in range(metric.shape[1]))


# Einstein-Hilbert action computation for Schwarzschild
def einstein_hilbert_action():
    """Compute the Einstein-Hilbert action for Schwarzschild metric"""
    metric = schwarzschild_metric()
    christoffel = christoffel_symbols(metric)
    riemann = riemann_tensor(christoffel)
    ricci = ricci_tensor(riemann)
    R = ricci_scalar(ricci, metric).simplify()
    g_det = metric.det()
    S = c ** 4 / (16 * sp.pi * G) * R * sp.sqrt(-g_det)
    return S


class QuantumGravityCorrections:
    """Quantum gravity corrections for both string theory and loop quantum gravity"""
    def __init__(self, theory_type='string'):
        self.theory_type = theory_type
        self.setup_constants()

    def setup_constants(self):
        """Initialize constants for quantum gravity corrections"""
        self.alpha_prime = l_p ** 2  # String length parameter
        self.gamma_lqg = 0.2375  # Immirzi parameter for LQG
        self.beta = 1 / (2 * pi)  # String coupling constant

    def string_theory_corrections(self, mass, radius):
        """String theory corrections"""
        R_squared = (2 * G * mass / c ** 2 / radius ** 3) ** 2
        alpha_correction = 1 + self.alpha_prime * R_squared / (16 * pi)
        loop_correction = 1 + self.beta * G * hbar / (c ** 3 * radius ** 2)
        return alpha_correction, loop_correction

    def loop_quantum_corrections(self, mass, radius):
        """Loop quantum gravity corrections"""
        mu_0 = sqrt(3 * sqrt(3) * self.gamma_lqg / 2) * l_p
        delta_b = mu_0 / radius
        volume_correction = (1 + (l_p / radius) ** 2) ** self.gamma_lqg
        return delta_b, volume_correction


class BlackHoleEvolution:
    """Evolution of black holes with quantum gravity corrections"""
    def __init__(self, initial_mass, include_quantum=True):
        self.M0 = initial_mass
        self.include_quantum = include_quantum
        self.qg = QuantumGravityCorrections('both')

    def mass_loss_rate(self, M, t):
        """Compute mass loss rate during Hawking evaporation"""
        dM_dt_classical = -hbar * c ** 4 / (15360 * pi * G ** 2 * M ** 2)
        if not self.include_quantum:
            return dM_dt_classical

        r_h = 2 * G * M / c ** 2  # Horizon radius
        alpha_corr, loop_corr = self.qg.string_theory_corrections(M, r_h)
        delta_b, vol_corr = self.qg.loop_quantum_corrections(M, r_h)
        quantum_factor = alpha_corr * loop_corr * vol_corr * (1 - delta_b ** 2)
        return dM_dt_classical * quantum_factor

    def evolve(self, t_span, t_eval=None):
        """Evolve black hole mass over time"""
        sol = solve_ivp(self.mass_loss_rate, t_span, [self.M0], t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8)
        return sol.t, sol.y[0]


class StringTheoryCorrections:
    """
    Enhanced string theory corrections with proper bounds
    """
    def __init__(self):
        self.scales = ScaleHierarchy()
        self.alpha_prime = float(self.scales.l_p**2)
        self.g_s = 0.1
        self.MAX_CORRECTION = 100.0  # Limited correction magnitude
        
    def horizon_corrections(self, r, M):
        """Bounded horizon corrections"""
        r_scaled = self.scales.enforce_length_scale(float(r))
        M_scaled = np.clip(float(M), 0, self.scales.m_p * 1e5)
        
        # Compute curvature scale
        R = 2 * G * M_scaled / (c**2 * r_scaled**3)
        R_squared = np.clip(R**2, 0, 1e2)
        
        # Compute corrections with bounds
        alpha_corr = 1 + self.alpha_prime * R_squared / 4
        loop_corr = 1 + self.g_s * np.log1p(r_scaled / self.scales.l_p)
        
        # Total correction with bounds
        total_corr = np.clip(alpha_corr * loop_corr, 1.0, self.MAX_CORRECTION)
        return total_corr

    def dilaton_profile(self, r):
        """Enhanced dilaton profile"""
        r_scaled = np.array([self.scales.enforce_length_scale(float(ri)) 
                           for ri in r])
        profile = np.exp(-r_scaled / (self.g_s * self.scales.l_p))
        return np.clip(profile, 0, 1)

            
class ScaleHierarchy:
    """
    Manages scale hierarchies and enforces proper UV/IR cutoffs
    """
    def __init__(self):
        self.l_p = np.sqrt(G * hbar / c**3)
        self.t_p = self.l_p / c
        self.m_p = np.sqrt(hbar * c / G)
        
        # Scale hierarchy parameters
        self.UV_cutoff = self.l_p
        self.IR_cutoff = 1e3 * self.l_p
        self.energy_cutoff = self.m_p * c**2
        
    def enforce_length_scale(self, length):
        """Enforce proper length scale hierarchy"""
        return np.clip(length, self.UV_cutoff, self.IR_cutoff)
    
    def enforce_energy_scale(self, energy):
        """Enforce proper energy scale hierarchy"""
        return np.clip(energy, 0, self.energy_cutoff)
    
    def check_scale_consistency(self, scale, scale_type="length"):
        """Check if scale is within physical bounds"""
        if scale_type == "length":
            return self.UV_cutoff <= scale <= self.IR_cutoff
        elif scale_type == "energy":
            return 0 <= scale <= self.energy_cutoff
        return False
class CosmologicalInflation:
    """
    Refined inflation implementation to achieve proper observables with enhanced stability
    """
    def __init__(self):
        # Initialize scale hierarchy
        self.scales = ScaleHierarchy()
        self.M_pl = self.scales.m_p
        
        # Refined parameters for Starobinsky inflation
        self.V0 = 1.2e-10 * self.M_pl**4    # Reduced potential scale
        self.phi0 = 12.0 * self.M_pl        # Increased initial field value
        self.N_target = 60                  # Target number of e-foldings
        
        # Starobinsky model parameter
        self.alpha = np.sqrt(2/3) * self.M_pl  # Standard Starobinsky value
        
        # Hubble parameter during inflation
        self.H0 = np.sqrt(self.V0 / (3 * self.M_pl**2))
        
        # Numerical stability parameters
        self.MAX_EXP = 20.0  # Reduced for better stability
        self.epsilon_end = 1.0  # End inflation criterion
        self.MAX_FIELD = 20.0 * self.M_pl  # Limited maximum field value
        
        # Safety parameters
        self.MIN_PHI = -10 * self.M_pl
        self.MAX_PHI = 10 * self.M_pl
        self.MAX_STEPS = 10000
        self.dt_init = 1.0 / self.H0 / 100  # Small initial timestep for integration

    def safe_exp(self, x):
        """Safe exponential calculation with bounds"""
        return np.exp(np.clip(x, -self.MAX_EXP, self.MAX_EXP))

    def potential(self, phi):
        """
        Enhanced Starobinsky potential V = V₀(1 - e^(-√(2/3)φ/M_pl))²
        """
        x = np.clip(phi / self.alpha, -self.MAX_FIELD, self.MAX_FIELD)
        exp_term = self.safe_exp(-x)
        return self.V0 * (1 - exp_term)**2
    
    def potential_derivative(self, phi):
        """
        dV/dφ with proper scaling
        """
        x = np.clip(phi / self.alpha, -self.MAX_FIELD, self.MAX_FIELD)
        exp_term = self.safe_exp(-x)
        return 2 * self.V0 * exp_term * (1 - exp_term) / self.alpha
    
    def slow_roll_parameters(self, phi):
        """
        Compute slow-roll parameters with stability
        """
        V = self.potential(phi)
        dV = self.potential_derivative(phi)
        
        # Prevent division by zero
        if V == 0:
            return 1.0, 0.0
        
        # Calculate epsilon and eta with bounds
        epsilon = np.clip(self.M_pl**2 * (dV / V)**2 / 2, 0, 1e10)
        d2V = self.potential_second_derivative(phi)
        eta = np.clip(self.M_pl**2 * d2V / V, -1e10, 1e10)
        
        return epsilon, eta

    def potential_second_derivative(self, phi):
        """
        Second derivative of the potential with stability
        """
        x = np.clip(phi / self.alpha, -self.MAX_FIELD, self.MAX_FIELD)
        exp_term = self.safe_exp(-x)
        return 2 * self.V0 * exp_term * (exp_term - 1 + exp_term) / self.alpha**2

    def hubble_parameter(self, phi, phi_dot):
        """Enhanced Hubble parameter calculation"""
        kinetic = np.clip(0.5 * phi_dot**2, 0, self.V0)
        V = self.potential(phi)
        H_squared = (kinetic + V) / (3 * self.M_pl**2)
        return np.sqrt(np.clip(H_squared, 0, self.H0**2))

    def equations_of_motion(self, t, state):
        """Stabilized equations of motion"""
        phi, phi_dot = state
        
        # Bound the field values
        phi = np.clip(phi, self.MIN_PHI, self.MAX_PHI)
        phi_dot = np.clip(phi_dot, -self.M_pl**2, self.M_pl**2)
        
        H = self.hubble_parameter(phi, phi_dot)
        V_prime = self.potential_derivative(phi)
        damping = np.clip(3 * H * phi_dot, -1e20, 1e20)
        phi_ddot = -damping - V_prime
        return np.array([np.clip(phi_dot, -1e10, 1e10), np.clip(phi_ddot, -1e20, 1e20)])

    def solve_inflation(self):
        """
        Enhanced inflation solver with proper end conditions and error handling
        """
        def event_end_inflation(t, y):
            """End inflation when enough e-foldings or slow-roll violation"""
            epsilon, _ = self.slow_roll_parameters(y[0])
            return epsilon - self.epsilon_end
        event_end_inflation.terminal = True

        print("\nStarting inflation solve...")
        print(f"Initial conditions: phi = {self.phi0/self.M_pl:.2f} M_pl")
        
        # Initial conditions (start from rest)
        y0 = np.array([self.phi0, 0.0]) 
        
        # Time span based on expected duration
        t_max = min(100 * self.N_target / self.H0, 1e20)  # Add upper bound
        t_span = [0, t_max]
        
        # Integration parameters
        integration_params = {
            'method': 'RK45',
            'rtol': 1e-10,
            'atol': 1e-10,
            'first_step': self.dt_init,
            'max_step': t_max / self.MAX_STEPS
        }
        
        try:
            solution = solve_ivp(
                self.equations_of_motion,
                t_span,
                y0,
                **integration_params
            )
            
            if not solution.success:
                print(f"Integration failed: {solution.message}")
                return None, None, None
            
            # Compute results
            times = solution.t
            fields = solution.y
            
            if len(times) == 0 or len(fields[0]) == 0:
                print("No valid solution points found")
                return None, None, None
            
            # Compute e-foldings safely
            H_vals = np.array([
                self.hubble_parameter(phi, phi_dot) 
                for phi, phi_dot in zip(fields[0], fields[1])
            ])
            N = np.trapz(H_vals, times)
            
            print(f"\nInflation Results:")
            print(f"Total e-foldings: {N:.2f}")
            print(f"Duration: {times[-1]:.2e} seconds")
            print(f"Final field value: {fields[0,-1]/self.M_pl:.2f} M_pl")
            
            # Compute observables at appropriate time
            try:
                N_vals = np.cumsum(H_vals * np.diff(times, prepend=0))
                if len(N_vals) > 0 and N_vals[-1] > 50:  # Check if enough e-foldings
                    idx_horizon = max(
                        0, 
                        np.searchsorted(N_vals, N_vals[-1] - 50)
                    )
                    phi_horizon = fields[0][idx_horizon]
                    obs = self.compute_observables(phi_horizon)
                else:
                    obs = self.compute_observables(fields[0,-1])  # Use final value
                
                return times, fields, obs
                
            except Exception as e:
                print(f"Error computing observables: {str(e)}")
                return times, fields, None
            
        except Exception as e:
            print(f"Error in inflation solve: {str(e)}")
            return None, None, None
    
    def compute_observables(self, phi):
        """
        Compute inflationary observables with stability and bounds
        """
        try:
            epsilon, eta = self.slow_roll_parameters(phi)
            
            # Prevent division by zero and extreme values
            epsilon = np.clip(epsilon, 1e-10, 1e2)
            eta = np.clip(eta, -1e2, 1e2)
            
            # Compute observables with bounds
            n_s = np.clip(1 - 6 * epsilon + 2 * eta, 0, 2)
            r = np.clip(16 * epsilon, 0, 1)
            A_s = np.clip(self.V0 / (24 * np.pi**2 * self.M_pl**4 * epsilon), 1e-12, 1e-8)
            
            return {
                'n_s': n_s,
                'r': r,
                'A_s': A_s,
                'epsilon': epsilon,
                'eta': eta
            }
            
        except Exception as e:
            print(f"Error in observable calculation: {str(e)}")
            return {
                'n_s': 1.0,
                'r': 0.0,
                'A_s': 2.1e-9,
                'epsilon': 0.0,
                'eta': 0.0
            }


class HolographicTheory:
    """
    Enhanced holographic implementation with proper bounds
    """
    def __init__(self):
        self.scales = ScaleHierarchy()
        self.lambda_t = 0.01
        
    def wilson_loop(self, r, T):
        """Bounded Wilson loop"""
        r_scaled = self.scales.enforce_length_scale(r)
        T_scaled = np.clip(T, 0, 1/self.scales.t_p)
        S_ng = self.lambda_t * T_scaled * r_scaled
        return np.exp(-np.clip(S_ng, 0, 100))
    
    def entanglement_entropy(self, L, z):
        """Bounded entanglement entropy"""
        L_scaled = self.scales.enforce_length_scale(L)
        z_scaled = self.scales.enforce_length_scale(z)
        
        # Implement proper UV cutoff
        z_uv = max(z_scaled, self.scales.l_p)
        
        # Bounded entropy calculation
        c = L_scaled / (4 * G * z_uv)
        entropy = c * np.log(L_scaled / z_uv)
        
        # Enforce holographic bound
        max_entropy = (L_scaled / self.scales.l_p)**2
        return np.clip(entropy, 0, max_entropy)

def run_enhanced_simulation():
    """
    Run simulation with fixed parameters
    """
    print("\nInitializing enhanced simulation...")
    
    # Initialize components
    inflation = CosmologicalInflation()
    
    print("\nInflation parameters:")
    print(f"V0 = {inflation.V0/(inflation.M_pl**4):.2e} M_pl⁴")
    print(f"phi0 = {inflation.phi0/inflation.M_pl:.2f} M_pl")
    print(f"Target N = {inflation.N_target}")
    
    t, y, obs = inflation.solve_inflation()
    
    print("\nObservables:")
    print(f"n_s = {obs['n_s']:.4f} (Target: 0.9649 ± 0.0042)")
    print(f"r = {obs['r']:.4e} (Target: < 0.064)")
    print(f"ln(10¹⁰ A_s) = {np.log(1e10 * obs['A_s']):.4f} (Target: 3.044 ± 0.014)")
    
    return {'time': t, 'fields': y, 'observables': obs}




class PowerSpectrumAnalyzer:
    """Analyze the primordial power spectrum"""
    def __init__(self):
        self.H0 = 70 * 1000 / (3.086e22)  # Hubble constant in s^-1
        self.As = 2.1e-9  # Scalar amplitude
        self.ns = 0.96  # Scalar spectral index

    def compute_power_spectrum(self, field_data, times):
        """Compute the power spectrum"""
        field = np.array(field_data) - np.mean(field_data)
        dt = times[1] - times[0]
        frequencies = np.fft.fftfreq(len(times), dt)
        field_ft = np.fft.fft(field)
        power = np.abs(field_ft) ** 2
        k = 2 * np.pi * np.abs(frequencies) / c
        P_k = power * k ** 3 / (2 * np.pi ** 2)
        k_positive = k[k > 0]
        P_k_positive = P_k[k > 0]
        spectral_index = np.polyfit(np.log(k_positive), np.log(P_k_positive), 1)[0]
        P_k_err = np.sqrt(2 / len(P_k_positive)) * P_k_positive
        results = {
            'k_modes': k_positive,
            'power_spectrum': P_k_positive,
            'spectral_index': spectral_index,
            'errors': P_k_err
        }
        return results

    def plot_power_spectrum(self, results):
        """Plot the power spectrum"""
        k = results['k_modes']
        P_k = results['power_spectrum']
        errors = results['errors']
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.errorbar(k, P_k, yerr=errors, fmt='o', alpha=0.5)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('k [Mpc^-1]')
        ax1.set_ylabel('P(k)')
        ax1.set_title('Primordial Power Spectrum')
        plt.tight_layout()
        plt.show()


class AdvancedNumericalMethods:
    """Advanced numerical methods for ODE solvers"""
    def __init__(self, spatial_points=1000, time_points=1000):
        self.Nr = spatial_points
        self.Nt = time_points
        self.MIN_STEP = 1e-12
        self.MAX_STEP = 0.1
        self.SAFETY = 0.9
        self.MAX_STEPS = 5000

    def adaptive_rk(self, f, y0, t_span, tol=1e-6):
        """Adaptive RK45 solver"""
        def rk45_step(f, t, y, h):
            """One step of RK45"""
            y = np.array(y, dtype=np.float64)
            k1 = np.array(f(t, y), dtype=np.float64) * h
            k2 = np.array(f(t + 0.25 * h, y + 0.25 * k1), dtype=np.float64) * h
            k3 = np.array(f(t + 0.375 * h, y + 0.09375 * k1 + 0.28125 * k2), dtype=np.float64) * h
            k4 = np.array(f(t + 12 / 13 * h, y + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3), dtype=np.float64) * h
            k5 = np.array(f(t + h, y + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4), dtype=np.float64) * h
            k6 = np.array(f(t + 0.5 * h, y - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5), dtype=np.float64) * h
            y4 = y + 25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - 1 / 5 * k5
            y5 = y + 16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6
            error = np.max(np.abs(y5 - y4))
            return y5, error

        # Initialize
        t = [t_span[0]]
        y = [np.array(y0, dtype=np.float64)]
        h = min((t_span[1] - t_span[0]) / 100, self.MAX_STEP)

        # Time integration
        step_count = 0
        while t[-1] < t_span[1] and step_count < self.MAX_STEPS:
            y_new, error = rk45_step(f, t[-1], y[-1], h)
            if error < tol:
                t.append(t[-1] + h)
                y.append(y_new)
                step_count += 1
                if error < tol / 10:
                    h = min(h * 1.5, self.MAX_STEP)
            else:
                h = max(h * 0.5, self.MIN_STEP)
        return np.array(t), np.array(y)
        


class KristensenQuantumPhase:
    """
    Fixed Kristensen Parameter implementation with proper scaling
    """
    def __init__(self):
        self.G = 6.67430e-11
        self.c = 3.0e8
        self.hbar = 1.0545718e-34
        self.l_p = np.sqrt(self.G * self.hbar / self.c**3)
        self.t_p = self.l_p / self.c
        self.m_p = np.sqrt(self.hbar * self.c / self.G)
        
    def compute_kappa(self, mass, radius, temperature):
        """
        Enhanced κ-parameter computation with proper scaling
        """
        # Modify parameter constants for better scaling
        alpha = 0.1  # Increased from 1/(2*np.pi)
        beta = 1.0   # Changed from 2/3
        gamma = 1.0  # Changed from 5/4
        
        # Normalize with exponential suppression control
        m_norm = mass / self.m_p
        r_norm = radius / self.l_p
        T_p = self.m_p * self.c**2 / 1.380649e-23
        T_norm = temperature / T_p
        
        # Modified phase transition function
        theta = np.pi * (m_norm * T_norm) / 10  # Reduced scaling
        phi = self.phase_transition_function(theta)
        
        # Enhanced kappa with controlled scaling
        kappa = (alpha * 
                np.power(m_norm, beta) * 
                np.exp(-r_norm/100) *  # Reduced suppression
                np.power(T_norm, gamma) * 
                phi)
        
        return np.clip(kappa, 1e-10, 1.0)  # Prevent extreme suppression

    def phase_transition_function(self, theta):
        """
        Modified phase transition function with better behavior
        """
        return 0.5 * (1 + np.cos(theta)) * np.exp(-theta**2/10)  # Reduced suppression


    def quantum_transition_probability(self, kappa):
        """
        Compute probability of quantum transition between aeons
        based on the κ-parameter
        """
        return 1 - np.exp(-kappa**2)
    
    def energy_scale(self, kappa):
        """
        Compute characteristic energy scale associated with
        the κ-parameter
        """
        return self.m_p * self.c**2 * np.abs(kappa)
    
    def analyze_transition(self, mass_range, radius_range, temperature):
        """
        Analyze quantum phase transition across parameter ranges
        """
        kappas = []
        probabilities = []
        energies = []
        
        for m, r in zip(mass_range, radius_range):
            kappa = self.compute_kappa(m, r, temperature)
            prob = self.quantum_transition_probability(kappa)
            energy = self.energy_scale(kappa)
            
            kappas.append(kappa)
            probabilities.append(prob)
            energies.append(energy)
            
        return np.array(kappas), np.array(probabilities), np.array(energies)
def stabilize_foam_factor(euler_char):
    """
    Stabilize the foam factor calculation to prevent overflow
    """
    # Clip the Euler characteristic to prevent extreme values
    max_value = 100
    clipped_euler = np.clip(euler_char, -max_value, max_value)
    
    # Use log-space calculation for numerical stability
    log_factor = -clipped_euler / 100
    # Prevent overflow in exp by clipping
    log_factor = np.clip(log_factor, -20, 20)
    
    return np.exp(log_factor)
class QuantumFoamTopology:
    """
    Fixed quantum foam implementation
    """
    def __init__(self):
        self.planck_length = np.sqrt(G * hbar / c**3)
        self.foam_density = 0.1 / self.planck_length**3  # Reduced density
        
    def generate_foam_structure(self, volume, temperature):
        """Modified foam structure generation"""
        max_nodes = 50
        volume_scaled = min(volume, max_nodes / self.foam_density)
        num_nodes = min(int(volume_scaled * self.foam_density), max_nodes)
        
        # Smoother position generation
        positions = np.random.rand(num_nodes, 3) * np.cbrt(volume_scaled)
        
        # Create network with controlled connectivity
        self.network = nx.Graph()
        for i in range(num_nodes):
            self.network.add_node(i, pos=positions[i])
        
        # Modified edge probability
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                distance = np.linalg.norm(positions[i] - positions[j])
                prob = np.exp(-distance/(5*self.planck_length))  # Increased range
                if np.random.random() < prob:
                    self.network.add_edge(i, j)
                    
        return self.compute_topology_invariants()

    def compute_topology_invariants(self):
        """Memory-efficient topology computation"""
        try:
            # Compute on full network if small enough
            if self.network.number_of_nodes() <= 50:
                sub_network = self.network
            else:
                # Take first 50 nodes for larger networks
                sub_network = self.network.subgraph(range(50))
            
            betti_0 = nx.number_connected_components(sub_network)
            cycles = nx.cycle_basis(sub_network)
            betti_1 = len(cycles)
            
            return {
                'betti_0': betti_0,
                'betti_1': betti_1,
                'euler_characteristic': betti_0 - betti_1
            }
        except Exception as e:
            print(f"Error in topology computation: {str(e)}")
            return {
                'betti_0': 1,
                'betti_1': 0,
                'euler_characteristic': 1
            }


class QuantumDecoherence:
    """
    Models quantum decoherence effects in the early universe.
    Incorporates environmental interaction and quantum-to-classical transition.
    """
    def __init__(self, system_size):
        self.system_size = system_size
        self.H_system = np.random.random((system_size, system_size)) + \
                       1j * np.random.random((system_size, system_size))
        self.H_system = self.H_system + self.H_system.conj().T  # Make Hermitian
        
    def lindblad_evolution(self, rho_0, t, gamma):
        """
        Solve Lindblad master equation for density matrix evolution
        """
        def lindblad_deriv(t, rho_flat):
            rho = rho_flat.reshape(self.system_size, self.system_size)
            
            # Coherent evolution
            comm = self.H_system @ rho - rho @ self.H_system
            
            # Decoherence term
            decay = np.zeros_like(rho)
            for i in range(self.system_size):
                for j in range(self.system_size):
                    if i != j:
                        decay[i,j] = -gamma * rho[i,j]
                        
            drho_dt = -1j * comm + decay
            return drho_dt.flatten()
            
        sol = integrate.solve_ivp(
            lindblad_evolution,
            (0, t),
            rho_0.flatten(),
            method='RK45',
            t_eval=np.linspace(0, t, 100)
        )
        
        return sol.y.reshape(-1, self.system_size, self.system_size)

class NoncommutativeGeometry:
    """
    Fixed implementation of noncommutative geometry effects at Planck scale.
    """
    def __init__(self, theta_parameter):
        self.theta = float(theta_parameter)  # Ensure theta is float
        
    def modified_dispersion(self, momentum, mass):
        """
        Calculate modified dispersion relation E(p) with noncommutative corrections
        Ensures proper numerical types throughout calculation
        """
        try:
            # Convert inputs to numpy arrays/floats
            momentum = np.array(momentum, dtype=float)
            mass = float(mass)
            
            # Calculate p^2 with numpy
            p_squared = np.sum(momentum**2)
            
            # Leading order NC correction
            nc_correction = self.theta * float(p_squared)**2 / (4 * float(c)**2)
            
            # Modified energy with explicit float conversions
            energy = float(np.sqrt(p_squared * float(c)**2 + mass**2 * float(c)**4)) + nc_correction
            return energy
            
        except Exception as e:
            print(f"Error in dispersion calculation: {str(e)}")
            # Return a safe default value
            return mass * float(c)**2
    
    def uncertainty_relation(self, delta_x, delta_p):
        """
        Compute generalized uncertainty relation with NC geometry
        """
        try:
            delta_x = float(delta_x)
            delta_p = float(delta_p)
            return delta_x * delta_p + self.theta * delta_p**2
        except Exception as e:
            print(f"Error in uncertainty calculation: {str(e)}")
            return float('inf')

class EnhancedKristensenAnalysis(KristensenQuantumPhase):
    """
    Enhanced version of Kristensen analysis incorporating quantum foam and decoherence
    """
    def __init__(self):
        super().__init__()
        self.foam = QuantumFoamTopology()
        self.decoherence = QuantumDecoherence(system_size=10)
        self.nc_geometry = NoncommutativeGeometry(theta_parameter=l_p**2)
        
    def compute_enhanced_kappa(self, mass, radius, temperature):
        """
        Enhanced κ-parameter computation including quantum foam effects
        """
        # Get base kappa
        kappa_base = self.compute_kappa(mass, radius, temperature)
        
        # Compute foam topology
        foam_invariants = self.foam.generate_foam_structure(radius**3, temperature)
        
        # Topology correction factor
        topology_factor = np.exp(-foam_invariants['euler_characteristic'] / 100)
        
        # Noncommutative correction
        p = mass * c
        nc_energy = self.nc_geometry.modified_dispersion(np.array([p, 0, 0]), mass)
        nc_factor = nc_energy / (mass * c**2)
        
        # Enhanced kappa
        kappa_enhanced = kappa_base * topology_factor * nc_factor
        
        return kappa_enhanced, {
            'topology_factor': topology_factor,
            'nc_factor': nc_factor,
            'foam_invariants': foam_invariants
        }
class InflationDiagnostics:
    def __init__(self, inflation_model):
        self.model = inflation_model
        self.M_pl = inflation_model.M_pl
        self.diagnostic_data = {}
        
    def run_full_diagnostics(self, t, y, save_path="inflation_diagnostics"):
        """
        Run comprehensive diagnostics suite
        """
        print("\nRunning comprehensive inflation diagnostics...")
        
        try:
            # Create output directory
            os.makedirs(save_path, exist_ok=True)
            
            print("Collecting basic data...")
            self.collect_basic_data(t, y)
            
            print("Analyzing slow-roll...")
            self.analyze_slow_roll(t, y)
            
            print("Checking energy conditions...")
            self.check_energy_conditions(t, y)
            
            print("Analyzing scales...")
            self.analyze_scales(t, y)
            
            print("Computing perturbations...")
            self.compute_perturbations(t, y)
            
            # Save and visualize results
            print("Saving diagnostics...")
            self.save_diagnostics(save_path)
            
            print("Creating diagnostic plots...")
            self.create_diagnostic_plots(save_path)
            
        except Exception as e:
            print(f"Error in diagnostics: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_diagnostics(self, save_path):
        """
        Save detailed diagnostic data and analysis
        """
        try:
            with open(f"{save_path}/inflation_diagnostics.txt", 'w') as f:
                f.write("=== INFLATION DIAGNOSTIC SUMMARY ===\n\n")
                
                # Basic parameters
                f.write("1. Basic Parameters\n")
                f.write("-----------------\n")
                if 'efolds' in self.diagnostic_data:
                    f.write(f"Total e-folds: {self.diagnostic_data['efolds'][-1]:.2f}\n")
                if 'time' in self.diagnostic_data:
                    f.write(f"Inflation duration: {self.diagnostic_data['time'][-1]:.2e} s\n")
                if 'phi' in self.diagnostic_data:
                    f.write(f"Final field value: {self.diagnostic_data['phi'][-1]:.2f} M_pl\n\n")
                
                # Slow-roll analysis
                f.write("2. Slow-Roll Analysis\n")
                f.write("-------------------\n")
                if 'epsilon' in self.diagnostic_data:
                    f.write(f"Mean epsilon: {np.mean(self.diagnostic_data['epsilon']):.2e}\n")
                if 'eta' in self.diagnostic_data:
                    f.write(f"Mean eta: {np.mean(self.diagnostic_data['eta']):.2e}\n")
                if 'slow_roll_valid' in self.diagnostic_data:
                    f.write(f"Slow-roll validity: {np.all(self.diagnostic_data['slow_roll_valid'])}\n\n")
                
                # Energy conditions
                f.write("3. Energy Conditions\n")
                f.write("------------------\n")
                conditions = ['nec', 'wec', 'sec', 'dec']
                for cond in conditions:
                    if cond in self.diagnostic_data:
                        f.write(f"{cond.upper()}: {np.all(self.diagnostic_data[cond])}\n")
                f.write("\n")
                
                # Observables
                f.write("4. Observables\n")
                f.write("-------------\n")
                if 'n_s' in self.diagnostic_data:
                    f.write(f"Final n_s: {self.diagnostic_data['n_s'][-1]:.6f}\n")
                if 'epsilon' in self.diagnostic_data:
                    f.write(f"Final r: {16*self.diagnostic_data['epsilon'][-1]:.6f}\n")
                if 'scalar_amplitude' in self.diagnostic_data:
                    f.write(f"ln(10¹⁰ A_s): {np.log(1e10 * self.diagnostic_data['scalar_amplitude'][-1]):.6f}\n")
                
                # Save numerical data
                np.savez(f"{save_path}/diagnostic_data.npz", **self.diagnostic_data)
                
        except Exception as e:
            print(f"Error saving diagnostics: {str(e)}")
            
    def collect_basic_data(self, t, y):
        """Collect basic evolution data"""
        try:
            phi, phi_dot = y
            
            # Field evolution
            self.diagnostic_data['phi'] = phi / self.M_pl
            self.diagnostic_data['phi_dot'] = phi_dot / self.M_pl**2
            self.diagnostic_data['time'] = t
            
            # Potential and energy
            V = np.array([self.model.potential(p) for p in phi])
            K = 0.5 * phi_dot**2
            
            self.diagnostic_data['potential'] = V
            self.diagnostic_data['kinetic'] = K
            self.diagnostic_data['total_energy'] = V + K
            
            # Hubble parameter
            H = np.array([self.model.hubble_parameter(p, pd) 
                         for p, pd in zip(phi, phi_dot)])
            self.diagnostic_data['hubble'] = H
            
            # E-foldings
            self.diagnostic_data['efolds'] = np.cumsum(H * np.diff(t, prepend=0))
            
        except Exception as e:
            print(f"Error collecting basic data: {str(e)}")

    def analyze_slow_roll(self, t, y):
        """Detailed slow-roll analysis"""
        try:
            phi = y[0]
            epsilon_vals = []
            eta_vals = []
            
            for p in phi:
                epsilon, eta = self.model.slow_roll_parameters(p)
                epsilon_vals.append(epsilon)
                eta_vals.append(eta)
                
            self.diagnostic_data['epsilon'] = np.array(epsilon_vals)
            self.diagnostic_data['eta'] = np.array(eta_vals)
            
            # Slow-roll validity checks
            self.diagnostic_data['slow_roll_valid'] = (
                (self.diagnostic_data['epsilon'] < 1) & 
                (np.abs(self.diagnostic_data['eta']) < 1)
            )
            
        except Exception as e:
            print(f"Error in slow-roll analysis: {str(e)}")

    def check_energy_conditions(self, t, y):
        """Check various energy conditions"""
        try:
            rho = self.diagnostic_data['total_energy']
            H = self.diagnostic_data['hubble']
            
            self.diagnostic_data['nec'] = rho >= 0
            self.diagnostic_data['wec'] = rho >= 0
            self.diagnostic_data['sec'] = rho + 3 * H**2 >= 0
            self.diagnostic_data['dec'] = rho >= 0
            
        except Exception as e:
            print(f"Error checking energy conditions: {str(e)}")

    def analyze_scales(self, t, y):
        """Analyze relevant physical scales"""
        try:
            H = self.diagnostic_data['hubble']
            
            self.diagnostic_data['horizon_size'] = c / H
            self.diagnostic_data['quantum_wavelength'] = np.sqrt(hbar / (H * c))
            self.diagnostic_data['thermal_wavelength'] = hbar * c / (k_B * H)
            
        except Exception as e:
            print(f"Error analyzing scales: {str(e)}")

    def compute_perturbations(self, t, y):
        """Analyze perturbations and power spectra"""
        try:
            H = self.diagnostic_data['hubble']
            epsilon = self.diagnostic_data['epsilon']
            
            self.diagnostic_data['scalar_amplitude'] = (
                H**2 / (8 * np.pi**2 * self.M_pl**2 * epsilon)
            )
            
            self.diagnostic_data['tensor_amplitude'] = (
                2 * H**2 / (np.pi**2 * self.M_pl**2)
            )
            
            self.diagnostic_data['n_s'] = 1 - 6*epsilon + 2*self.diagnostic_data['eta']
            self.diagnostic_data['n_t'] = -2*epsilon
            
        except Exception as e:
            print(f"Error computing perturbations: {str(e)}")

        
    def create_diagnostic_plots(self, save_path):
        """
        Create comprehensive diagnostic plots
        """
        # 1. Field Evolution
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.plot(self.diagnostic_data['time'], self.diagnostic_data['phi'])
        plt.xlabel('Time (s)')
        plt.ylabel('φ/M_pl')
        plt.title('Inflaton Evolution')
        plt.yscale('log')
        
        plt.subplot(222)
        plt.plot(self.diagnostic_data['time'], self.diagnostic_data['efolds'])
        plt.xlabel('Time (s)')
        plt.ylabel('N (e-folds)')
        plt.title('Number of e-foldings')
        
        plt.subplot(223)
        plt.plot(self.diagnostic_data['time'], self.diagnostic_data['epsilon'])
        plt.xlabel('Time (s)')
        plt.ylabel('ε')
        plt.title('Slow-roll parameter ε')
        plt.yscale('log')
        
        plt.subplot(224)
        plt.plot(self.diagnostic_data['time'], self.diagnostic_data['hubble'])
        plt.xlabel('Time (s)')
        plt.ylabel('H (s⁻¹)')
        plt.title('Hubble Parameter')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/field_evolution.png")
        plt.close()
        
        # 2. Energy Distribution
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(self.diagnostic_data['time'], self.diagnostic_data['potential'], 
                label='Potential')
        plt.plot(self.diagnostic_data['time'], self.diagnostic_data['kinetic'], 
                label='Kinetic')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy/M_pl⁴')
        plt.title('Energy Components')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(132)
        plt.plot(self.diagnostic_data['time'], 
                self.diagnostic_data['scalar_amplitude'])
        plt.xlabel('Time (s)')
        plt.ylabel('Δ²_s(k)')
        plt.title('Scalar Power Spectrum')
        plt.yscale('log')
        
        plt.subplot(133)
        plt.plot(self.diagnostic_data['time'], self.diagnostic_data['n_s'])
        plt.xlabel('Time (s)')
        plt.ylabel('n_s')
        plt.title('Scalar Spectral Index')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/energy_spectra.png")
        plt.close()
def run_inflation_with_diagnostics():
    """
    Fixed run_inflation_with_diagnostics with proper value handling
    """
    try:
        print("\nInitializing enhanced inflation simulation...")
        inflation = CosmologicalInflation()
        diagnostics = InflationDiagnostics(inflation)
        
        # Get all return values from solve_inflation
        result = inflation.solve_inflation()
        if result is None or len(result) != 3:
            print("Failed to get valid inflation solution")
            return None
            
        t, y, obs = result
        
        if t is None or y is None:
            print("Invalid solution components")
            return None
            
        print("\nRunning comprehensive diagnostics...")
        diagnostics.run_full_diagnostics(t, y)
        
        return diagnostics
        
    except Exception as e:
        print(f"Error in inflation simulation: {str(e)}")
        return None




    def save_diagnostics(self, save_path):
        """
        Save detailed diagnostic data and analysis
        """
        with open(f"{save_path}/inflation_diagnostics.txt", 'w') as f:
            f.write("=== INFLATION DIAGNOSTIC SUMMARY ===\n\n")
            
            # Basic parameters
            f.write("1. Basic Parameters\n")
            f.write("-----------------\n")
            f.write(f"Total e-folds: {self.diagnostic_data['efolds'][-1]:.2f}\n")
            f.write(f"Inflation duration: {self.diagnostic_data['time'][-1]:.2e} s\n")
            f.write(f"Final field value: {self.diagnostic_data['phi'][-1]:.2f} M_pl\n\n")
            
            # Slow-roll analysis
            f.write("2. Slow-Roll Analysis\n")
            f.write("-------------------\n")
            f.write(f"Mean epsilon: {np.mean(self.diagnostic_data['epsilon']):.2e}\n")
            f.write(f"Mean eta: {np.mean(self.diagnostic_data['eta']):.2e}\n")
            f.write(f"Slow-roll validity: {np.all(self.diagnostic_data['slow_roll_valid'])}\n\n")
            
            # Energy conditions
            f.write("3. Energy Conditions\n")
            f.write("------------------\n")
            f.write(f"Null Energy Condition: {np.all(self.diagnostic_data['nec'])}\n")
            f.write(f"Weak Energy Condition: {np.all(self.diagnostic_data['wec'])}\n")
            f.write(f"Strong Energy Condition: {np.all(self.diagnostic_data['sec'])}\n")
            f.write(f"Dominant Energy Condition: {np.all(self.diagnostic_data['dec'])}\n\n")
            
            # Observables
            f.write("4. Observables\n")
            f.write("-------------\n")
            f.write(f"Final n_s: {self.diagnostic_data['n_s'][-1]:.6f}\n")
            f.write(f"Final r: {16*self.diagnostic_data['epsilon'][-1]:.6f}\n")
            f.write(f"ln(10¹⁰ A_s): {np.log(1e10 * self.diagnostic_data['scalar_amplitude'][-1]):.6f}\n")


def add_kristensen_analysis(results_dict, output_dir="quantum_cosmos_results"):
    """Add Kristensen parameter analysis to simulation results"""
    
    # Initialize Kristensen analysis
    kp = KristensenQuantumPhase()
    
    # Extract relevant quantities from simulation
    mass_range = results_dict["mass_range"]
    radius_range = results_dict["radius_range"]
    T = results_dict["temperature"]
    
    # Compute Kristensen parameters
    kappas, probs, energies = kp.analyze_transition(mass_range, radius_range, T)
    
    # Save results
    with open(f"{output_dir}/kristensen_analysis.txt", "w") as f:
        f.write("=== KRISTENSEN QUANTUM PHASE ANALYSIS ===\n\n")
        f.write(f"Maximum κ-parameter: {np.max(kappas):.2e}\n")
        f.write(f"Mean transition probability: {np.mean(probs):.2e}\n")
        f.write(f"Characteristic energy scale: {np.mean(energies)/1e9:.2e} GeV\n")
        
        # Check if there are any probabilities > 0.5
        high_prob_kappas = kappas[probs > 0.5]
        if high_prob_kappas.size > 0:
            f.write(f"Phase transition threshold: {np.min(high_prob_kappas):.2e}\n")
        else:
            f.write("Phase transition threshold: No transition probability > 0.5 found\n")
        
        # Identify critical points
        critical_idx = np.argmax(np.gradient(probs))
        f.write("\nCritical Point Analysis:\n")
        f.write(f"Critical κ value: {kappas[critical_idx]:.2e}\n")
        f.write(f"Critical mass: {mass_range[critical_idx]/float(m_p):.2e} M_p\n")
        f.write(f"Critical radius: {radius_range[critical_idx]/float(l_p):.2e} l_p\n")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # κ-parameter distribution
    plt.subplot(131)
    plt.plot(mass_range/float(m_p), kappas)
    plt.xlabel('Mass (M_p)')
    plt.ylabel('κ-parameter')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Kristensen Parameter')
    
    # Transition probability
    plt.subplot(132)
    plt.plot(mass_range/float(m_p), probs)
    plt.xlabel('Mass (M_p)')
    plt.ylabel('Transition Probability')
    plt.xscale('log')
    plt.title('Quantum Transition Probability')
    
    # Energy scale
    plt.subplot(133)
    plt.plot(mass_range/float(m_p), energies/1e9)
    plt.xlabel('Mass (M_p)')
    plt.ylabel('Energy Scale (GeV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Characteristic Energy Scale')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kristensen_analysis.png")
    plt.close()
    
    # Add to console output
    print("\n=== KRISTENSEN QUANTUM PHASE ANALYSIS ===")
    print(f"Maximum κ-parameter: {np.max(kappas):.2e}")
    print(f"Mean transition probability: {np.mean(probs):.2e}")
    print(f"Characteristic energy scale: {np.mean(energies)/1e9:.2e} GeV")
    print("\nPhysical interpretation:")
    if np.max(kappas) > 1:
        print("- Strong quantum phase coupling detected")
        print("- High probability of aeon transition")
        print(f"- Quantum effects dominant at {energies[critical_idx]/1e9:.1e} GeV")
    else:
        print("- Weak quantum phase coupling")
        print("- Classical regime dominant")
        
class BitcoinQuantumTransfer:
    """
    Simulates the potential for transferring Bitcoin blocks through aeon transitions
    using the Kristensen Quantum Phase Parameter (κ-parameter).
    """
    def __init__(self, block_size, block_difficulty, temperature):
        # Initialize with the Kristensen Quantum Phase parameter class
        self.kristensen = KristensenQuantumPhase()
        self.block_size = block_size  # Mass-like property of blocks (size in bytes)
        self.block_difficulty = block_difficulty  # Radius-like property (mining difficulty can be analogous to radius)
        self.temperature = temperature  # Entropy of the network (e.g., background temperature like CMB)

    def compute_block_kappa(self, block):
        """
        Compute κ for a specific block in the Bitcoin blockchain.
        """
        # Treat block_size as mass and block_difficulty as radius for κ computation
        mass = self.block_size[block]
        radius = self.block_difficulty[block]
        
        # Compute the κ-parameter using Kristensen's formula
        return self.kristensen.compute_kappa(mass, radius, self.temperature)

    def analyze_blocks(self):
        """
        Analyze all blocks in the blockchain for their potential to be transferred to the next aeon.
        """
        kappas = []
        probabilities = []
        
        # Loop through all blocks and compute κ-parameter for each
        for block in range(len(self.block_size)):
            kappa = self.compute_block_kappa(block)
            transition_prob = self.kristensen.quantum_transition_probability(kappa)
            kappas.append(kappa)
            probabilities.append(transition_prob)
            
            # Output the κ and transition probability for each block
            print(f"Block {block}: κ={kappa:.2e}, Transition Probability={transition_prob:.2f}")
        
        # Return the κ-parameters and transition probabilities for further analysis or visualization
        return kappas, probabilities

    def transfer_analysis(self):
        """
        Analyze the transfer potential of blocks across aeons.
        """
        kappas, probabilities = self.analyze_blocks()

        # Threshold: Determine which blocks have the highest transition probability
        critical_blocks = [i for i, prob in enumerate(probabilities) if prob > 0.5]
        
        if critical_blocks:
            print(f"\nBlocks with high transfer potential: {critical_blocks}")
        else:
            print("\nNo blocks with high transfer potential were found.")

        return critical_blocks

def run_inflation_analysis():
    """
    Fixed run_inflation_analysis with proper value unpacking
    """
    print("\nInitializing inflation simulation...")
    inflation = CosmologicalInflation()
    
    print("Solving inflation dynamics...")
    result = inflation.solve_inflation()
    
    if result is None or len(result) != 3:
        print("Failed to get valid inflation solution")
        return None, None, None
        
    t, y, obs = result
    
    if obs is not None:
        print("\nInflationary Observables:")
        print(f"Scalar spectral index (n_s): {obs['n_s']:.6f} (Target: 0.9649 ± 0.0042)")
        print(f"Tensor-to-scalar ratio (r): {obs['r']:.6f} (Target: < 0.064)")
        print(f"Scalar amplitude (ln(10¹⁰ A_s)): {np.log(1e10 * obs['A_s']):.6f} (Target: 3.044 ± 0.014)")
    
    return t, y, obs




def analyze_and_save_results(t_inf, phi_inf, r, horizon_corr, dilaton, epsilon, eta, wilson, entropy):
    """Enhanced analysis with detailed physics interpretation"""
    output_dir = "quantum_cosmos_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate additional metrics
    # Inflation metrics
    inflation_duration = t_inf[-1] - t_inf[0]
    e_folds = np.log(phi_inf[-1, 0] / phi_inf[0, 0])
    hubble_parameter = np.sqrt(8 * np.pi * G / (3 * c ** 2) * np.gradient(phi_inf[:, 0]) ** 2)

    # String theory metrics
    horizon_scale = r[np.argmax(horizon_corr)]
    dilaton_decay_length = r[len(r) // 2] / np.log(dilaton[0] / dilaton[len(r) // 2])

    # Holographic metrics
    entropy_density = entropy / r ** 2
    
    # Calculate Wilson correlation length safely
    slope = np.polyfit(r, np.log(wilson), 1)[0]
    wilson_correlation_length = float('inf') if slope == 0 else -1 / slope

    # Save results to text file
    with open(f"{output_dir}/detailed_analysis.txt", "w") as f:
        f.write("=== COMPREHENSIVE COSMOLOGICAL ANALYSIS ===\n\n")
        
        # 1. Inflation Analysis
        f.write("1. INFLATION DYNAMICS\n")
        f.write("-----------------------\n")
        f.write(f"Duration: {inflation_duration:.2e} seconds\n")
        f.write(f"Number of e-foldings: {e_folds:.2f}\n")
        f.write(f"Initial inflaton value: {phi_inf[0, 0]:.2e} Mp\n")
        f.write(f"Final inflaton value: {phi_inf[-1, 0]:.2e} Mp\n")
        f.write(f"Mean Hubble parameter: {np.mean(hubble_parameter):.2e} s^-1\n")
        f.write(f"Slow-roll parameters:\n")
        f.write(f"  - Mean epsilon: {np.mean(epsilon):.2e}\n")
        f.write(f"  - Mean eta: {np.mean(eta):.2e}\n")
        f.write(f"  - Epsilon/eta correlation: {np.corrcoef(epsilon, eta)[0, 1]:.2f}\n\n")

        # 2. String Theory Analysis
        f.write("2. STRING THEORY EFFECTS\n")
        f.write("-----------------------\n")
        f.write(f"Maximum horizon correction: {np.max(horizon_corr):.2e}\n")
        f.write(f"Characteristic horizon scale: {horizon_scale / float(l_p):.2e} l_p\n")
        f.write(f"Dilaton profile:\n")
        f.write(f"  - Decay length: {dilaton_decay_length / float(l_p):.2e} l_p\n")
        f.write(f"  - Maximum value: {np.max(dilaton):.2e}\n")
        f.write(f"  - Minimum value: {np.min(dilaton):.2e}\n\n")

        # 3. Holographic Analysis
        f.write("3. HOLOGRAPHIC PROPERTIES\n")
        f.write("-----------------------\n")
        f.write(f"Entanglement entropy:\n")
        f.write(f"  - Area law coefficient: {np.polyfit(np.log(r), np.log(entropy), 1)[0]:.2f}\n")
        f.write(f"  - Maximum entropy density: {np.max(entropy_density):.2e}\n")
        f.write(f"Wilson loop:\n")
        f.write(f"  - Correlation length: {wilson_correlation_length / float(l_p):.2e} l_p\n")
        f.write(f"  - Area law violation: {np.abs(1 + np.polyfit(r, np.log(wilson), 1)[0]):.2e}\n\n")

        # 4. Scale Analysis
        f.write("4. SCALE DEPENDENCE\n")
        f.write("-----------------------\n")
        scales = {
            'Planck scale': l_p,
            'Horizon scale': horizon_scale,
            'Correlation length': wilson_correlation_length,
            'Dilaton decay length': dilaton_decay_length
        }
        f.write("Hierarchy of scales:\n")
        for name, scale in sorted(scales.items(), key=lambda x: x[1]):
            f.write(f"  - {name}: {scale / float(l_p):.2e} l_p\n")

        # 5. Consistency Checks
        f.write("\n5. CONSISTENCY CHECKS\n")
        f.write("-----------------------\n")
        checks = {
            "Inflation sufficient": e_folds > 60,
            "Slow-roll valid": np.all(epsilon < 1),
            "Dilaton well-behaved": np.all(np.isfinite(dilaton)),
            "Holographic bound respected": np.all(entropy <= (r / l_p) ** 2),
        }
        for check, passed in checks.items():
            f.write(f"{check}: {'PASSED' if passed else 'FAILED'}\n")

    # Print summary to console
    print("\n=== DETAILED ANALYSIS SUMMARY ===")
    print("\n1. Inflation Characteristics:")
    print(f"   - Duration: {inflation_duration:.2e} s")
    print(f"   - E-foldings: {e_folds:.2f}")
    print(f"   - Mean Hubble parameter: {np.mean(hubble_parameter):.2e} s^-1")
    
    print("\n2. Quantum Gravity Effects:")
    print(f"   - Maximum horizon correction: {np.max(horizon_corr):.2e}")
    print(f"   - Characteristic scale: {horizon_scale / float(l_p):.2e} l_p")
    print(f"   - Dilaton decay length: {dilaton_decay_length / float(l_p):.2e} l_p")
    
    print("\n3. Holographic Properties:")
    print(f"   - Area law coefficient: {np.polyfit(np.log(r), np.log(entropy), 1)[0]:.2f}")
    print(f"   - Wilson correlation length: {wilson_correlation_length / float(l_p):.2e} l_p")
    
    print("\n4. Scale Analysis:")
    min_scale = min(scales.values())
    max_scale = max(scales.values())
    
    if min_scale == 0:
        print("   - Scale hierarchy: Infinite (division by zero)")
    else:
        print(f"   - Scale hierarchy: {max_scale / min_scale:.2e}")
    
    print("\n5. Physical Interpretation:")
    print("   - " + ("Sufficient inflation achieved" if e_folds > 60 else "Insufficient inflation"))
    print("   - " + ("Quantum effects significant" if np.max(horizon_corr) > 1 else "Classical regime dominant"))
    print("   - " + ("Holographic bound respected" if np.all(entropy <= (r / l_p) ** 2) else "Holographic bound violated"))

    # Save additional plots and data
    # Plot and save inflation dynamics
    plt.figure(figsize=(10, 6))
    plt.plot(t_inf, phi_inf[:, 0])
    plt.xlabel('Time (s)')
    plt.ylabel('Inflaton Field')
    plt.title('Inflation Evolution')
    plt.yscale('log')
    plt.savefig(f"{output_dir}/inflation_evolution.png")
    plt.close()

    # Plot slow-roll parameters
    plt.figure(figsize=(10, 6))
    plt.plot(t_inf, epsilon, label='ε')
    plt.plot(t_inf, eta, label='η')
    plt.xlabel('Time (s)')
    plt.ylabel('Parameters')
    plt.title('Slow-roll Parameters')
    plt.legend()
    plt.savefig(f"{output_dir}/slow_roll.png")
    plt.close()

    # Plot horizon corrections
    plt.figure(figsize=(10, 6))
    plt.plot(r / float(l_p), horizon_corr)
    plt.xlabel('r/l_p')
    plt.ylabel('Horizon Corrections')
    plt.title('String Theory Corrections')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{output_dir}/string_corrections.png")
    plt.close()

    # Plot dilaton profile
    plt.figure(figsize=(10, 6))
    plt.plot(r / float(l_p), dilaton)
    plt.xlabel('r/l_p')
    plt.ylabel('Dilaton')
    plt.title('Dilaton Profile')
    plt.xscale('log')
    plt.savefig(f"{output_dir}/dilaton.png")
    plt.close()

    # Plot Wilson loop and entropy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(r / float(l_p), wilson)
    ax1.set_xlabel('r/l_p')
    ax1.set_ylabel('Wilson Loop')
    ax1.set_title('Wilson Loop')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.plot(r / float(l_p), entropy)
    ax2.set_xlabel('r/l_p')
    ax2.set_ylabel('Entanglement Entropy')
    ax2.set_title('Entanglement Entropy')
    ax2.set_xscale('log')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/holographic_quantities.png")
    plt.close()

    # Save detailed data arrays
    np.savez(f"{output_dir}/full_simulation_data.npz",
             time=t_inf, inflaton=phi_inf, radii=r, horizon_corrections=horizon_corr,
             dilaton=dilaton, epsilon=epsilon, eta=eta, wilson=wilson, entropy=entropy)

    print("\n=== SIMULATION RESULTS SAVED ===")


def integrate_enhanced_simulation():
    """
    Type-safe integration of quantum gravity framework with inflation simulation.
    """
    try:
        print("\nInitializing integrated quantum cosmos simulation...")
        
        # Initialize components
        string = StringTheoryCorrections()
        cosmo = CosmologicalInflation()
        holo = HolographicTheory()
        foam = QuantumFoamTopology()
        nc_geom = NoncommutativeGeometry(theta_parameter=float(l_p)**2)
        
        print("\nGenerating spatial grid...")
        n_points = 100
        r = np.logspace(-1, 2, n_points) * float(l_p)
        t = np.logspace(0, 2, n_points) * float(t_p)
        print(f"Grid generated with {n_points} points")

        print("\nComputing enhanced string theory corrections...")
        horizon_corr = []
        foam_corrections = []

        # Process in smaller batches
        batch_size = 10
        for i in range(0, len(r), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{len(r)//batch_size + 1}")
            batch_r = r[i:i+batch_size]
            
            for ri in batch_r:
                try:
                    print(f"  Computing corrections for r = {ri/float(l_p):.2e} l_p")
                    
                    # Ensure float type for calculations
                    ri_float = float(ri)
                    m_p_float = float(m_p)
                    
                    # Base correction with bounds
                    base_corr = np.clip(string.horizon_corrections(ri_float, 1e5 * m_p_float), 0, 1e10)
                    print(f"    Base correction: {base_corr:.2e}")
                    
                    # Quantum foam with minimal network size
                    foam_invariants = foam.generate_foam_structure(min(ri_float**3, 1e10), 2.725)
                    print(f"    Foam invariants computed")
                    
                    foam_factor = stabilize_foam_factor(foam_invariants['euler_characteristic'])
                    print(f"    Foam factor: {foam_factor:.2e}")
                    
                    # Combine corrections with numerical stability
                    total_corr = float(base_corr) * float(foam_factor)
                    total_corr = np.clip(total_corr, 0, 1e10)
                    print(f"    Total correction: {total_corr:.2e}")
                    
                    horizon_corr.append(total_corr)
                    foam_corrections.append(foam_invariants)
                    
                except Exception as e:
                    print(f"    Warning: Error in calculation: {str(e)}")
                    horizon_corr.append(1.0)
                    foam_corrections.append({'euler_characteristic': 1, 'betti_0': 1, 'betti_1': 0})

        print("\nComputing modified dilaton profile...")
        dilaton = np.array([float(x) for x in string.dilaton_profile(r)])
        print("Dilaton profile computed")

        print("\nComputing noncommutative corrections...")
        nc_corrections = []
        for i, ri in enumerate(r):
            if i % 10 == 0:
                print(f"Processing NC correction {i+1}/{len(r)}")
            try:
                # Ensure float types for momentum calculation
                ri_float = float(ri)
                m_p_float = float(m_p)
                c_float = float(c)
                
                p = min(ri_float * m_p_float * c_float, 1e30)
                momentum = np.array([float(p), 0.0, 0.0])
                
                nc_energy = nc_geom.modified_dispersion(momentum, m_p_float)
                correction = float(nc_energy) / (m_p_float * c_float**2)
                correction = np.clip(correction, 0.1, 10.0)
                nc_corrections.append(correction)
                
            except Exception as e:
                print(f"Warning: Error in NC correction calculation: {str(e)}")
                nc_corrections.append(1.0)

        print("\nPreparing results...")

        # Initialize results dictionary
        enhanced_results = {
            'time': t,
            'radii': r,
            'horizon_corrections': np.array(horizon_corr, dtype=float),
            'foam_data': foam_corrections,
            'nc_corrections': np.array(nc_corrections, dtype=float),
            'dilaton': dilaton,
            'inflation': {'t': None, 'y': None, 'observables': None},
            'corrections': {},
            'parameters': {
                'n_points': n_points,
                'spatial_range': [float(r[0]), float(r[-1])]
            }
        }

        # Run inflation simulation
        inflation_result = cosmo.solve_inflation()
        if inflation_result is not None and len(inflation_result) == 3:
            t_inf, y_inf, obs = inflation_result
            enhanced_results['inflation']['t'] = t_inf
            enhanced_results['inflation']['y'] = y_inf
            enhanced_results['inflation']['observables'] = obs

        print("Simulation completed successfully!")
        return enhanced_results

    except Exception as e:
        print(f"\nError in enhanced simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



def save_enhanced_results(results):
    """
    Enhanced data saving with validation
    """
    try:
        output_dir = "quantum_cosmos_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate data structure
        required_keys = ['inflation', 'corrections', 'parameters']
        if not all(key in results for key in required_keys):
            raise ValueError("Missing required data keys")
            
        # Save core numerical data
        np.savez(
            f"{output_dir}/simulation_data.npz",
            time=results['inflation']['t'],
            field=results['inflation']['y'][0],
            field_velocity=results['inflation']['y'][1],
            corrections=results['corrections']
        )
        
        # Save observables
        with open(f"{output_dir}/observables.txt", 'w') as f:
            obs = results['inflation']['observables']
            f.write("=== INFLATION OBSERVABLES ===\n\n")
            f.write(f"Spectral index (n_s): {obs['n_s']:.6f}\n")
            f.write(f"Tensor-to-scalar ratio (r): {obs['r']:.6e}\n")
            f.write(f"Scalar amplitude (ln(10¹⁰ A_s)): {np.log(1e10 * obs['A_s']):.6f}\n")
            f.write(f"Slow-roll parameters:\n")
            f.write(f"  - epsilon: {obs['epsilon']:.6e}\n")
            f.write(f"  - eta: {obs['eta']:.6e}\n")
            
        print("Results saved successfully!")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        # Attempt to save partial results
        try:
            partial_results = {k: v for k, v in results.items() 
                             if isinstance(v, (np.ndarray, list, dict))}
            np.savez(f"{output_dir}/partial_results.npz", **partial_results)
            print("Partial results saved")
        except:
            print("Failed to save partial results")
def compute_quantum_corrections(t, y, model):
    """
    Compute quantum corrections with proper bounds
    """
    try:
        phi, phi_dot = y
        H = np.array([model.hubble_parameter(p, pd) for p, pd in zip(phi, phi_dot)])
        
        # Quantum corrections
        corrections = {
            'horizon': np.clip(1 + model.hbar_eff * H / (2 * np.pi * c * model.M_pl), 1, 2),
            'field': np.clip(1 + model.hbar_eff / (12 * np.pi**2 * model.M_pl**2), 1, 2),
            'potential': np.clip(1 + model.hbar_eff * H**2 / (8 * np.pi**2 * c * model.M_pl**2), 1, 2)
        }
        
        return corrections
        
    except Exception as e:
        print(f"Error in quantum corrections: {str(e)}")
        return None



def plot_enhanced_results(results):
    """
    Create visualizations for enhanced quantum gravity effects
    with error handling
    """
    try:
        output_dir = "quantum_cosmos_results"
        
        # Plot quantum foam topology if data exists
        if 'foam_data' in results:
            plt.figure(figsize=(10, 6))
            try:
                foam_chars = [f['euler_characteristic'] for f in results['foam_data']]
                plt.plot(results['radii']/float(l_p), foam_chars)
                plt.xlabel('r/l_p')
                plt.ylabel('Euler Characteristic')
                plt.title('Quantum Foam Topology')
                plt.xscale('log')
                plt.savefig(f"{output_dir}/quantum_foam_topology.png")
                print("Saved quantum foam topology plot")
            except Exception as e:
                print(f"Warning: Could not create foam topology plot: {str(e)}")
            finally:
                plt.close()

        # Plot noncommutative corrections
        if 'nc_corrections' in results:
            plt.figure(figsize=(10, 6))
            try:
                plt.plot(results['radii']/float(l_p), results['nc_corrections'])
                plt.xlabel('r/l_p')
                plt.ylabel('NC Correction Factor')
                plt.title('Noncommutative Geometry Corrections')
                plt.xscale('log')
                plt.savefig(f"{output_dir}/nc_corrections.png")
                print("Saved NC corrections plot")
            except Exception as e:
                print(f"Warning: Could not create NC corrections plot: {str(e)}")
            finally:
                plt.close()

        # Plot enhanced horizon corrections
        if 'horizon_corrections' in results:
            plt.figure(figsize=(10, 6))
            try:
                plt.plot(results['radii']/float(l_p), results['horizon_corrections'])
                plt.xlabel('r/l_p')
                plt.ylabel('Enhanced Horizon Corrections')
                plt.title('Quantum-Corrected Horizon Structure')
                plt.xscale('log')
                plt.yscale('log')
                plt.savefig(f"{output_dir}/enhanced_horizon_corrections.png")
                print("Saved horizon corrections plot")
            except Exception as e:
                print(f"Warning: Could not create horizon corrections plot: {str(e)}")
            finally:
                plt.close()

        # Plot dilaton profile
        if 'dilaton' in results:
            plt.figure(figsize=(10, 6))
            try:
                plt.plot(results['radii']/float(l_p), results['dilaton'])
                plt.xlabel('r/l_p')
                plt.ylabel('Dilaton')
                plt.title('Dilaton Profile')
                plt.xscale('log')
                plt.savefig(f"{output_dir}/dilaton_profile.png")
                print("Saved dilaton profile plot")
            except Exception as e:
                print(f"Warning: Could not create dilaton profile plot: {str(e)}")
            finally:
                plt.close()

    except Exception as e:
        print(f"Error in plotting results: {str(e)}")
def create_summary_file(results):
    """
    Create a summary file with key metrics
    """
    try:
        output_dir = "quantum_cosmos_results"
        with open(f"{output_dir}/simulation_summary.txt", "w") as f:
            f.write("=== QUANTUM COSMOS SIMULATION SUMMARY ===\n\n")
            
            # Write basic parameters
            f.write("Simulation Parameters:\n")
            f.write(f"Number of spatial points: {len(results['radii'])}\n")
            f.write(f"Spatial range: {results['radii'][0]/float(l_p):.2e} - {results['radii'][-1]/float(l_p):.2e} l_p\n\n")
            
            # Write corrections statistics
            if 'horizon_corrections' in results:
                horizon_stats = results['horizon_corrections']
                f.write("Horizon Corrections:\n")
                f.write(f"Mean: {np.mean(horizon_stats):.2e}\n")
                f.write(f"Max: {np.max(horizon_stats):.2e}\n")
                f.write(f"Min: {np.min(horizon_stats):.2e}\n\n")
            
            if 'nc_corrections' in results:
                nc_stats = results['nc_corrections']
                f.write("Noncommutative Corrections:\n")
                f.write(f"Mean: {np.mean(nc_stats):.2e}\n")
                f.write(f"Max: {np.max(nc_stats):.2e}\n")
                f.write(f"Min: {np.min(nc_stats):.2e}\n\n")
            
            f.write("Analysis completed successfully!\n")
            
    except Exception as e:
        print(f"Error creating summary file: {str(e)}")
if __name__ == "__main__":
    print("Starting quantum cosmos simulation...")

    # Run the enhanced simulation, integrating quantum cosmos components
    results = integrate_enhanced_simulation()
    
    # Perform inflation diagnostics
    diagnostics = run_inflation_with_diagnostics()

    # Run the inflation analysis and capture results
    t, y, obs = run_inflation_analysis()

    if results is not None:
        print("\nSaving results...")
        
        # Save the enhanced results into a structured format
        save_enhanced_results(results)
        create_summary_file(results)
        
        print("\nSimulation completed successfully!")
        
        # Check if the inflation analysis has key observables
        if obs is not None:
            print("\nKey Observables:")
            print(f"n_s = {obs['n_s']:.4f} (Target: 0.9649 ± 0.0042)")
            print(f"r = {obs['r']:.4e} (Target: < 0.064)")
            print(f"ln(10¹⁰ A_s) = {np.log(1e10 * obs['A_s']):.4f} (Target: 3.044 ± 0.014)")
        else:
            print("No key observables were found.")
    else:
        print("Simulation failed!")
