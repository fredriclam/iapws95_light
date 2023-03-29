''' Ordinary water substance (IAPWS95) properties based on Wagner and Pruss
(2002) in J. Phys. Chem. Ref. Data.
  
Computations are typically done nondimensionally, and in SI units otherwise.
  '''

import csv
import numpy as np
import scipy.optimize
from time import perf_counter

''' Module-level data loading ''' 

DATAPATH_IDEAL = 'ideal.csv'
DATAPATH_RES0 = 'residual_1_51.csv'
DATAPATH_RES1 = 'residual_52_54.csv'
DATAPATH_RES2 = 'residual_55_56.csv'

''' Load data ''' 
with open(DATAPATH_IDEAL, newline='') as csvfile:
  reader = csv.DictReader(csvfile, delimiter='\t')
  ideal_arraydict = [row for row in reader]
with open(DATAPATH_RES0, newline='') as csvfile:
  reader = csv.DictReader(csvfile, delimiter='\t')
  residual0_arraydict = [row for row in reader]
with open(DATAPATH_RES1, newline='') as csvfile:
  reader = csv.DictReader(csvfile, delimiter='\t')
  residual1_arraydict = [row for row in reader]
with open(DATAPATH_RES2, newline='') as csvfile:
  reader = csv.DictReader(csvfile, delimiter='\t')
  residual2_arraydict = [row for row in reader]

''' Data prep '''
# Compile coefficients for ideal terms
n_ideal = np.array([float(obj["n_i"]) for obj in ideal_arraydict])
g_ideal = np.array([float(obj["g_i"]) for obj in ideal_arraydict])
# Compile shared coefficients for all residual terms indexed from start to end
n_res = np.hstack(([float(obj["n_i"]) for obj in residual0_arraydict],
  [float(obj["n_i"]) for obj in residual1_arraydict],
  [float(obj["n_i"]) for obj in residual2_arraydict]))
d_res = np.hstack(([float(obj["d_i"]) for obj in residual0_arraydict],
  [float(obj["d_i"]) for obj in residual1_arraydict],
  [1.0 for obj in residual2_arraydict]))
t_res = np.hstack(([float(obj["t_i"]) for obj in residual0_arraydict],
  [float(obj["t_i"]) for obj in residual1_arraydict],
  [0.0 for obj in residual2_arraydict]))
# Compile coefficients for residual terms 1 to 51
c_res1_51 = np.array([float(obj["c_i"]) for obj in residual0_arraydict])
# Compile coefficients for residual terms 52 to 54
c_res52_54 = np.array([float(obj["c_i"]) for obj in residual1_arraydict])
alpha_res52_54 = np.array([float(obj["alpha_i"]) for obj in residual1_arraydict])
beta_res52_54 = np.array([float(obj["beta_i"]) for obj in residual1_arraydict])
gamma_res52_54 = np.array([float(obj["gamma_i"]) for obj in residual1_arraydict])
eps_res52_54 = np.array([float(obj["eps_i"]) for obj in residual1_arraydict])
# Compile coefficients for residual terms 55 to 56
a_res55_56 = np.array([float(obj["a_i"]) for obj in residual2_arraydict])
b_res55_56 = np.array([float(obj["b_i"]) for obj in residual2_arraydict])
B_res55_56 = np.array([float(obj["B_i"]) for obj in residual2_arraydict])
C_res55_56 = np.array([float(obj["C_i"]) for obj in residual2_arraydict])
D_res55_56 = np.array([float(obj["D_i"]) for obj in residual2_arraydict])
A_res55_56 = np.array([float(obj["A_i"]) for obj in residual2_arraydict])
beta_res55_56 = np.array([float(obj["beta_i"]) for obj in residual2_arraydict])

''' Set up Saul and Wagner 1987 saturation curve estimates '''
# Saturated liquid density correlation (Eq. 2.3)
_powsb = np.array([1/3, 2/3, 5/3, 16/3, 43/3, 110/3])
_coeffsb = [1.99206, 1.10123, -5.12506e-1, -1.75263, -45.4485, -6.75615e5]
d_satl = lambda t: 1.0 + np.dot((1.0-1.0/t)**_powsb, _coeffsb)
# Saturated vapour density correlation (Eq. 2.2)
_powsc = np.array([1/3, 2/3, 4/3, 9/3, 37/6, 71/6])
_coeffsc = [-2.02957, -2.68781, -5.38107, -17.3151, -44.6384, -64.3486]
d_satv = lambda t: np.exp(np.dot((1.0-1.0/t)**_powsc, _coeffsc))

''' Set up static parameters '''
Tc = 647.096  # K
rhoc = 322    # kg / m^3
R = 0.46151805 * 1e3 # J / kg K
# Generic precomputation
_exp1_55_56 = 0.5 / (beta_res55_56)

''' Function dispatch utilities. '''

def autocast(f):
  ''' Wrapper for functions of two arguments to take type float or np.array. '''
  def wrapped(rho, T):
    rho = np.array(rho)
    T = np.array(T)
    if rho.shape != T.shape:
      raise ValueError("Dimensions of rho, T do not match.")
    if len(rho.shape) == 0:
      # Pad and return to scalar
      return float(f(np.expand_dims(rho, axis=-1), np.expand_dims(T, axis=-1)))
    elif rho.shape[-1] != 1:
      # Add axis and squeeze out
      return f(np.expand_dims(rho, axis=-1),
        np.expand_dims(T, axis=-1)).squeeze(axis=-1)
    else:
      return f(rho, T)
  return wrapped

def dispatch_vec(functions, rho, T):
  ''' Wraps and dispatches an iterable collection functions. ''' 
  # Reciprocal reduced volume
  d = np.array(rho / rhoc)
  # Reciprocal reduced temperature
  t = np.array(Tc / T)
  if d.shape != t.shape:
    raise ValueError("Dimensions of rho, T do not match.")
  # Padding for a reduction dimension for model coefficients
  if len(d.shape) == 0:
    # Pad and return to scalar
    return [float(f(np.expand_dims(d,axis=-1),
      np.expand_dims(t,axis=-1))) for f in functions]
  elif d.shape[-1] != 1:
    # Add axis and squeeze out
    return [f(np.expand_dims(d,axis=-1),
      np.expand_dims(t,axis=-1)).squeeze(axis=-1) for f in functions]
  else:
    return [f(d,t) for f in functions]

''' Vectorized thermodynamic functions in (rho, T) space. '''

@autocast
def Z(rho:np.array, T:np.array):
  ''' Compressibility factor as a function of dimensional density rho,
  temperature T.'''
  # Reciprocal reduced volume and temperature
  d, t = np.array(rho / rhoc), np.array(Tc / T)
  return 1.0 + d * phir_d(d, t)

@autocast
def p(rho:np.array, T:np.array):
  ''' Pressure as a function of dimensional density rho, temperature T.
  If rho_satv(T) < rho < rho_satl(T), the returned value may not be accurate.
  Due to the parametrization of the Helmholtz function, the pressure, which is a
  function of derivatives of the Helmholtz function, is not well-behaved.
  A more reliable representation of p is obtained by computing the saturation
  pressure by Maxwell's construction (see function
  iapws95_light.prho_sat).
  
  For performance, this function does not check whether the inputs are in
  the mixed phase regime. '''
  # Reciprocal reduced volume and temperature
  d, t = np.array(rho / rhoc), np.array(Tc / T)
  return rho * R * T * (1.0 + d * phir_d(d, t))

''' Scalar (iterative) thermodynamic functions. '''

def rho_pt(p:float, T:float, newton_rtol=1e-9, max_steps=16):
  ''' Density computed iteratively using Newton's method. '''
  t = Tc / T
  damping_factor = 1.0
  # Determine initial value
  psat, rho_satl, rho_satv = prho_sat(T)
  if p > psat:
    # Set initial value to saturated liquid density
    d = rho_satl / rhoc
  elif p < psat:
    # Set initial value to saturated vapour density
    d = rho_satv / rhoc
  else:
    raise ValueError(f"Saturation state p = {p} Pa, T = {T} K entered; " +
      "density is ambiguous.")

  # Refine d solution using Newton's method
  for i in range(max_steps):
    # Evaluate residual value, derivative
    _phir_d = phir_d(d, t)
    val = -p / (rhoc * R * T) + d * (1.0 + d * _phir_d)
    deriv = 1.0 + d * (2.0 * _phir_d + d * phir_dd(d, t))
    # Take Newton step
    step = -damping_factor * val / deriv
    d += step
    # Check termination condition
    if np.abs(step/d) < newton_rtol:
      break
  return d * rhoc

def prho_sat(T):
  ''' Returns isothermal saturation curve properties as tuple
  (psat, rho_satl, rho_satv). Solves the Maxwell construction (see e.g. P.
  Junglas). '''
  # Compute reciprocal reduced temperature
  t = Tc / T
  if t < 1:
    return None, None, None

  # Define size-2 system for Maxwell construction for phase transition at T
  _phir_d = lambda d: float(phir_d(d, t))
  _phir = lambda d: float(phir(d, t))
  _phi0 = lambda d: float(phi0(d, t))
  eq1 = lambda d1, d2: d2 * _phir_d(d2) - d1 * _phir_d(d1) \
    - _phir(d1) - _phi0(d1) + _phir(d2) + _phi0(d2)
  eq2 = lambda d1, d2: d1 + d1**2 * _phir_d(d1) - d2 - d2**2 * _phir_d(d2)
  eqvec = lambda d: np.array([eq1(d[0], d[1]), eq2(d[0], d[1])])

  # Jacobian
  # [ -2 phir_d(d1) - d1 * phir_dd(d1) - phi0_d(d1),  + for d2 ]
  # [  1 + 2*d1 * phir_d(d1) + d1**2 * phir_dd(d1),  - for d2 ]
  _phir_dd = lambda d: float(phir_dd(d, t))
  _phi0_d = lambda d: float(phi0_d(d, t))
  def jac(d):
    _cache_phir_d0 = _phir_d(d[0])
    _cache_phir_d1 = _phir_d(d[1])
    _cache_phir_dd0 = _phir_dd(d[0])
    _cache_phir_dd1 = _phir_dd(d[1])
    return np.array(
      [[-2.0 * _cache_phir_d0 - d[0] * _cache_phir_dd0 - _phi0_d(d[0]),
        2.0 * _cache_phir_d1 + d[1] * _cache_phir_dd1 + _phi0_d(d[1])],
      [1.0 + 2.0*d[0] * _cache_phir_d0 + d[0]**2 * _cache_phir_dd0,
        -1.0 - 2.0*d[1] * _cache_phir_d1 - d[1]**2 * _cache_phir_dd1]])

  # Solve system using fsolve, initial guess using older sat curve correlations
  d_init = np.array([float(d_satl(t)), float(d_satv(t))])
  # d_vec = scipy.optimize.fsolve(eqvec, d_init, fprime=jac)

  # Two-step Newton
  def newton_step(d):
    ''' Compute Newton step. '''
    _cache_phir_d0 = _phir_d(d[0])
    _cache_phir_d1 = _phir_d(d[1])
    _cache_phir_dd0 = _phir_dd(d[0])
    _cache_phir_dd1 = _phir_dd(d[1])
    f = np.array([d[1] * _cache_phir_d1 - d[0] * _cache_phir_d0 \
      - _phir(d[0]) - _phi0(d[0]) + _phir(d[1]) + _phi0(d[1]),
      d[0] + d[0]**2 * _cache_phir_d0 - d[1] - d[1]**2 * _cache_phir_d1])
    J = np.array(
      [[-2.0 * _cache_phir_d0 - d[0] * _cache_phir_dd0 - _phi0_d(d[0]),
        2.0 * _cache_phir_d1 + d[1] * _cache_phir_dd1 + _phi0_d(d[1])],
      [1.0 + 2.0*d[0] * _cache_phir_d0 + d[0]**2 * _cache_phir_dd0,
        -1.0 - 2.0*d[1] * _cache_phir_d1 - d[1]**2 * _cache_phir_dd1]])
    detJ = J[0,0] * J[1,1] - J[0,1] * J[1,0]
    # Update d with -J^{-1} f
    return -np.array([J[1,1] * f[0] - J[0,1] * f[1],
      -J[1,0] * f[0] + J[0,0] * f[1]]) / detJ

  d_vec = d_init
  d_vec += newton_step(d_vec)
  final_step = newton_step(d_vec)
  # print(np.linalg.norm(final_step))
  d_vec += final_step

  # Get saturation densities at given T
  rho_satl, rho_satv = d_vec * rhoc
  # Compute saturation pressure (use either d_final[0] or d_final[1])
  psat = d_vec[0]*(1.0 + d_vec[0]*_phir_d(d_vec[0])) \
    * rhoc * R * T
  return np.array([psat]), rho_satl, rho_satv

def x(rho, T):
  ''' Vapour mass fraction (steam quality). '''
  psat, rho_satl, rho_satv = prho_sat(T)
  if rho <= rho_satv:
    # Saturated vapour
    return 1.0
  elif rho_satv < rho and rho < rho_satl:
    # Mixture
    return (rho - rho_satl)/(rho_satv - rho_satl)
  else:
    # Saturated liquid
    return 0.0

''' Verification utilities. '''

def print_verification_values():
  ''' Prints verification values (p. 436 of Wagner and Pruss)'''
  names0 = ["phi0   ", "phi0_d ", "phi0_dd", "phi0_t ", "phi0_tt", "phi0_dt"]
  namesr = ["phir   ", "phir_d ", "phir_dd", "phir_t ", "phir_tt", "phir_dt"]

  def print_table(results0, resultsr):
    ''' Prints results0, resultsr (each length 6) into a fixed format table.'''
    print("===============================================")
    for tup in zip(names0, results0, namesr, resultsr):
      print(f"{tup[0]} | " +
        f"{tup[1]:{'.9f' if tup[1] < 0 else ' .9f'}} | " +
        f"{tup[2]} | " +
        f"{tup[3]:{'.9f' if tup[3] < 0 else ' .9f'}}")
  
  print("Test case 1: rho = 838.025 kg m^{-3}, T = 500 K")
  rho = 838.025
  T = 500
  # Evaluate results using this implementation
  results0 = dispatch_vec([
    phi0, phi0_d, phi0_dd, phi0_t, phi0_tt, phi0_dt], rho, T)
  resultsr = dispatch_vec([
    phir, phir_d, phir_dd, phir_t, phir_tt, phir_dt], rho, T)
  # Print results
  print("Computed: ")
  print_table(results0, resultsr)
  print("Reference (9 significant figures): ")
  ref0 = [0.204_797_734e1, 0.384_236_747, -0.147_637_878,
          0.904_611_106e1, -0.193_249_185e1, 0]
  refr = [-0.342_693_206e1, -0.364_366_650, 0.856_063_701,
        -0.581_403_435e1, -0.223_440_737e1, -0.112_176_915e1]
  print_table(ref0, refr)

  print("")
  print("Test case 2: rho = 358 kg m^{-3}, T = 647 K")
  rho = 358
  T = 647
  # Evaluate results using this implementation
  results0 = dispatch_vec([
    phi0, phi0_d, phi0_dd, phi0_t, phi0_tt, phi0_dt], rho, T)
  resultsr = dispatch_vec([
    phir, phir_d, phir_dd, phir_t, phir_tt, phir_dt], rho, T)
  # Print results
  print("Computed: ")
  print_table(results0, resultsr)
  print("Reference (9 significant figures): ")
  ref0 = [-0.156_319_605e1, 0.899_441_341, -0.808_994_726,
    0.980_343_918e1, -0.343_316_334e1, 0]
  refr = [-0.121_202_657e1, -0.714_012_024, 0.475_730_696,
    -0.321_722_501e1, -0.996_029_507e1, -0.133_214_720e1]
  print_table(ref0, refr)

def print_timing():
  ''' Prints timing values as compared to ideal gas computations. '''
  print(f"Timing p(rho, T) calculations for scalar input.")
  rho = 358
  T = 647
  # Timing for pressure evaluation
  N_timing = 36000
  t1 = perf_counter()
  for i in range(N_timing):
    d = rho / rhoc
    t = Tc / T
    p = (1.0 + d * phir(d, t)) * rho * R * T
  t2 = perf_counter()

  t1_ideal = perf_counter()
  for i in range(N_timing):
    p_ideal = rho * R * T
  t2_ideal = perf_counter()

  print(f"iapws95_light: {(t2-t1)/N_timing * 1e6} us")
  print(f"Ideal gas    : {(t2_ideal-t1_ideal)/N_timing * 1e6} us")
  print(f"Relative load: {(t2-t1)/(t2_ideal-t1_ideal):.3f}x")

  num_coeffs = 44*4 + 21 + 21 + 16 + 13
  print(f"=== Additional details ===")
  print(f"Number of coefficients in model: {num_coeffs}")
  print(f"Relative load per model dof:     " +
        f"{(t2-t1)/(t2_ideal-t1_ideal)/num_coeffs:.3f}x")

def get_saturation_density_curves(range_T=None):
  ''' Returns saturation density curves rho_satl(T), rho_satv(T). '''
  if range_T is None:
    range_T = np.linspace(273.15, Tc, 60)
  # Sample saturation curve
  range_rho_l, range_rho_v = zip(*[prho_sat(T)[1:3] for T in range_T])
  return np.array(range_rho_l), np.array(range_rho_v)

''' Ideal-gas part of dimensionless Helmholtz function and its derivatives. '''
  
def phi0(d:np.array, t:np.array) -> np.array:
  ''' Ideal-gas part of dimless Helmholtz function
      phi = f/(RT). '''
  return (np.log(d) + n_ideal[0] + n_ideal[1] * t + n_ideal[2] * np.log(t)
    + np.expand_dims(np.einsum("i, ...i -> ...",
    n_ideal[3:8], np.log(1.0 - np.exp(-g_ideal[3:8] * t))), axis=-1))

def phi0_d(d:np.array, t:np.array) -> np.array:
  ''' First delta-derivative of ideal-gas part of dimless Helmholtz function
      phi = f/(RT). '''
  return 1.0/d

def phi0_dd(d:np.array, t:np.array) -> np.array:
  ''' Second delta-derivative of ideal-gas part of dimless Helmholtz function
      phi = f/(RT). '''
  return -1.0/d**2

def phi0_t(d:np.array, t:np.array) -> np.array:
  ''' First tau-derivative of ideal-gas part of dimless Helmholtz function
      phi = f/(RT). '''
  return (n_ideal[1] + n_ideal[2] / t
    + np.expand_dims(np.einsum("i, ...i -> ...", n_ideal[3:8] * g_ideal[3:8],
    1.0/(1.0 - np.exp(-g_ideal[3:8] * t)) - 1.0), axis=-1))

def phi0_tt(d:np.array, t:np.array) -> np.array:
  ''' Second tau-derivative of ideal-gas part of dimless Helmholtz function
      phi = f/(RT). '''
  exp_result = np.exp(-g_ideal[3:8] * t)
  return (-n_ideal[2] / t**2
    + np.expand_dims(np.einsum("i, ...i -> ...",
      -n_ideal[3:8] * g_ideal[3:8]**2,
      exp_result/(1.0 - exp_result)**2), axis=-1))

def phi0_dt(d:np.array, t:np.array) -> np.array:
  ''' Mixed second derivative of ideal-gas part of dimless Helmholtz function
      phi = f/(RT). '''
  return np.zeros_like(d)

''' Residual part of dimensionless Helmholtz function and its derivatives. '''

def phir(d:np.array, t:np.array) -> np.array:
  ''' Residual part of dimless Helmholtz function
      phi = f/(RT).
  Evaluated primarily using two registers that combine as
    np.dot( coeffs, np.exp(exponents) ).
  Derivatives of phi typically require extra memory. '''
  # Precompute quantities
  d_quad = (d-1.0)**2
  # Allocate and evaluate coeffs
  coeffs = n_res * (d ** d_res) * (t ** t_res)
  # Compute distance term for 1-indices 55 to 56
  theta = (1 - t) + A_res55_56 * d_quad ** _exp1_55_56
  Delta = theta**2 + B_res55_56 * d_quad ** a_res55_56
  # Factor in Delta**b_i term for 1-indices from 55 to 56
  coeffs[...,54:56] *= Delta ** b_res55_56
  
  # Allocate exponent cache
  exponents = np.zeros_like(coeffs)
  # Compute exponents for 1-indices 8 to 51 as -d**c_i
  exponents[...,7:51] = -d ** c_res1_51[7:51]
  # Compute exponents for 1-indices from 52 to 54
  exponents[...,51:54] = -alpha_res52_54 * (d - eps_res52_54) ** 2 \
    -beta_res52_54*(t - gamma_res52_54)**2
  # Compute exponents for 1-indices from 55 to 56
  exponents[...,54:56] = -C_res55_56 * d_quad \
    -D_res55_56*(t - 1)**2

  return np.expand_dims(np.einsum("...i, ...i -> ...",
    coeffs, np.exp(exponents)), axis=-1)

def phir_d(d:np.array, t:np.array) -> np.array:
  ''' First delta-derivative of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details. '''
  # Precompute quantities
  d_quad = (d-1.0)**2
  # Allocate and partially evaluate coeffs
  coeffs = n_res * (d ** (d_res-1.0)) * (t ** t_res)
  # Factor in d_i - c_i * d**c_i term
  coeffs[...,0:51] *= (d_res[0:51] - c_res1_51 * d ** c_res1_51)
  coeffs[...,51:54] *= d_res[51:54] - 2.0 * alpha_res52_54 * d * (d - eps_res52_54)
  # Compute distance term for 1-indices 55 to 56
  theta = (1 - t) + A_res55_56 * d_quad ** _exp1_55_56
  Delta = theta**2 + B_res55_56 * d_quad ** a_res55_56
  # Factor in other terms for 1-indices from 55 to 56 in two steps
  coeffs[...,54:56] *= (
    Delta * (1.0 - 2.0 * C_res55_56 * (d-1.0) * d)
    + b_res55_56 * d * (d-1.0) * (
      A_res55_56 * theta * 2 / beta_res55_56 * d_quad**(_exp1_55_56 - 1.0)
      + 2 * B_res55_56 * a_res55_56 * d_quad**(a_res55_56 - 1.0)
    )
  )
  coeffs[...,54:56] *= Delta ** np.where(Delta != 0, b_res55_56-1, 1.0)

  # Allocate exponent cache
  exponents = np.zeros_like(coeffs)
  # Compute exponents for 1-indices 8 to 51 as -d**c_i
  exponents[...,7:51] = -d ** c_res1_51[7:51]
  # Compute exponents for 1-indices from 52 to 54
  exponents[...,51:54] = -alpha_res52_54 * (d - eps_res52_54) ** 2 \
    -beta_res52_54*(t - gamma_res52_54)**2
  # Compute exponents for 1-indices from 55 to 56
  exponents[...,54:56] = -C_res55_56 * d_quad \
    -D_res55_56*(t - 1)**2
  # print(coeffs * np.exp(exponents))
  return np.expand_dims(np.einsum("...i, ...i -> ...",
    coeffs, np.exp(exponents)), axis=-1)

def phir_dd(d:np.array, t:np.array) -> np.array:
  ''' Second delta-derivative of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details. '''
  # Scalar precomputation
  d_quad = (d-1.0)**2
  # Allocate and partially evaluate coeffs
  coeffs = n_res * (d ** (d_res-2.0)) * (t ** t_res)
  # Temporary space
  cdc = c_res1_51 * (d ** c_res1_51)
  # Factor for 1-indices 1 to 51
  coeffs[...,0:51] *= (d_res[0:51] - cdc) * (d_res[0:51] - 1.0 - cdc) \
    - c_res1_51 * cdc
  # Factor for 1-indices 52 to 54
  coeffs[...,51:54] *= -2 * alpha_res52_54 * d**2 \
    + 4 * alpha_res52_54**2 * d**2 * (d - eps_res52_54)**2 \
    - 4 * d_res[51:54] * alpha_res52_54 * d * (d - eps_res52_54) \
    + d_res[51:54] * (d_res[51:54] - 1.0)
  # Compute distance term for 1-indices 55 to 56
  theta = (1.0 - t) + A_res55_56 * d_quad ** _exp1_55_56
  Delta = theta**2 + B_res55_56 * d_quad ** a_res55_56
  # Compute d(Delta)/d(delta) divided by (delta - 1.0) for numerical stability
  dDelta_div = (
    A_res55_56 * theta * 2.0 / beta_res55_56 * d_quad**(_exp1_55_56 - 1.0)
    + 2 * B_res55_56 * a_res55_56 * d_quad**(a_res55_56 - 1.0)
  )
  # Set power to non-negative when argument is negative; the limit is finite
  limited_power = np.where(d_quad != 0, _exp1_55_56 - 2.0, 1.0)
  ddDelta = dDelta_div + ((d-1.0)**2) * (
    4.0 * B_res55_56 * a_res55_56 * (a_res55_56 - 1.0)
    * d_quad**(a_res55_56 - 2.0)
    + 2.0 * (A_res55_56 / beta_res55_56 * d_quad**(_exp1_55_56 - 1.0))**2.0
    + 4.0 * theta * A_res55_56 / beta_res55_56 * (_exp1_55_56 - 1.0)
    * d_quad**limited_power
  )
  # Finish d(Delta)/d(delta) computation
  dDelta = (d-1.0) * dDelta_div
  # Replace (t_res is zero, so coeffs[54:56] contains invalid entries) for
  #   1-indices from 55 to 56
  coeffs[...,54:56] = Delta**2 * (-4 * C_res55_56 * (d-1.0) 
    + d * (2*C_res55_56*d_quad - 1.0) * 2.0 * C_res55_56)
  coeffs[...,54:56] += Delta * 2.0 * b_res55_56 * dDelta \
    * (1.0 - 2.0 * d * C_res55_56 * (d - 1.0))
  coeffs[...,54:56] += b_res55_56 * (Delta * ddDelta
    + (b_res55_56 - 1.0) * dDelta**2) * d
  coeffs[...,54:56] *= n_res[54:56] \
    * Delta ** np.where(Delta != 0, b_res55_56 - 2.0, 1.0)

  # Allocate exponent cache
  exponents = np.zeros_like(coeffs)
  # Compute exponents for 1-indices 8 to 51 as -d**c_i
  exponents[...,7:51] = -cdc[7:51] / c_res1_51[7:51]
  # Compute exponents for 1-indices from 52 to 54
  exponents[...,51:54] = -alpha_res52_54 * (d - eps_res52_54) ** 2 \
    -beta_res52_54*(t - gamma_res52_54)**2
  # Compute exponents for 1-indices from 55 to 56
  exponents[...,54:56] = -C_res55_56 * d_quad \
    -D_res55_56*(t - 1.0)**2

  # Reduce
  return np.expand_dims(np.einsum("...i, ...i -> ...",
    coeffs, np.exp(exponents)), axis=-1)

def phir_t(d:np.array, t:np.array) -> np.array:
  ''' First tau-derivative of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details. '''
  # Scalar precomputation
  d_quad = (d-1.0)**2
  # Allocate and partially evaluate coeffs
  coeffs = n_res * (d ** d_res) * (t ** (t_res-1.0))
  # Factor for 1-indices 1 to 51
  coeffs[...,0:51] *= t_res[0:51]
  # Factor in d_i - c_i * d**c_i term for 1-indices 52 to 54
  coeffs[...,51:54] *= t_res[51:54] - 2.0 * beta_res52_54 * t * (t - gamma_res52_54)
  # Compute distance term for 1-indices 55 to 56
  theta = (1 - t) + A_res55_56 * d_quad ** _exp1_55_56
  Delta = theta**2 + B_res55_56 * d_quad ** a_res55_56
  # Replace (t_res is zero, so coeffs[54:56] contains invalid entries) for
  #   1-indices from 55 to 56 in two steps
  coeffs[...,54:56] = n_res[54:56] * d * 2.0 * (
    -theta * b_res55_56 + Delta * D_res55_56 * (1.0 - t))
  coeffs[...,54:56] *= Delta ** np.where(Delta != 0, b_res55_56-1, 1.0)

  # Allocate exponent cache
  exponents = np.zeros_like(coeffs)
  # Compute exponents for 1-indices 8 to 51 as -d**c_i
  exponents[...,7:51] = -d ** c_res1_51[7:51]
  # Compute exponents for 1-indices from 52 to 54
  exponents[...,51:54] = -alpha_res52_54 * (d - eps_res52_54) ** 2 \
    -beta_res52_54*(t - gamma_res52_54)**2
  # Compute exponents for 1-indices from 55 to 56
  exponents[...,54:56] = -C_res55_56 * d_quad \
    -D_res55_56*(t - 1)**2

  # Reduce
  return np.expand_dims(np.einsum("...i, ...i -> ...",
    coeffs, np.exp(exponents)), axis=-1)

def phir_tt(d, t):
  ''' Second derivative tt of reduced Helmholtz function with respect to recip. reduced
  temperature.
  '''

  # Scalar precomputation
  d_quad = (d-1)**2
  # Allocate and partially evaluate coeffs
  coeffs = n_res * (d ** d_res) * (t ** (t_res-2.0))
  # Factor for 1-indices 1 to 51
  coeffs[...,0:51] *= t_res[0:51] * (t_res[0:51] - 1.0)
  # Factor for 1-indices 52 to 54
  coeffs[...,51:54] *= (t_res[51:54] - 2.0 * beta_res52_54 * t * 
    (t - gamma_res52_54))**2 - t_res[51:54] - 2.0 * beta_res52_54 * t**2

  # Compute distance term for 1-indices 55 to 56
  theta = (1.0 - t) + A_res55_56 * d_quad ** _exp1_55_56
  Delta = theta**2 + B_res55_56 * d_quad ** a_res55_56
  # Replace (t_res is zero, so coeffs[54:56] contains invalid entries) for
  #   1-indices from 55 to 56 in two steps
  coeffs[...,54:56] = n_res[54:56] * d * (
    2.0 * b_res55_56 * (Delta + 2.0 * theta**2 * (b_res55_56 - 1.0)
    + 4.0 * theta * Delta * D_res55_56 * (t - 1.0))
    + Delta ** 2 * 2.0 * D_res55_56 * (2.0*D_res55_56 * (t - 1.0)**2 - 1.0)
  )
  coeffs[...,54:56] *= Delta ** np.where(Delta != 0, b_res55_56 - 2.0, 1.0)
  # Set phir_tt at rho == 1 to -inf gracefully
  coeffs[...,54:56] = np.where(Delta != 0, coeffs[...,54:56], -np.inf)

  # Allocate exponent cache
  exponents = np.zeros_like(coeffs)
  # Compute exponents for 1-indices 8 to 51 as -d**c_i
  exponents[...,7:51] = -d ** c_res1_51[7:51]
  # Compute exponents for 1-indices from 52 to 54
  exponents[...,51:54] = -alpha_res52_54 * (d - eps_res52_54) ** 2 \
    -beta_res52_54*(t - gamma_res52_54)**2
  # Compute exponents for 1-indices from 55 to 56
  exponents[...,54:56] = -C_res55_56 * d_quad \
    -D_res55_56*(t - 1.0)**2

  # Reduce
  return np.expand_dims(np.einsum("...i, ...i -> ...",
    coeffs, np.exp(exponents)), axis=-1)

def phir_dt(d:np.array, t:np.array) -> np.array:
  ''' Mixed second derivative of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details. '''
  # Scalar precomputation
  d_quad = (d-1.0)**2
  # Allocate and partially evaluate coeffs
  coeffs = n_res * (d ** (d_res-1.0)) * (t ** (t_res-1.0))
  dc = d ** c_res1_51
  # Factor for 1-indices 1 to 51
  coeffs[...,0:51] *= t_res[0:51] * (d_res[0:51] - c_res1_51 * dc)
  # Factor for 1-indices 52 to 54
  coeffs[...,51:54] *= d_res[51:54] \
    - 2.0 * alpha_res52_54 * d * (d - eps_res52_54)
  coeffs[...,51:54] *= t_res[51:54] \
    - 2.0 * beta_res52_54 * t * (t - gamma_res52_54)
  # Compute distance term for 1-indices 55 to 56
  theta = (1 - t) + A_res55_56 * d_quad ** _exp1_55_56
  Delta = theta**2 + B_res55_56 * d_quad ** a_res55_56
  # Compute d(Delta)/d(delta)
  dDelta = (d-1.0) * (
    A_res55_56 * theta * 2.0 / beta_res55_56 * d_quad**(_exp1_55_56 - 1.0)
    + 2.0 * B_res55_56 * a_res55_56 * d_quad**(a_res55_56 - 1.0)
  )
  # Replace (t_res is zero, so coeffs[54:56] contains invalid entries) for
  #   1-indices from 55 to 56 in two steps
  coeffs[...,54:56] = n_res[54:56] * (
    Delta**2 * (-2.0 * D_res55_56 * (t - 1.0) + d * 4.0 * C_res55_56 *
    D_res55_56 * (d - 1.0) * (t - 1.0))
    + d * Delta * b_res55_56 * dDelta * (-2.0 * D_res55_56 * (t - 1.0))
    - 2.0 * theta * b_res55_56 * Delta * (1.0 - 2.0*d*C_res55_56*(d - 1.0))
    + d * (
      -A_res55_56 * b_res55_56 * 2.0 / beta_res55_56 * Delta * (d - 1.0)
      * d_quad ** (_exp1_55_56 - 1.0)
      - 2.0 * theta * b_res55_56 * (b_res55_56 -1.0) * dDelta
    )
  )
  coeffs[...,54:56] *= Delta ** np.where(Delta != 0, b_res55_56 - 2.0, 1.0)

  # Allocate exponent cache
  exponents = np.zeros_like(coeffs)
  # Compute exponents for 1-indices 8 to 51 as -d**c_i
  exponents[...,7:51] = -dc[7:51]
  # Compute exponents for 1-indices from 52 to 54
  exponents[...,51:54] = -alpha_res52_54 * (d - eps_res52_54) ** 2 \
    -beta_res52_54*(t - gamma_res52_54)**2
  # Compute exponents for 1-indices from 55 to 56
  exponents[...,54:56] = -C_res55_56 * d_quad \
    -D_res55_56*(t - 1.0)**2

  # Reduce
  return np.expand_dims(np.einsum("...i, ...i -> ...",
    coeffs, np.exp(exponents)), axis=-1)

def fused_phir_d_phir_dd(d:float, t:float) -> float:
  ''' Interface for consistent syntax with Cython backend. '''
  d = np.array([d])
  t = np.array([t])
  return phir_d(np.array([d])), phir_dd(np.array([d]), np.array([t]))