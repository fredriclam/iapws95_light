''' Tools for computing mixture properties of IAPWS95 water with a linearized
equation of state p = p(rho). 
WLM equation of state
'''

import iapws95_light
import numpy as np

# Set backend for phi computation
backend = iapws95_light
try:
  import float_phi_functions
  backend = float_phi_functions
except ModuleNotFoundError:
  print("Cython module for scalar phi computations not found. Defaulting to " +
    "python implementation. " )

# Material parameters
K = 10e9
p_m0 = 5e6
rho_m0 = 2.7e3
v_m0 = 1.0 / rho_m0
# Set range of pressures
p_min = 1e2
p_max = 1e9
# Set max temperature
T_max = 1500
# Set numerical options
max_newton_iters = 12
newton_rtol = 1e-9

# Extract water constants to module scope
rhoc = iapws95_light.rhoc
vc = 1.0 / rhoc
Rw = iapws95_light.R

# Estimate limits on phasic densities
rhow_max = float(iapws95_light.rho_pt(p_max, 273))
rhom_max = (1 + (p_max - p_m0) / K) * rho_m0
rhow_min = p_min / (Rw * T_max)
rhom_min = (1 + (0.0 - p_m0) / K) * rho_m0
# Define feasible range
range_rho_mix_coords = np.linspace(0, 1, 64)
range_yw = np.linspace(0.01, 0.99, 96)
# Generate meshgrid of rho_mix, yw coordinates covering feasible range
mg_rho_mix, mg_yw = np.meshgrid(range_rho_mix_coords , range_yw)
for i, j in np.ndindex(mg_rho_mix.shape):
  yw = range_yw[i]
  ym = 1.0 - yw
  rho_mix_min = 1.0/(yw / rhow_min +  ym / rhom_min)
  mg_rho_mix[i,j] = rho_mix_min + range_rho_mix_coords[j] * (
    1/(yw / rhow_max +  ym / rhom_max) - rho_mix_min)

def _itersolve_d(rho_mix, yw, T, d_init=1.0) -> float:
  ''' Solve for reduced density d = rho/rho_c using Newton's method. '''
  # Compute dependent variables
  ym = 1.0 - yw
  v_mix = 1.0 / rho_mix
  t = iapws95_light.Tc / T
  # Compute coefficients
  a = rhoc / rho_mix
  b = - yw
  c2 = a
  c1 = (v_mix * (K - p_m0) - K * ym * v_m0)/(Rw * T) - yw
  c0 = -yw * vc / (Rw * T) * (K - p_m0)

  def scaled_pressure_difference(d:float) -> float:
    ''' Return value of scaled pressure difference '''
    return (a*d + b)*d**2*backend.phir_d(d, t) + ((c2 * d) + c1)*d + c0

  def residual_and_slope(d:float) -> float:
    ''' Return value of scaled pressure difference and slope '''
    out = backend.fused_phir_d_phir_dd(d, t)
    phir_d, phir_dd = out["first"], out["second"]
    val = (a*d + b)*d**2*phir_d + ((c2 * d) + c1)*d + c0
    slope = (3*a*d + 2*b)*d*phir_d \
      + (a*d + b)*d**2*phir_dd \
      + 2*c2*d + c1
    return val, slope

  # Initial guess
  d = d_init
  success = False

  # Attempt (1/2) to apply Newton's method
  for i in range(max_newton_iters):
    # Compute Newton step
    val, slope = residual_and_slope(d)
    step = -val/slope
    # Check rtol
    if (step / d) * (step / d) < newton_rtol*newton_rtol and d > 0.0:
      success = True
    # Take step
    d += step
    if success:
      break
  if success:
    return d, np.abs(step)
  
  # Attempt (2/2) to apply Newton's method
  d = 5.0
  for i in range(max_newton_iters):
    # Compute Newton step
    val, slope = residual_and_slope(d)
    step = val/slope
    # Check rtol
    if np.any(np.abs(step / d) < newton_rtol) and d > 0.0:
      success = True
    # Take step
    d -= step
    if success:
      break

  if not success:
    print("Warning: Newton's method did not converge. Returning anyway. " +
      f"d={d}, step={step}, rho_mix={rho_mix}, ym={ym}, yw={yw}, t={t}")
  return d, np.abs(step)

def solve_rhow(rho_mix:float, yw:float, T:float) -> float:
  ''' Solve for water density. '''
  d_init = 1.0
  if T < iapws95_light.Tc:
    # Compute saturation properties
    sat_info = backend.prho_sat(T)
    psat, rhol, rhov = \
      float(sat_info["psat"]), sat_info["rho_satl"], sat_info["rho_satv"]
    # Compute dependent variables
    ym = 1.0 - yw
    v_mix = 1.0 / rho_mix
    # Compute tentative density value
    rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0))
    if rhov <= rhow and rhow <= rhol:
      # Accept tentative value
      return rhow
    elif rhow < rhov:
      # Vapour phase
      d_init = rhov / rhoc
    else:
      # Liquid phase
      d_init = rhol / rhoc
  # Iteratively solve for water reduced volume d
  d, absstep = _itersolve_d(rho_mix, yw, T, d_init=d_init)
  return d * rhoc