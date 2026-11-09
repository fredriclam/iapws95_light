'''
This file contains the middleware that calls the appropriate Cython function
for usual applications involving only water.
'''

import numpy as np
import iapws95_light.float_phi_functions as float_phi_functions
import iapws95_light.float_mix_functions as float_mix_functions

''' Load critical properties. '''
Tc = 647.096      # K
rhoc = 322.0      # kg / m^3
vc   = 1.0 / rhoc 
R = 0.46151805e3  # J / kg K
pc = 22.064e6     # MPa

def state_shallow_fallback(rho, T, dump=False):
  ''' Returns pure-phase water properties as a function of rho, T.
  Returns specific entropy (s), specific enthalpy (h), specific energy (u),
  wave speed (w), pressure (p).
  The flag `dump` triggers a dump of all internal calculation variables,
  replacing output pressure with a dict containing these internal variables.

  Shallow implementation using separate calls to backend
  '''
  d = rho / rhoc
  t = Tc / T
  phi0 = float_phi_functions.phi0(d, t)
  phi0_t = float_phi_functions.phi0_t(d, t)
  phi0_tt = float_phi_functions.phi0_tt(d, t)
  phir = float_phi_functions.phir(d, t)
  phir_d = float_phi_functions.phir_d(d, t)
  phir_t = float_phi_functions.phir_t(d, t)
  phir_dd = float_phi_functions.phir_dd(d, t)
  phir_dt = float_phi_functions.phir_dt(d, t)
  phir_tt = float_phi_functions.phir_tt(d, t)

  _g1 = 1 + d * phir_d - d * t * phir_dt
  reduced = 1 + 2 * d * phir_d + d * d * phir_dd - (
    _g1 * _g1 / (t * t * (phi0_tt + phir_tt))
  )
  w = np.sqrt(reduced * R * T)
  s = R * (t * (phi0_t + phir_t) - phi0 - phir)
  h = R * T * (1 + t * (phi0_t + phir_t) + d * phir_d)
  u = R * T * t * (phi0_t + phir_t)
  p = (1 + d * phir_d) * rho * R * T

  if dump:
    return s, h, u, w, {
      "phi0": phi0,
      "phi0_t": phi0_t,
      "phi0_tt": phi0_tt,
      "phir": phir,
      "phir_d": phir_d,
      "phir_t": phir_t,
      "phir_dd": phir_dd,
      "phir_dt": phir_dt,
      "phir_tt": phir_tt,}
  else:
    return float(s), float(h), float(u), float(w), float(p)


def fused_phi0_all(d, t):
  return float_phi_functions.fused_phi0_all(d, t)

def fused_phir_all(d, t):
  return float_phi_functions.fused_phi0_all(d, t)

def fused_phir_d3(d, t):
  return float_phi_functions.fused_phir_d3(d, t)

def prho_sat(T: float):
  ''' Saturation triplet (p, rho_satl, rho_satv).
  See also flaot_phi_functions.prho_sat_stepinfo and
  flaot_phi_functions.newton.
  '''
  return float_phi_functions.prho_sat(T)

def p(rho, T):
  return float_phi_functions.p(rho, T)

def u(rho, T):
  return float_phi_functions.u(rho, T)

def c_v(rho, T):
  return float_phi_functions.c_v(rho, T)

def c_p(rho, T):
  return float_phi_functions.c_p(rho, T)

def sound_speed(rho, T):
  return float_phi_functions.sound_speed(rho, T)

def rho_l_pt(p, T):
  return float_mix_functions.rho_l_pt(p, T)

def s(rho, T):
  return float_phi_functions.s(rho, T)

def h(rho, T):
  ''' Specific enthalpy (per mass) '''
  return float_phi_functions.h(rho, T)

def expansion_coeff(rho, T):
  ''' Thermal expansion coefficient '''
  print("Warning: expansion_coeff only calculated for pure phase.")
  return float_phi_functions.alpha(rho, T)

def dedrho_dedT(rho, T, return_px=True, return_as_dict=True):
  ''' Return gradient of e (de/drho, de/dT). If return_px, returns
   (de/drho, de/dT, p, x), where rho is the mixture density. Else, returns
   (de/drho, de/dT). If return_as_dict, the return is a dict instead of an
   array; p and x are included and return_px is ignored.
   
   Return value x == -1 if liquid and vapour do not coexist at the given
   (rho, T) conditions.
   '''

  out = float_mix_functions.grad_e_tv(T, 1.0/rho)
  dedrho = - out["dedv"] / (rho * rho)

  if return_as_dict:
    out["dedrho"] = dedrho
    return out
  else:
    if return_px:
      return np.array([dedrho, out["dedT"], out["p"], out["x"]])
    else:
      return np.array([dedrho, out["dedT"]])

def dedv_dedT(rho, T, return_px=True, return_as_dict=True):
  ''' Return gradient of e (de/dv, de/dT). If return_px, returns
   (de/dv, de/dT, p, x), where v is the mixture spec. volume. Else, returns
   (de/dv, de/dT). If return_as_dict, the return is a dict instead of an
   array; p and x are included and return_px is ignored.
   
   Return value x == -1 if liquid and vapour do not coexist at the given
   (rho, T) conditions.
   '''

  out = float_mix_functions.grad_e_tv(T, 1.0/rho)
  
  if return_as_dict:
    return out
  else:
    if return_px:
      return np.array([out["dedv"], out["dedT"], out["p"], out["x"]])
    else:
      return np.array([out["dedv"], out["dedT"]])\

def dpdrho_dpdT(rho, T):
  ''' Return gradient of p (dp/drho, dp/dT). '''

  out = float_mix_functions.grad_p_tv(T, 1.0/rho)
  return np.array([-out["second"]/(rho * rho), out["first"]]) # dpdrho, dpdT

def dpdv_dpdT(rho, T):
  ''' Return gradient of p (dp/dv, dp/dT). '''
  out = float_mix_functions.grad_p_tv(T, 1.0/rho)
  return np.array([out["second"], out["first"]]) # dpdv, dpdT

def isobar(p):
  ''' Compute isobar (rho, T) at pressure p '''
  pass

# Compute saturation curve (p, T, rhol, rhov)
_Tsat_curve_near_crit = Tc - np.geomspace(1e-6, 1, 100,endpoint=False)[::-1]
_Tsat_curve_away_crit = np.linspace(273.16, Tc - 1.0, 100)
Tsat_curve = np.append(_Tsat_curve_away_crit, _Tsat_curve_near_crit)

_sat_tups = [prho_sat(T) for T in Tsat_curve]
psat_curve = np.array([tup["psat"] for tup in _sat_tups])
rhosatl_curve = np.array([tup["rho_satl"] for tup in _sat_tups])
rhosatv_curve = np.array([tup["rho_satv"] for tup in _sat_tups])