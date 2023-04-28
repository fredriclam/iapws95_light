''' Tools for computing mixture properties of IAPWS95 water with a linearized
magma equation of state p = p(rho) and air. Provides middleware for Cython
implementation of underlying iterative scheme. Wraps in an object for object-
oriented codes like Quail.
'''

import iapws95_light
import numpy as np

try:
  import float_phi_functions
  import float_mix_functions
except ModuleNotFoundError as e:
  raise ModuleNotFoundError("Cython module for scalar phi computations not "
    + "found. Install Cython, update python ()>=3.9) and compile locally." ) from e

class WLM():
  ''' Mixture material parameters with mappings to pressure, temperature, and
  sound speed. Special case of WLMA assuming ya = 0.0. '''

  def __init__(self, K=10e9, p_m0=5e6, rho_m0=2.6e3, c_v_m0=3e3,
      R_a=287, gamma_a=1.4):
    # Material parameters
    self.K, self.p_m0, self.rho_m0, self.c_v_m0 = \
      K, p_m0, rho_m0, c_v_m0
    # Extract water constants to module scope
    self.rhoc = iapws95_light.rhoc
    self.vc = 1.0 / self.rhoc
    self.R = iapws95_light.R
    self.R_a = R_a
    self.gamma_a = gamma_a

  def pTvf_map(self, rho_vec:np.array, momentum:np.array, vol_energy:np.array):
    ''' Compute pressure and temperature from vector of conservative variables.
    Uses python map over a float function.
    Inputs:
      rho_vec[...,:] = [arhoA=0.0, arhoW, arhoM]
      momentum[...,:] = [rhou, (rhov)]
      vol_energy[...,:] = [rhoe]
      NDIMS = 1 or 2 (if 1, eliminates rhov)
    '''
    # Endmember assumption
    ya = 0.0
    # Compute mixture composition
    rho_mix = rho_vec.sum(axis=-1, keepdims=True)
    y = rho_vec / rho_mix
    # Replace yW > 1 - 1e-9 with yW = 1 - 1e-9
    y[...,1:2] = np.clip(y[...,1:2], 0.0, 1-1e-9)
    # Readjust other mass fractions to add up to one
    y[...,2:3] = 1.0 - y[...,1:2]
    yw = y[...,1:2]
    # Compute internal energy
    kinetic = 0.5 * (momentum * momentum).sum(axis=-1, keepdims=True) / rho_mix
    vol_energy_internal = vol_energy - kinetic
    
    # Allocate
    rhow        = np.empty((*rho_vec.shape[:-1], 1))
    p           = np.empty((*rho_vec.shape[:-1], 1))
    T           = np.empty((*rho_vec.shape[:-1], 1))
    sound_speed = np.empty((*rho_vec.shape[:-1], 1))
    # Call Cython iterative solve for rhow, p, T
    _out = list(map(
      lambda evol_mix, rho_mix, yw: float_mix_functions.conservative_to_pT_WLMA(
        evol_mix, rho_mix, yw, ya,
        self.K, self.p_m0, self.rho_m0, self.c_v_m0,
        self.R_a, self.gamma_a),
      vol_energy_internal.ravel(),
      rho_mix.ravel(),
      yw.ravel()))
    # Python-level data rearrangement
    rhow.ravel()[:], p.ravel()[:], T.ravel()[:] = \
      [d["rhow"] for d in _out], [d["p"] for d in _out], [d["T"] for d in _out]
    
    # Postprocess for volume fraction
    volfracW = rho_vec[...,1:2] / rhow
    # Postprocess for sound speed
    sound_speed.ravel()[:] = list(map(
      lambda rhow, T: float_phi_functions.sound_speed(rhow, T),
      rhow.ravel(), T.ravel()
    ))
    # TODO: propagate error

    return rhow, p, T, sound_speed, volfracW
  
  def pTvf_native(self, arho_vec:np.array, momentum:np.array, vol_energy:np.array):
    ''' Compute pressure and temperature from vector of conservative variables.
    Uses Cython native iteration.
    Inputs:
      arho_vec[...,:] == [arhoA=0.0, arhoW, arhoM]: partial densities
      momentum[...,:] == [rhou, (rhov)]: momentum components
      vol_energy[...,:] == [rhoe]: volumetric total energy
    '''
    # Compute mixture composition
    rho_mix = arho_vec.sum(axis=-1, keepdims=True)
    y = arho_vec / rho_mix
    # Replace yW > 1 - 1e-9 with yW = 1 - 1e-9
    y[...,1:2] = np.clip(y[...,1:2], 0.0, 1-1e-9)
    # Readjust other mass fractions to add up to one
    y[...,2:3] = 1.0 - y[...,1:2]
    yw = y[...,1:2]
    ya = np.zeros_like(yw)
    # Compute internal energy
    kinetic = 0.5 * (momentum * momentum).sum(axis=-1, keepdims=True) / rho_mix
    vol_energy_internal = vol_energy - kinetic
    
    # Call Cython iterative solve for rhow, p, T
    _out = float_mix_functions.vec_conservative_to_pT_WLMA(
        vol_energy_internal, rho_mix, yw, ya,
        self.K, self.p_m0, self.rho_m0, self.c_v_m0,
        self.R_a, self.gamma_a)
    # Python-level data rearrangement
    rhow        = np.reshape(_out[0::4], yw.shape)
    p           = np.reshape(_out[1::4], yw.shape)
    T           = np.reshape(_out[2::4], yw.shape)
    sound_speed = np.reshape(_out[3::4], yw.shape)
    # Postprocess for volume fraction
    volfracW = arho_vec[...,1:2] / rhow

    # TODO: propagate error

    return rhow, p, T, sound_speed, volfracW