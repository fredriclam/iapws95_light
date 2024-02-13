''' Tools for computing mixture properties of IAPWS95 water with a linearized
magma equation of state p = p(rho), water and air. Provides an interface for
Cython implementation of underlying iterative scheme. The class interface is
useful for object-oriented codes like Quail.

The middleware provided here adapts the data format, vector-level parallel
splitting and joining, and caching for repeated calls with the same inputs.

The data format matches that of Quail (np.array with shape [...,nstates]) where
nstates is the number of states in the state vector.

Vector-level parallel computation is done through multiprocessing.pool if a
pool is passed (otherwise, the computation is done serially). The vector is
chunked into approximately equal parts that are processed in parallel.

Caching prevents redundant computation when the same set of vector inputs is
passed consecutively.The inputs and outputs are cached in a circular buffer
where data are replaced when inputs of the same size but with different values
are provided, or when the maximum number of cached inputs/outputs is reached.
'''

import numpy as np
import multiprocessing as mp
import datetime
from typing import Tuple

try:
  # Import locally Cython compiled modules
  from . import float_phi_functions
  from . import float_mix_functions
except ImportError:
  import float_phi_functions
  import float_mix_functions
except ModuleNotFoundError as e:
  raise ModuleNotFoundError("Cython module for scalar phi computations not "
    + "found. Install Cython, update python ()>=3.9) and compile locally." ) from e

''' Set up static parameters '''
Tc = 647.096               # K
rhoc = 322.                # kg / m^3
R = 0.461_518_05e3         # J / kg K
Ttriple = 273.16           # K (definition ITS-90)
rhol_triple = 999.793      # kg/m^3
rhov_triple = 0.004_854_58 # kg/m^3
ptriple = 611.654_771      # Pa (2018 revised release)
pc = 22.064e6              # Pa

class SimpleLogger():
  ''' Simple bundled logger that stores logs as a list of lists. '''
  def __init__(self):
    self.curr_log = []
    self.all_logs = [self.curr_log]
  def log(self, level, data):
    self.curr_log.append((level, data))
  def create_new_log(self):
    self.curr_log = []
    self.all_logs.append(self.curr_log)
  def pop_log(self):
    return self.all_logs.pop()

class WLMA():
  ''' Mixture material parameters with mappings to pressure, temperature, and
  sound speed '''

  class WLMACache():
    def __init__(self, buffer_count=8, buffer_thresh=50):
      ''' Cache for input->output association. A circular buffer is used. Arrays
      of a size existing in the buffer, but with different values, replace the
      existing array. Arrays that are too small (less than buffer_thresh tuples
      of inputs) are ignored.
      To retrieve, access the raw buffers as needed using index_in_cache if
      index_in_cache is not None.

      Inputs:
        buffer_count: number of arrays to buffer
        buffer_thresh: buffer only if poll size >= this
      '''
      self.buffer_count = buffer_count
      self.buffer_thresh = buffer_thresh
      self.buffer_index = 0
      self.num_buffers_filled = 0
      # Input buffers
      self.buffer_arho_vec = [None] * self.buffer_count
      self.buffer_momentum = [None] * self.buffer_count
      self.buffer_vol_energy = [None] * self.buffer_count
      # Output buffers
      self.buffer_rhow = [None] * self.buffer_count
      self.buffer_p = [None] * self.buffer_count
      self.buffer_T = [None] * self.buffer_count
      self.buffer_sound_speed = [None] * self.buffer_count
      self.buffer_volfracW = [None] * self.buffer_count
      # Diagnostic stats
      self.cache_hits = 0
      self.cache_misses = 0
      self.cache_ignores = 0

    def _is_cacheable(self, arho_vec:np.array, momentum:np.array,
                    vol_energy:np.array) -> bool:
      ''' Returns whether something is large enough to cache. '''
      return vol_energy.size >= self.buffer_thresh
    
    def index_in_cache(self, arho_vec:np.array, momentum:np.array,
                    vol_energy:np.array):
      ''' Returns index in cache if in cache, else returns None. '''
      # Filter out inputs that are too small
      if not self._is_cacheable(arho_vec, momentum, vol_energy):
        self.cache_ignores += 1
        return None
      # Find a match in the circular buffer
      for i in range(self.num_buffers_filled):
        if arho_vec.shape == self.buffer_arho_vec[i].shape \
            and momentum.shape == self.buffer_momentum[i].shape \
            and vol_energy.shape == self.buffer_vol_energy[i].shape:
          if np.all(arho_vec == self.buffer_arho_vec[i]) \
              and np.all(momentum == self.buffer_momentum[i]) \
              and np.all(vol_energy == self.buffer_vol_energy[i]):
            self.cache_hits += 1
            return i
      self.cache_misses += 1
      return None

    def write_to_buffer(self, arho_vec:np.array, momentum:np.array,
                        vol_energy:np.array, rhow:np.array, p:np.array,
                        T:np.array, sound_speed:np.array, volfracW:np.array):
      ''' Add to buffer, replacing any entry with the same shape. '''
      # Ignore values if data is too small
      if not self._is_cacheable(arho_vec, momentum, vol_energy):
        return
      # Set default write location
      write_index = self.buffer_index
      create_new_entry = True
      # Set write index to any existing shape
      for i in range(self.num_buffers_filled):
        if arho_vec.shape == self.buffer_arho_vec[i].shape \
            and momentum.shape == self.buffer_momentum[i].shape \
            and vol_energy.shape == self.buffer_vol_energy[i].shape:
          write_index = i
          create_new_entry = False
          break
      if create_new_entry:
        # Track number of filled buffers
        if self.num_buffers_filled < self.buffer_count:
          self.num_buffers_filled += 1
        # Move write pointer
        self.buffer_index = (self.buffer_index + 1) % self.buffer_count
      # Fill input buffers
      self.buffer_arho_vec[write_index] = arho_vec
      self.buffer_momentum[write_index] = momentum
      self.buffer_vol_energy[write_index] = vol_energy
      # Fill output buffers
      self.buffer_rhow[write_index] = rhow
      self.buffer_p[write_index] = p
      self.buffer_T[write_index] = T
      self.buffer_sound_speed[write_index] = sound_speed
      self.buffer_volfracW[write_index] = volfracW

    def get_buffer_bytes(self):
      ''' Estimate bytes occupied by buffer. '''
      nbytes_total = 0
      for i in range(self.num_buffers_filled):
        nbytes_total += self.buffer_arho_vec[i].nbytes \
                      + self.buffer_momentum[i].nbytes \
                      + self.buffer_vol_energy[i].nbytes * (1 + 4)
      return nbytes_total

  def __init__(self, K=10e9, p_m0=5e6, rho_m0=2.6e3, c_v_m0=3e3,
      R_a=287, gamma_a=1.4):
    # Material parameters
    self.K, self.p_m0, self.rho_m0, self.c_v_m0 = \
      K, p_m0, rho_m0, c_v_m0
    # Extract water constants to module scope
    self.rhoc = rhoc
    self.vc = 1.0 / self.rhoc
    self.R = R
    self.R_a = R_a
    self.gamma_a = gamma_a
    self.cache = WLMA.WLMACache()
    self.timings = {} # TODO: size -> (hits, totaltime)

    # Supervisor (index, [inputs]) -> (residual errors) pairs
    self.has_supervisor = True
    self.supervisor_buffer = np.empty((0,8))
    self.supervisor_index = 0
    _stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    self.supervisor_filename_prefix = f"slogs_{_stamp}_"
    self.supervisor_maxbytes = 2**28 # (~256 MiB tables)
  
  def __call__(self, arho_vec:np.array, momentum:np.array, vol_energy:np.array,
               pool:mp.Pool=None) -> Tuple[np.array, np.array, np.array,
                                           np.array, np.array]:
    ''' Entry point for computing (rhow, p, T) for the WLMA model. Calls the
    appropriate backend and returns (rhow, p, T, vf, soundspeed). '''

    # Check cache
    i = self.cache.index_in_cache(arho_vec, momentum, vol_energy)
    if i is not None:
      # Read from cache
      out = (self.cache.buffer_rhow[i],
             self.cache.buffer_p[i],
             self.cache.buffer_T[i],
             self.cache.buffer_sound_speed[i],
             self.cache.buffer_volfracW[i])
    else:
      # Call native conservative-to-primitive routine
      out = self.WLM_rhopT_native(arho_vec, momentum, vol_energy, pool)
      # Write to cache
      self.cache.write_to_buffer(arho_vec, momentum, vol_energy, *out)

    # TODO: feature: logger passing (this would take a lot of memory because of
    # the verbosity of the iterative solver logging)
    self.has_supervisor = False
    if self.has_supervisor:
      self.supervisor_pack_entry((arho_vec, momentum, vol_energy), out)

    return out
  
  def supervisor_pack_entry(self, input, output):
    ''' Supervisor encoding format.
    arg input is the tuple (arho_vec, momentum, vol_energy)
    arg output is the tuple (rho, p, T, c, alpha). '''
    arho_vec, momentum, vol_energy = input
    rhow, p, T, c, alphaw = output
    
    def _scalar_objfn(alpha_w, T, vol_energy_internal, rho_mix, yw, ya):
      params = {
        "vol_energy": vol_energy_internal,
        "rho_mix": rho_mix,
        "yw": yw,
        "ya": ya,
        "K": self.K,
        "p_m0": self.p_m0,
        "rho_m0": self.rho_m0,
        "c_v_m0": self.c_v_m0,
        "R_a": self.R_a,
        "gamma_a": self.gamma_a,
      }
      # As f0, f1, p_mix
      return float_mix_functions.poll_kernel(alpha_w, T, params)[0:2]
    
    # Vector arithmetic
    vol_energy_internal = vol_energy \
          - 0.5 * (momentum * momentum).sum(axis=-1, keepdims=True) / rho_mix,
    rho_mix = arho_vec.sum(axis=-1, keepdims=True)
    ya = arho_vec[...,0:1] / rho_mix
    yw = arho_vec[...,1:2] / rho_mix
    
    f0 = np.zeros_like(p)
    f1 = np.zeros_like(p)

    # Map available variables to gradients of (v, e) wr.t. (T, p)
    f0.ravel()[:], f1.ravel()[:] = \
      np.array(list(zip(*map(
        lambda alpha_w, T, vol_energy_internal, rho_mix, yw, ya:
          _scalar_objfn(alpha_w, T, vol_energy_internal, rho_mix, yw, ya),
        alphaw.ravel(), T.ravel(), vol_energy_internal.ravel(),
        rho_mix.ravel(), yw.ravel(), ya.ravel()))))

    # Assemble data table
    _table = np.stack((arho_vec[...,0].ravel(),
              arho_vec[...,1].ravel(),
              arho_vec[...,2].ravel(),
              np.sqrt(momentum*momentum).sum(axis=-1).ravel(),
              vol_energy.ravel(),
              f0.ravel(),
              f1.ravel(),
              p.ravel()
              ), axis=1)
    self.supervisor_buffer = np.append(self.supervisor_buffer, _table, axis=0)
    if self.supervisor_buffer.nbytes > self.supervisor_maxbytes:
      # Flush buffer to disk
      _outfile_name = self.supervisor_filename_prefix + f"{self.supervisor_index}"
      np.save(_outfile_name, self.supervisor_buffer)
      # Remake supervisor buffer
      self.supervisor_buffer = np.empty((0, self.supervisor_table.shape[1]))
      self.supervisor_index += 1

  def pressure_sgradient(self, vol_energy, rho_mix, yw, ya, rhow, p, T, u, v):
    ''' Compute gradient of pressure with respect to conservative variables. '''

    # weighted_vT, weighted_vp, weighted_eT, weighted_ep = \
    #   float_mix_functions.cons_derivatives_pT(rhow, p, T, params)
    v_T, v_p, e_T, e_p, e_w, e_m_mech = np.zeros_like(p), np.zeros_like(p), \
      np.zeros_like(p), np.zeros_like(p), np.zeros_like(p), np.zeros_like(p)

    def _cons_derivatives_pT(rhow, p, T, vol_energy, rho_mix, yw, ya):
      params = {
        "vol_energy": vol_energy,
        "rho_mix": rho_mix,
        "yw": yw,
        "ya": ya,
        "K": self.K,
        "p_m0": self.p_m0,
        "rho_m0": self.rho_m0,
        "c_v_m0": self.c_v_m0,
        "R_a": self.R_a,
        "gamma_a": self.gamma_a,
      }
      return float_mix_functions.cons_derivatives_pT(rhow, p, T, params)

    # Map available variables to gradients of (v, e) wr.t. (T, p)
    v_T.ravel()[:], v_p.ravel()[:], e_T.ravel()[:], e_p.ravel()[:] = \
      np.array(list(zip(*map(
        lambda rhow, p, T, vol_energy, rho_mix, yw, ya:
          _cons_derivatives_pT(rhow, p, T, vol_energy, rho_mix, yw, ya),
        rhow.ravel(), p.ravel(), T.ravel(), vol_energy.ravel(),
        rho_mix.ravel(), yw.ravel(), ya.ravel()))))

    # Compute energies
    # e_m_mech = np.array([float_mix_functions.magma_mech_energy(_p,
      # self.K, self.p_m0, self.rho_m0) for _p in p.ravel()])
    e_m_mech.ravel()[:] = np.array(list(map(
      lambda _p: float_mix_functions.magma_mech_energy(_p, self.K, self.p_m0, self.rho_m0),
      p.ravel())))
    # print(rhow.ravel().shape)
    # print(T.ravel().shape)
    # print(list(zip(*map(float_mix_functions.u, rhow.ravel(), T.ravel()))))
    e_w.ravel()[:] = np.array(list(map(float_mix_functions.u, rhow.ravel(), T.ravel())))
    
    # Compute determinant for transformation in state space
    det = v_T * e_p - v_p * e_T
    # Compute kinetic energy per mass
    kinetic = (u*u + v*v)/2
    e_m = self.c_v_m0 * T + e_m_mech
    # print(v_T.shape, v_p.shape, e_T.shape, e_p.shape, e_m_mech.shape,
          # e_w.shape, kinetic.shape, e_m.shape, u.shape, v.shape)
    # assert(T.shape == e_m_mech.shape)

    # Allocate native gradients (assume p has shape (ne,nq,...) )
    f = np.zeros((*p.shape[0:2], 9))
    f[...,0:1] = kinetic - self.R_a / (self.gamma_a - 1.0) * T
    f[...,1:2] = kinetic - e_w
    f[...,2:3] = kinetic - e_m
    f[...,3:4] = -u
    f[...,4:5] = -v
    f[...,5:6] = 1.0
    rho_m = self.rho_m0 * (1.0 + (p - self.p_m0) / self.K)
    g = np.zeros((*p.shape[0:2], 9))
    g[...,0:1] = -self.R_a * T / p
    g[...,1:2] = -1.0/rhow
    g[...,2:3] = -1.0/rho_m
    # Compute pressure gradient
    return 1/(rho_mix * det) * (v_T * f - e_T * g)

  def WLM_rhopT_map(self, rho_vec:np.array,
                    momentum:np.array, vol_energy:np.array):
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
    pass

    return rhow, p, T, sound_speed, volfracW
  
  def WLM_rhopT_native(self, arho_vec:np.array,
                       momentum:np.array, vol_energy:np.array,
                       pool:mp.Pool=None) -> Tuple[np.array, np.array, np.array,
                                                   np.array, np.array]:
    ''' Computes WLMA equation of state primitives (rhow, p, T, c, volfracW).
    Inputs:
      arho_vec with shape (...,3)
      momentum with shape (...,[1,2,...])
      vol_energy with shape (...,1)
      [pool, optional]; if truthy, is used as an mp.pool
    '''
    if pool and vol_energy.size >= self.cache.buffer_thresh: # TODO: NEW
      # _t1=perf_counter()
      par_result = pool.starmap(self.WLM_rhopT_native_serial,
        zip(np.array_split(arho_vec, pool._processes, axis=0),
          np.array_split(momentum, pool._processes, axis=0),
          np.array_split(vol_energy, pool._processes, axis=0)))
      _out_temp =  [np.concatenate(par_output, axis=0)
                    for par_output in (zip(*par_result))]
      # _t2=perf_counter()
      # self.WLM_rhopT_native_serial(arho_vec, momentum, vol_energy)
      # _t3=perf_counter()
      # print(f"Wallratio:{(_t2-_t1)/(_t3-_t2):.3f}; load:{arho_vec.shape[0]}")
      # print(traceback.print_stack(file=sys.stdout))
      return _out_temp
    else:
      return self.WLM_rhopT_native_serial(arho_vec, momentum, vol_energy)

  def WLM_rhopT_native_serial(self, arho_vec:np.array,
                       momentum:np.array, vol_energy:np.array, logger=False):
    ''' Compute pressure and temperature from vector of conservative variables.
    Uses Cython native iteration.
    Inputs:
      arho_vec[...,:] == [arhoA, arhoW, arhoM]: partial densities
      momentum[...,:] == [rhou, (rhov)]: momentum components
      vol_energy[...,:] == [rhoe]: volumetric total energy
      [logger,]: logger object with method log(self, level:str, data:dict)
    '''
    # Compute mixture composition
    arho_vec = arho_vec.copy()
    rho_mix = arho_vec.sum(axis=-1, keepdims=True)
    
    ''' Pseudo-solubility approximation
    Due to the convex shape of the curve of mixture sound speed as a function of
    air mass fraction, numerical perturbation to the solution on the order of
    the squared (acoustic) impedance constrast can lead to a large change in the
    sound speed. This may cause numerical instability. Here we choose to
    essentially omit the air phase when below the equilibrium solubility of air
    in water. '''

    # Zero-water shortcircuit
    if np.all(arho_vec[...,1:2] == 0):
      # Preprocess energy, mass fraction
      kinetic = 0.5 * (momentum * momentum).sum(axis=-1, keepdims=True) / rho_mix
      vol_energy_internal = vol_energy - kinetic
      ya = arho_vec[...,0:1] / rho_mix

      def scalar_p_LMA(vol_energy_internal, rho_mix, ya):
        ''' Scalar (vars...) -> p'''
        ym = 1.0 - ya
        # Magma mech energy neglected
        T = (vol_energy_internal / rho_mix
           - float_mix_functions.magma_mech_energy(
               self.p_m0, self.K, self.p_m0, self.rho_m0)) / (
            ya * self.R_a / (self.gamma_a - 1.0)
            + ym * self.c_v_m0)
        params = {
          "vol_energy": vol_energy_internal,
          "rho_mix": rho_mix,
          "yw": 0.0,
          "ya": ya,
          "K": self.K,
          "p_m0": self.p_m0,
          "rho_m0": self.rho_m0,
          "c_v_m0": self.c_v_m0,
          "R_a": self.R_a,
          "gamma_a": self.gamma_a,
        }
        p = float_mix_functions.p_LMA(T, ya, params)
        # Correct for mech energy
        T = (vol_energy_internal / rho_mix
           - float_mix_functions.magma_mech_energy(
               p, self.K, self.p_m0, self.rho_m0)) / (
            ya * self.R_a / (self.gamma_a - 1.0)
            + ym * self.c_v_m0)
        
        return float_mix_functions.p_LMA(T, ya, params), T
    
      # Map
      rhow = 1e3 * np.ones_like(vol_energy_internal)
      p = np.zeros_like(vol_energy_internal)
      T = np.zeros_like(vol_energy_internal)
      sound_speed = np.sqrt(self.K / self.rho_m0) \
        * np.ones_like(vol_energy_internal)
      volfracW = np.zeros_like(vol_energy_internal)
      p.ravel()[:], T.ravel()[:] = \
        np.array(list(zip(*map(
          lambda vol_energy_internal, rho_mix, ya:
            scalar_p_LMA(vol_energy_internal, rho_mix, ya),
          vol_energy_internal.ravel(), rho_mix.ravel(), ya.ravel()))))

      return rhow, p, T, sound_speed, volfracW

    # Compute target mass fraction
    y = arho_vec / rho_mix
    # Clip water fraction to leave 1 ppm of air, magma
    # TODO: sync with POS_TOL in PPL
    # TODO: transfer mass fractions can lead to >1060kg/m^3 water phasic, beware
    y[...,1:2] = np.clip(y[...,1:2], 1e-6, 1-2e-6)
    # Clip air mass fraction
    y[...,0:1] = np.clip(y[...,0:1], 1e-6, None)
    # Readjust other mass fractions to add up to one
    y /= y.sum(axis=-1, keepdims=True)

    # Reconstruct mixture density # TODO: CHECK:
    arho_vec = y * rho_mix
    # rho_mix = arho_vec.sum(axis=-1, keepdims=True)

    # Extract mass fractions
    yw = y[...,1:2]
    ya = y[...,0:1]
    # Compute internal energy
    kinetic = 0.5 * (momentum * momentum).sum(axis=-1, keepdims=True) / rho_mix
    vol_energy_internal = vol_energy - kinetic

    # Critical monitor
    class FilterLog():
      def __init__(self):
        self.buffer = []
        self.critical = False
      def log(self, level:str, data:dict):
        if level == "critical":
          self.critical = True
        self.buffer.append(data)
        # Cicular buffer
        if len(self.buffer) > 10000:
          self.buffer = self.buffer[-5000:]
    logger = FilterLog()

    # Call Cython iterative solve for rhow, p, T
    _out = float_mix_functions.vec_conservative_to_pT_WLMA(
        vol_energy_internal, rho_mix, yw, ya,
        self.K, self.p_m0, self.rho_m0, self.c_v_m0,
        self.R_a, self.gamma_a, logger=logger)
    # Python-level data rearrangement
    rhow        = np.reshape(_out[0::4], yw.shape)
    p           = np.reshape(_out[1::4], yw.shape)
    T           = np.reshape(_out[2::4], yw.shape)
    sound_speed = np.reshape(_out[3::4], yw.shape)
    # Postprocess for volume fraction (no boundedness guarantee if rhow == 0)
    volfracW = arho_vec[...,1:2] / (1e-16 + rhow)

    # Pressure clipping for long-term stability against iteration fail
    p_clipped = np.clip(p, 1e3, 100e6)

    ''' Debug section '''

    # Propagate errors
    # if logger.critical > 0:
    #   print("Critical log")

    # Check kernel outputs
    # kernel_outputs = [float_mix_functions.kernel2_WLMA_debug(_volfracW, _T, _vol_energy_internal,
    #   _rho_mix, _yw, _ya, 
    #   self.K, self.p_m0, self.rho_m0, self.c_v_m0, self.R_a, self.gamma_a)
    #   for (_volfracW, _T, _vol_energy_internal, _rho_mix, _yw, _ya)
    #   in zip(volfracW.ravel(), T.ravel(), vol_energy_internal.ravel(), rho_mix.ravel(), yw.ravel(), ya.ravel())]
    # # Unpack outputs: residual vector, residual jac, hypothetical state, p, region int flag
    # _f, _J, _lv_hypothetical, _pmix, _region_type = zip(*kernel_outputs)
    # # Compute residual norms
    # fnorms = np.linalg.norm(_f, axis=-1)
    # # Check for large residuals and large water mass fractions
    # if np.any((fnorms > 1e-3) & (yw.ravel() >= 0.2)):
    #   _dummy_var = True
    # Check for clip activation
    if not np.all(p_clipped == p):
      _dummy_var = True

    return rhow, p_clipped, T, sound_speed, volfracW