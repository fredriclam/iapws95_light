import numpy as np
import scipy.optimize
import scipy.interpolate
import IAPWS_ice
import iapws95_light

from types import SimpleNamespace
from numpy.polynomial import Polynomial
from IAPWS_ice import T_n
from IAPWS_ice import p_n

class Boundary():
  def __init__(self, rho, u, rho_L, rho_R):
    self.rho = rho
    self.u = u
    self.rho_L = rho_L
    self.rho_R = rho_R

    if np.all(np.diff(self.rho) < 0):
      self.monotone = True
      # Reverse
      self.rho = self.rho[::-1]
      self.u = self.u[::-1]
    elif np.all(np.diff(self.rho) > 0):
      self.monotone = True
    else:
      ValueError("Boundary rho is not monotone. Possibly u(rho) is multi-valued.")

    # Construct interpolator
    self.interpolator = scipy.interpolate.PchipInterpolator(
      self.rho, self.u, extrapolate=True)
  
  def u_interp(self, rho):
    ''' Interpolator call wrapper. '''
    return self.interpolator(rho)

  def under(self, rho, u):
    ''' Returns whether boundary curve is under specified point.
    Returns False if point is outside the domain of the curve (horizontal
    coordinate). '''

    
    if type(rho) is float or type(rho) is int:
      in_curve_domain = (rho >= self.rho_L) and (rho <= self.rho_R)
      return bool(in_curve_domain and u >= self.u_interp(rho))
    else:
      in_curve_domain = (rho >= self.rho_L) & (rho <= self.rho_R)
      return np.where(in_curve_domain, u >= self.u_interp(rho), False)

  def over(self, rho, u):
    ''' Returns whether boundary curve is over specified point.
    Returns False if point is outside the domain of the curve (horizontal
    coordinate). '''

    if type(rho) is float or type(rho) is int:
      in_curve_domain = (rho >= self.rho_L) and (rho <= self.rho_R)
      return bool(in_curve_domain and u < self.u_interp(rho))
    else:
      in_curve_domain = (rho >= self.rho_L) & (rho <= self.rho_R)
      return np.where(in_curve_domain, u < self.u_interp(rho), False)

def all_pure_lv(rho, T, dump=False):
  ''' Pure-phase sound speed for liquid or vapor. '''
  d = rho / iapws95_light.rhoc
  t = iapws95_light.Tc / T
  phi0 = iapws95_light.phi0(d, t)
  phi0_t = iapws95_light.phi0_t(d, t)
  phi0_tt = iapws95_light.phi0_tt(d, t)
  phir = iapws95_light.phir(d, t)
  phir_d = iapws95_light.phir_d(d, t)
  phir_t = iapws95_light.phir_t(d, t)
  phir_dd = iapws95_light.phir_dd(d, t)
  phir_dt = iapws95_light.phir_dt(d, t)
  phir_tt = iapws95_light.phir_tt(d, t)

  _g1 = 1 + d * phir_d - d * t * phir_dt
  reduced = 1 + 2 * d * phir_d + d * d * phir_dd - (
    _g1 * _g1 / (t * t * (phi0_tt + phir_tt))
  )
  w = np.sqrt(reduced * iapws95_light.R * T)
  s = iapws95_light.R * (t * (phi0_t + phir_t) - phi0 - phir)
  h = iapws95_light.R * T * (1 + t * (phi0_t + phir_t) + d * phir_d)
  u = iapws95_light.R * T * t * (phi0_t + phir_t)
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
    return float(s), float(h), float(u), float(w)

# Compute triple-point solid state
s_s_n = IAPWS_ice.s_iceI(p_n, T_n)
u_s_n = IAPWS_ice.u_iceI(p_n, T_n)
rho_s_n = IAPWS_ice.rho_iceI(p_n, T_n)
v_s_n = 1 / rho_s_n
# Compute triple-point liquid and vapor states
triple_point_sat = iapws95_light.prho_sat(T_n)
v_l_n = 1 / triple_point_sat["rho_satl"]
v_v_n = 1 / triple_point_sat["rho_satv"]
s_l_n, h_l_n, u_l_n, w_l_n = all_pure_lv(1/v_l_n, T_n)
s_v_n, h_v_n, u_v_n, w_v_n = all_pure_lv(1/v_v_n, T_n)

def sl_fractions(u,v):
  ''' Compute [y_s, y_l] for given u, v of mixture at triple point. '''
  _A = np.array([[u_s_n - u_v_n, u_l_n - u_v_n],
                [v_s_n - v_v_n, v_l_n - v_v_n]])
  _b = np.array([u - u_v_n, v - v_v_n])
  # Explicit inverse
  _det = (_A[0,0] * _A[1,1] - _A[0,1] * _A[1,0])
  _Ainvb = np.array([_A[1,1] * _b[0] - _A[0,1] * _b[1],
                  -_A[1,0] * _b[0] + _A[0,0] * _b[1]]) / _det
  return _Ainvb

def sl_uv(y_s, y_l):
  ''' Compute [u, v] for given [y_s, y_l] of mixture at triple point. '''
  _Ay = np.array([y_s * u_s_n + y_l * u_l_n + (1 - y_s - y_l) * u_v_n,
                  y_s * v_s_n + y_l * v_l_n + (1 - y_s - y_l) * v_v_n])
  return _Ay

def SV_rhov(p_target, T, num_newton_iter=4):
  ''' Return density of vapor at solid-vapor saturation curve. '''
  # Initial guess: ideal gas approx
  rho_v = p_target / (iapws95_light.R * T)
  for i in range(num_newton_iter):
    # Compute potential and derivatives at guessed rho_v and temperature T
    s_v, h_v, u_v, w_v, _potential = all_pure_lv(rho_v, T, dump=True)
    d = rho_v / iapws95_light.rhoc
    # Compute pressure
    p = (1 + d * _potential["phir_d"]) * rho_v * iapws95_light.R * T
    dpdd = iapws95_light.rhoc * iapws95_light.R * T * (
      1 + d * d * _potential["phir_dd"] + 2 * d * _potential["phir_d"]
    )
    rho_v += iapws95_light.rhoc * (p_target - p) / dpdd
  return float(rho_v)

def SL_rhol(p_target, T, num_newton_iter=4, verbose=False, init=None):
  ''' Return density of liquid at solid-liquid saturation curve. '''
  # Static initial guess curve
  init_curve = Polynomial(np.array([1058.05940709,  -43.85534939, 
                                    -10.44251961,   -3.67634714]),
                          domain=[250, 273.16])
 
  if init is None:
    # Initial guess: normal liquid density
    rho_l = init_curve(T)
  else:
    rho_l = init
  for i in range(num_newton_iter):
    # Compute potential and derivatives at guessed rho_l and temperature T
    _, _, _, _, _potential = all_pure_lv(rho_l, T, dump=True)
    d = rho_l / iapws95_light.rhoc
    # Compute pressure
    p = (1 + d * _potential["phir_d"]) * rho_l * iapws95_light.R * T
    dpdd = iapws95_light.rhoc * iapws95_light.R * T * (
      1 + d * d * _potential["phir_dd"] + 2 * d * _potential["phir_d"]
    )
    rho_l += iapws95_light.rhoc * (p_target - p) / dpdd
    if verbose:
      print(rho_l,p)
  return float(rho_l)

class Model():
  def __init__(self, debug=False):
    self.T_MIN = 150
    self.T_MAX = 500
    self.T_LS_min = 250
    # Choice of minimum temperature (K) valid for calculating ice I melt curve
    self.T_I_min = 251.165  
    self.P_MIN = IAPWS_ice.ice_I_sublimation_p(self.T_MIN)
    # Compute pressure at T_I_min
    self.p_at_T_I_min = IAPWS_ice.ice_I_melting_p(self.T_I_min)
    # Liquid density inversion temperature: between T_n and T_L_inversion,
    # the density increases with temperature; thereafter density decreases.
    # Static value is approximated using numeric optimization.
    self.T_L_inversion = 277.15015815

    # Set number of iterations for regridding LV region
    self.NUM_ITERATIONS = 4

    if debug:
      return
    
    print("Populating LSV")
    self.LSV = self.populate_LSV()
    print("Populating LV")
    self.LV = self.populate_LV()
    print("Populating SV")
    self.SV = self.populate_SV()
    print("Populating LS")
    self.LS = self.populate_LS()
    print("Populating L")
    self.L = self.populate_L()
    print("Populating V")
    self.V = self.populate_V()

    print("Populating boundaries")
    self.boundaries = SimpleNamespace(
      L_LV_normal=self.boundary_L_LV_normal(),
      L_LV_anomal=self.boundary_L_LV_anomal(),
      L_TMAX=self.boundary_L_TMAX(),
      L_PMAX=self.boundary_L_PMAX(),
      V_LV=self.boundary_V_LV(),
      LV_TMAX=self.boundary_LV_TMAX(),
      LV_LSV=self.boundary_LV_LSV(),
      V_SV=self.boundary_V_SV(),
      SV_TMIN=self.boundary_SV_TMIN(),
      S_SV=self.boundary_S_SV(),
      L_LS=self.boundary_L_LS(),
      S_LS=self.boundary_S_LS(),
      LS_T_I_MIN=self.boundary_LS_T_I_MIN(),
      LS_LSV=self.boundary_LS_LSV(),
      SV_LSV=self.boundary_SV_LSV(),
    )

    print("Defining regions")

    # Define a constant-type boundary that returns true when no_boundary.over
    # or no_boundary.under(rho, u) is called.
    no_boundary = Boundary(np.array([0, 1]), np.array([0, 1]), -1, 1)
    def const_true(rho, u):
      if type(rho) is float or type(rho) is int:
        return True
      else:
          return np.full(rho.shape, True)
    no_boundary.over = const_true
    no_boundary.under = const_true

    # Table: Under & Over then Region
    self.region_definitions = [
      # Upper and lower conditions
      (self.boundaries.LS_T_I_MIN, self.boundaries.L_LS, "LS"),
      (self.boundaries.LS_T_I_MIN, self.boundaries.LS_LSV, "LS"),
      (self.boundaries.S_LS, self.boundaries.LS_LSV, "LS"),
      (self.boundaries.SV_TMIN, self.boundaries.S_SV, "SV"),
      (self.boundaries.SV_TMIN, self.boundaries.SV_LSV, "SV"),
      (self.boundaries.SV_TMIN, self.boundaries.V_SV, "SV"),
      (self.boundaries.LV_LSV, self.boundaries.L_LV_normal, "LV"),
      (self.boundaries.L_LV_anomal, self.boundaries.L_LV_normal, "LV"),
      (self.boundaries.LV_LSV, self.boundaries.LV_TMAX, "LV"),
      (self.boundaries.LV_LSV, self.boundaries.V_LV, "LV"),
      (self.boundaries.LS_LSV, self.boundaries.LV_LSV, "LSV"),
      (self.boundaries.SV_LSV, self.boundaries.LV_LSV, "LSV"),
      # L region is overlapping with LV region a bit
      # Order matters: L here assumes no membership in LV
      # Strictly, the L region should divide L_LS into rho < rho_max(L_LV_anomal)
      # and rho > rho_max(L_LV_anomal)
      (self.boundaries.L_LV_normal, self.boundaries.L_TMAX, "L"),
      (self.boundaries.L_LV_normal, self.boundaries.L_PMAX, "L"),
      (self.boundaries.L_LS, self.boundaries.L_LV_anomal, "L"),
      (self.boundaries.L_LS, self.boundaries.L_PMAX, "L"),
      # Mono condition
      (self.boundaries.V_LV, no_boundary, "V"),
      (self.boundaries.V_SV, no_boundary, "V"),
      (self.boundaries.LV_TMAX, no_boundary, "LV_EXTRAPOLATE"),
    ]

    # Array of labels
    self.labels = np.array([region[2] for region in self.region_definitions],
      dtype=object)
    # num -> str representation of region labels
    self.label_str_by_index: np.array = np.unique([*self.labels, "DEFAULT"])
    # str -> num representation of region labels
    self.label_index_by_str: dict = {
      name: index for index, name in enumerate(self.label_str_by_index)
    }

    print("Building interpolators")
    # Define mapping from label (string) to region
    self.region_mapping = {
      "L": self.L,
      "LS": self.LS,
      "LSV": self.LSV,
      "LV": self.LV,
      "LV_EXTRAPOLATE": self.LV,
      "SV": self.SV,
      "V": self.V,
      "DEFAULT": self.L,
    }

    # Build list of interpolators 
    self.interpolators = [scipy.interpolate.LinearNDInterpolator(
        np.stack((1/self.region_mapping[name].v.ravel(),
                self.region_mapping[name].u.ravel()), axis=1),
        np.stack((self.region_mapping[name].T.ravel(),
                  self.region_mapping[name].p.ravel(),
                  self.region_mapping[name].s.ravel(),
                  self.region_mapping[name].y_l.ravel(),
                  self.region_mapping[name].y_s.ravel(),), axis=1),
      ) for name in self.label_str_by_index]

    print("Remeshing LV region")

    v_base = self.LV.v
    u_base = self.LV.u
    T_base = self.LV.T
    p_base = self.LV.p
    s_base = self.LV.s
    y_l_base = self.LV.y_l
    y_s_base = self.LV.y_s

    # Pick interpolator
    interpolator = self.interpolators[self.label_index_by_str["LV"]]

    v = self.LV.v.flatten()
    u = self.LV.u.flatten()
    T = self.LV.T.flatten()
    p = self.LV.p.flatten()
    s = self.LV.s.flatten()
    y_l = self.LV.y_l.flatten()
    y_s = self.LV.y_s.flatten()

    for i in range(self.NUM_ITERATIONS):

      (v_new, u_new, T_new, p_new, s_new, y_l_new, y_s_new) = \
        self.regrid_LV(interpolator,
        left_range_rho_resolution=100,
        right_range_rho_resolution=15,
        u_resolution=40)

      # Use as source points original points + all new regridded points
      v = np.concatenate((v.ravel(), v_new))
      u = np.concatenate((u.ravel(), u_new))
      T = np.concatenate((T.ravel(), T_new))
      p = np.concatenate((p.ravel(), p_new))
      s = np.concatenate((s.ravel(), s_new))
      y_l = np.concatenate((y_l.ravel(), y_l_new))
      y_s = np.concatenate((y_s.ravel(), y_s_new))

      interpolator = scipy.interpolate.LinearNDInterpolator(
        np.stack((1/v, u), axis=1),
        np.stack((T, p, s, y_l, y_s), axis=1),
      )

    # Use only original points + last set of regridded points
    (v_new, u_new, T_new, p_new, s_new, y_l_new, y_s_new) = self.regrid_LV(
      interpolator)
    v = np.concatenate((v_base.ravel(), v_new))
    u = np.concatenate((u_base.ravel(), u_new))
    T = np.concatenate((T_base.ravel(), T_new))
    p = np.concatenate((p_base.ravel(), p_new))
    s = np.concatenate((s_base.ravel(), s_new))
    y_l = np.concatenate((y_l_base.ravel(), y_l_new))
    y_s = np.concatenate((y_s_base.ravel(), y_s_new))
    # v = v_new
    # u = u_new
    # T = T_new
    # p = p_new
    # s = s_new
    # y_l = y_l_new
    # y_s = y_s_new

    self.interpolators[self.label_index_by_str["LV"]] = \
      scipy.interpolate.LinearNDInterpolator(
        np.stack((1/v, u), axis=1),
        np.stack((T, p, s, y_l, y_s), axis=1),
      )
    
    self.LV.u = u
    self.LV.v = v
    self.LV.s = s
    self.LV.p = p
    self.LV.T = T
    self.LV.y_l = y_l
    self.LV.y_s = y_s
    self.LV.y_v = 1.0 - (y_l + y_s)

    print("Init done")

  def regrid_LV(self, interpolator:callable,
              left_range_rho_resolution=100,
              right_range_rho_resolution=20,
              u_resolution=20):
    ''' Uses given interpolator to estimate grid in LV region
    Used to fix skinny triangle problem in LV region
    '''
    
    # Set fill points between LV_LVS and LV_TMAX
    fill_range_rho = np.linspace(1, self.boundaries.LV_TMAX.rho_R,
                                left_range_rho_resolution)[:-1]
    dummy_range_u = np.zeros((u_resolution,))
    mg_rho, mg_u = np.meshgrid(fill_range_rho, dummy_range_u, indexing="ij")
    # Interpolate range between top and bottom boundaries u(rho)
    for i in range(fill_range_rho.size):
      mg_u[i,:] = np.linspace(
        self.boundaries.LV_LSV.interpolator(fill_range_rho[i]), 
        self.boundaries.LV_TMAX.interpolator(fill_range_rho[i]),
        dummy_range_u.size)
    # Additional point coordinates
    add1_rho, add1_u = mg_rho[:,1:-1].flatten(), mg_u[:,1:-1].flatten()

    # Add between LV_LVS and L_LV
    fill_range_rho = np.linspace(self.boundaries.L_LV_normal.rho_L,
                                self.boundaries.L_LV_normal.rho_R,
                                right_range_rho_resolution)
    dummy_range_u = np.zeros((u_resolution,))
    mg_rho, mg_u = np.meshgrid(fill_range_rho, dummy_range_u, indexing="ij")
    # Interpolate range between top and bottom boundaries u(rho)
    for i in range(fill_range_rho.size):
      mg_u[i,:] = np.linspace(
        self.boundaries.LV_LSV.interpolator(fill_range_rho[i]), 
        self.boundaries.L_LV_normal.interpolator(fill_range_rho[i]),
        dummy_range_u.size)
    # Additional point coordinates
    add2_rho, add2_u = mg_rho[:,1:-1].flatten(), mg_u[:,1:-1].flatten()

    # Obtain origin coordinates (y_l, T) in LV region
    computed = interpolator(
      np.stack((np.concatenate((add1_rho, add2_rho)),
                np.concatenate((add1_u, add2_u))), axis=1))

    # Native computation from (y_l, T) coordinates
    T, p, s, y_l, y_s = computed.T
    T_target = T[np.where(~np.isnan(T))]
    y_l_target = y_l[np.where(~np.isnan(T))]
    p_computed = np.zeros_like(T_target)
    u_computed = np.zeros_like(T_target)
    v_computed = np.zeros_like(T_target)
    s_computed = np.zeros_like(T_target)
    for i in range(T_target.size):
      T = T_target[i]
      y_l = y_l_target[i]
      sat_triple = iapws95_light.prho_sat(T)
      p_computed[i], rhol, rhov = (sat_triple["psat"],
                                sat_triple["rho_satl"],
                                sat_triple["rho_satv"])
      s_l, h_l, u_l, w_l = all_pure_lv(rhol, T)
      s_v, h_v, u_v, w_v = all_pure_lv(rhov, T)

      u_computed[i] = y_l * u_l + (1 - y_l) * u_v
      v_computed[i] = y_l / rhol + (1 - y_l) / rhov
      s_computed[i] = y_l * s_l + (1 - y_l) * s_v

    return (v_computed, u_computed, T_target, p_computed, s_computed,
            y_l_target, np.zeros_like(y_l_target))

  def identify_regions(self, rho:np.array, u:np.array):
    ''' Identify region of (rho, u) and returns array of strings, and array
    of integers. Array of integers corresponds to strings through
    self.label_str_by_index[integer] and conversely
    self.label_index_by_str[string].

    Region labels can be {"L", "V", "LV", "LS", "SV", "LSV"}, or "DEFAULT" if
    the region is outside of the data region of validity. The region of validity
    can be extended by incorporating the solid phase region ("S"), higher
    temperatures, and lower pressures.

    '''

    # Find first region where membership, else label as "default"
    # In the case where multiple are true, only the first region is used as label
    membership_values = np.array([
      region[0].under(rho, u)
      & region[1].over(rho, u)
      for region in self.region_definitions])
    # Count # memberships (should be 1, or 0 if out of zone, and
    # possibly 2+ exactly on boundaries)
    membership_count = membership_values.sum(axis=0)
    # Extract region label, excluding default
    region_labels = self.labels[membership_values.argmax(axis=0)]
    # Add default case
    region_labels = np.where(membership_count >= 1, region_labels, "DEFAULT")
    # Convert to numeric
    region_labels_numeric = np.array([self.label_index_by_str[l]
                                      for l in region_labels])
    
    return region_labels, region_labels_numeric

  def approx_rho_u(self, rho:np.array, u:np.array, extrapolate=False):
    ''' Determine region for an array of (rho, u) and 
    return to interpolated value in corresponding regions.'''
    # Partition interpolators
    region_labels, region_labels_numeric = \
      self.identify_regions(rho, u)

    out = np.full((len(rho), 5), np.nan)
    for i, name in enumerate(self.label_str_by_index):
      if not extrapolate:
        if name == "DEFAULT" or name == "LV_EXTRAPOLATE":
          continue
      # Compute indices where label is i
      partition = np.where(region_labels_numeric == i)[0]
      # Use appropriate interpolant for region (T, p, s, y_l, y_s)
      out[partition, :] = self.interpolators[i](
        np.stack((rho[partition], u[partition],), axis=-1)
      )
    
    return out, region_labels_numeric

  def populate_LSV(self):
    ''' Populate LSV region. '''
    # Parametrize using y_cond and z_s, where y_s = z_s * y_cond
    y_cond_range = np.unique([
                            *np.linspace(0, 1 - 3e-2, 600),
                            *np.sort(1 - np.geomspace(1e-13, 3e-2, 301, endpoint=False)),
                            ])
    z_s_range = np.linspace(0, 1.0, 101)

    mg_y_cond, mg_z_s = np.meshgrid(y_cond_range, z_s_range)
    mg_u = np.zeros_like(mg_y_cond)
    mg_v = np.zeros_like(mg_y_cond)
    mg_s = np.zeros_like(mg_y_cond)
    mg_p = np.full_like(mg_y_cond, p_n)
    mg_T = np.full_like(mg_y_cond, T_n)

    for i in range(mg_y_cond.shape[0]):
      for j in range(mg_y_cond.shape[1]):
        y_cond, z_s = mg_y_cond[i,j], mg_z_s[i,j]
        y_s, y_l = z_s * y_cond, (1.0 - z_s) * y_cond
        mg_u[i,j] = y_s * u_s_n + y_l * u_l_n + (1 - y_s - y_l) * u_v_n
        mg_v[i,j] = y_s * v_s_n + y_l * v_l_n + (1 - y_s - y_l) * v_v_n
        mg_s[i,j] = y_s * s_s_n + y_l * s_l_n + (1 - y_s - y_l) * s_v_n
    return SimpleNamespace(
      u=mg_u, v=mg_v, s=mg_s, p=mg_p, T=mg_T,
      y_l=mg_y_cond * (1.0 - mg_z_s),
      y_s=mg_y_cond * mg_z_s,
      y_v=(1.0 - mg_y_cond),
      )

  def populate_LV(self):
    ''' Populate LV region. '''

    # LV_y_l_range = np.linspace(0, 1, 51)
    y_l_range = np.unique([
      *np.linspace(0, 1 - 3e-2, 40),
      *np.sort(1 - np.geomspace(1e-13, 3e-2, 21, endpoint=False)),
      ])
    T_range = np.linspace(T_n, self.T_MAX, 101)

    mg_y_l, mg_T = np.meshgrid(y_l_range, T_range)
    mg_u = np.zeros_like(mg_y_l)
    mg_v = np.zeros_like(mg_y_l)
    mg_s = np.zeros_like(mg_y_l)
    mg_p = np.zeros_like(mg_y_l)
    for i in range(mg_y_l.shape[0]):
      for j in range(mg_y_l.shape[1]):
        y_l, T = mg_y_l[i,j], mg_T[i,j]
        sat_triple = iapws95_light.prho_sat(T)
        mg_p[i,j], rhol, rhov = (sat_triple["psat"],
                                 sat_triple["rho_satl"],
                                 sat_triple["rho_satv"])
        s_l, h_l, u_l, w_l = all_pure_lv(rhol, T)
        s_v, h_v, u_v, w_v = all_pure_lv(rhov, T)

        mg_u[i,j] = y_l * u_l + (1 - y_l) * u_v
        mg_v[i,j] = y_l / rhol + (1 - y_l) / rhov
        mg_s[i,j] = y_l * s_l + (1 - y_l) * s_v
    return SimpleNamespace(
      u=mg_u, v=mg_v, s=mg_s, p=mg_p, T=mg_T,
      y_l=mg_y_l,
      y_s=np.zeros_like(mg_y_l),
      y_v=(1.0 - mg_y_l),
      )
  
  def populate_SV(self):
    # Populate SV region
    y_s_range = np.unique([
                            *np.linspace(0, 1 - 3e-2, 60),
                            *np.sort(1 - np.geomspace(1e-13, 3e-2, 31, endpoint=False)),
                            ])
    T_range = np.linspace(self.T_MIN, T_n, 51)
    mg_y_s, mg_T = np.meshgrid(y_s_range, T_range)
    mg_u = np.zeros_like(mg_y_s)
    mg_v = np.zeros_like(mg_y_s)
    mg_s = np.zeros_like(mg_y_s)
    mg_p = np.zeros_like(mg_y_s)
    for i in range(mg_y_s.shape[0]):
      for j in range(mg_y_s.shape[1]):
        y_s, T = mg_y_s[i,j], mg_T[i,j]

        mg_p[i,j] = IAPWS_ice.ice_I_sublimation_p(T)

        u_s = IAPWS_ice.u_iceI(mg_p[i,j], T)
        rho_s = IAPWS_ice.rho_iceI(mg_p[i,j], T)
        s_s = IAPWS_ice.s_iceI(mg_p[i,j], T)

        rhov = SV_rhov(mg_p[i,j], T)
        s_v, h_v, u_v, w_v = [float(out) for out in all_pure_lv(rhov, T)]

        mg_u[i,j] = y_s * u_s + (1 - y_s) * u_v
        mg_v[i,j] = y_s / rho_s + (1 - y_s) / rhov
        mg_s[i,j] = y_s * s_s + (1 - y_s) * s_v
    
    return SimpleNamespace(
      u=mg_u, v=mg_v, s=mg_s, p=mg_p, T=mg_T,
      y_l=np.zeros_like(mg_y_s),
      y_s=mg_y_s,
      y_v=(1.0 - mg_y_s),
      )

  def populate_LS(self):
    # Populate SL region
    y_s_range = np.linspace(0, 1, 30)
    T_range = np.linspace(self.T_I_min, T_n, 51)

    mg_y_s, mg_T = np.meshgrid(y_s_range, T_range)
    mg_u = np.zeros_like(mg_y_s)
    mg_v = np.zeros_like(mg_y_s)
    mg_p = np.zeros_like(mg_y_s)
    mg_s = np.zeros_like(mg_y_s)
    for i in range(mg_y_s.shape[0]):
      for j in range(mg_y_s.shape[1]):
        y_s, T = mg_y_s[i,j], mg_T[i,j]

        mg_p[i,j] = IAPWS_ice.ice_I_melting_p(T)

        u_s = IAPWS_ice.u_iceI(mg_p[i,j], T)
        rho_s = IAPWS_ice.rho_iceI(mg_p[i,j], T)
        s_s = IAPWS_ice.s_iceI(mg_p[i,j], T)

        rhol = SL_rhol(mg_p[i,j], T)
        s_l, h_l, u_l, w_l = [float(out) for out in all_pure_lv(rhol, T)]

        mg_u[i,j] = y_s * u_s + (1 - y_s) * u_l
        mg_v[i,j] = y_s / rho_s + (1 - y_s) / rhol
        mg_s[i,j] = y_s * s_s + (1 - y_s) * s_l
    return SimpleNamespace(
      u=mg_u, v=mg_v, s=mg_s, p=mg_p, T=mg_T,
      y_l=(1.0 - mg_y_s),
      y_s=mg_y_s,
      y_v=np.zeros_like(mg_y_s),
      )
  
  def populate_L(self):

    # Split temperature range at triple point
    T_range_sub = np.linspace(self.T_I_min, T_n, 31, endpoint=False)
    T_range_sup = np.linspace(T_n, self.T_MAX, 31)

    # Compute liquid density bounds for temperature range below triple point
    rho_low_sub = np.zeros_like(T_range_sub)
    rho_high_sub = np.zeros_like(T_range_sup)
    for i, T in enumerate(T_range_sub):
      # Minimum pressure for liquid from melting curve
      rho_low_sub[i] = SL_rhol(IAPWS_ice.ice_I_melting_p(T), T,
                               num_newton_iter=20, verbose=False)
      # Max pressure from chosen validity range
      rho_high_sub[i] = SL_rhol(self.p_at_T_I_min, T, num_newton_iter=20, verbose=False)

    # Compute liquid density bounds for temperature range above triple point
    rho_low_sup = np.zeros_like(T_range_sup)
    rho_high_sup = np.zeros_like(T_range_sup)
    for i, T in enumerate(T_range_sup):
      # Minimum pressure for liquid from liquid-vapor saturation curve
      sat_triple = iapws95_light.prho_sat(T)
      rho_low_sup[i] = sat_triple["rho_satl"]
      # Max pressure from chosen validity range
      rho_high_sup[i] = SL_rhol(self.p_at_T_I_min, T, num_newton_iter=20, verbose=False, init=1000)

    # Fuse T array
    T_range_all = ((*T_range_sub, *T_range_sup))
    # Construct temperature and density sample values (non-Cartesian)
    rho_temp = np.zeros((100,))
    mg_T, mg_rho = np.meshgrid(T_range_all, rho_temp)
    # Replace L_mg values with non-Cartesian
    for i, T in enumerate(T_range_sub):
      mg_rho[:,i] = np.linspace(rho_low_sub[i], rho_high_sub[i], len(rho_temp))
    for i, T in enumerate(T_range_sup):
      mg_rho[:,i+len(T_range_sub)] = np.linspace(rho_low_sup[i], rho_high_sup[i], len(rho_temp))

    # Compute thermodynamic variables for each pair (rho, T)
    mg_s = np.zeros_like(mg_T)
    mg_u = np.zeros_like(mg_T)
    mg_p = np.zeros_like(mg_T)
    for i, (rho, T) in enumerate(zip(mg_rho.ravel(), mg_T.ravel())):
      s, h, u, w, potentials = all_pure_lv(rho, T, dump=True)
      mg_p.ravel()[i] = rho * iapws95_light.R * T * (
        1 + rho / iapws95_light.rhoc * potentials["phir_d"])
      mg_s.ravel()[i], mg_u.ravel()[i] = s, u

    return SimpleNamespace(
      u=mg_u, v=1/mg_rho, s=mg_s, p=mg_p, T=mg_T,
      y_l=np.ones_like(mg_T),
      y_s=np.zeros_like(mg_T),
      y_v=np.zeros_like(mg_T),
      )
  
  def populate_V(self):

    T_range_sub = np.linspace(self.T_MIN, T_n, 51, endpoint=False)
    T_range_sup = np.linspace(T_n, self.T_MAX, 51)
    N_rho = 401

    # Compute liquid density bounds for temperature range below triple point
    rho_low_sub = np.zeros_like(T_range_sub)
    rho_high_sub = np.zeros_like(T_range_sub)
    for i, T in enumerate(T_range_sub):
      # Minimum pressure for liquid from melting curve
      rho_low_sub[i] = SV_rhov(self.P_MIN, T, num_newton_iter=20)
      # Max pressure from chosen validity range
      rho_high_sub[i] = SV_rhov(IAPWS_ice.ice_I_sublimation_p(T), T,
                                num_newton_iter=20)

    # Compute liquid density bounds for temperature range above triple point
    rho_low_sup = np.zeros_like(T_range_sup)
    rho_high_sup = np.zeros_like(T_range_sup)
    for i, T in enumerate(T_range_sup):
      # Minimum pressure for liquid from melting curve
      rho_low_sup[i] = SV_rhov(self.P_MIN, T, num_newton_iter=20)
      # Minimum pressure for liquid from liquid-vapor saturation curve
      sat_triple = iapws95_light.prho_sat(T)
      rho_high_sup[i] = sat_triple["rho_satv"]

    # Fuse T array
    T_range_all = ((*T_range_sub, *T_range_sup))
    # Construct temperature and density sample values (non-Cartesian)
    rho_temp = np.zeros((N_rho,))
    mg_T, mg_rho = np.meshgrid(T_range_all, rho_temp)
    # Replace values with non-Cartesian
    for i, T in enumerate(T_range_sub):
      mg_rho[:,i] = np.geomspace(rho_low_sub[i], rho_high_sub[i], len(rho_temp))
    for i, T in enumerate(T_range_sup):
      mg_rho[:,i+len(T_range_sub)] = np.geomspace(rho_low_sup[i], rho_high_sup[i], len(rho_temp))

    # Compute thermodynamic variables for each pair (rho, T)
    mg_s = np.zeros_like(mg_T)
    mg_u = np.zeros_like(mg_T)
    mg_p = np.zeros_like(mg_T)
    for i, (rho, T) in enumerate(zip(mg_rho.ravel(), mg_T.ravel())):
      s, h, u, w, potentials = all_pure_lv(rho, T, dump=True)
      mg_p.ravel()[i] = rho * iapws95_light.R * T * (
        1 + rho / iapws95_light.rhoc * potentials["phir_d"])
      mg_s.ravel()[i], mg_u.ravel()[i] = s, u

    return SimpleNamespace(
      u=mg_u, v=1/mg_rho, s=mg_s, p=mg_p, T=mg_T,
      y_l=np.ones_like(mg_T),
      y_s=np.zeros_like(mg_T),
      y_v=np.zeros_like(mg_T),
      )

  def boundary_L_LV_normal(self, N_samples=100):
    ''' Boundary between L and LV in the temperature range where thermal
    expansion is normal (volume increases with temperature).
    Curve is parametrized by temperature in range (T_L_inversion, T_max).
    Constraint is the liquid-vapor saturation curve.
    '''

    T_range = np.linspace(self.T_L_inversion, self.T_MAX, N_samples)
    rhol_sat = np.zeros_like(T_range)
    ul_sat = np.zeros_like(T_range)
    for i, T in enumerate(T_range):
      rhol_sat[i] = iapws95_light.prho_sat(T)["rho_satl"]
      ul_sat[i] = all_pure_lv(rhol_sat[i], T)[2]

    return Boundary(rhol_sat, ul_sat,
                    rhol_sat.min(),
                    rhol_sat.max())
  
  def boundary_L_LV_anomal(self, N_samples=100):
    ''' Boundary between L and LV in the temperature range where thermal
    expansion is anomalous (volume decreases with temperature).
    Curve is parametrized by temperature in range (T_L_inversion, T_n).
    Constraint is the liquid-vapor saturation curve.
    '''

    T_range = np.linspace(T_n, self.T_L_inversion, N_samples)
    rhol_sat = np.zeros_like(T_range)
    ul_sat = np.zeros_like(T_range)
    for i, T in enumerate(T_range):
      rhol_sat[i] = iapws95_light.prho_sat(T)["rho_satl"]
      ul_sat[i] = all_pure_lv(rhol_sat[i], T)[2]

    return Boundary(rhol_sat, ul_sat,
                    rhol_sat.min(),
                    rhol_sat.max())
  
  def boundary_L_TMAX(self, N_samples=100):
    ''' Outside boundary of L region at max temperature.
    Curve is parametrized by liquid density, in range
     (rho_LV(TMAX), rho(p_melt(T_I_min))).
    Constraint is the temperature T = T_MAX.
    Asserts that max temperature is below the critical point.

    Additional info: while the melting curve is not directly relevant to this
    boundary, the largest pressure this model samples at is the melting
    pressure at temperature T_I_min.
    '''

    # Check that the model specifies a temperature lower than the critical pt
    assert(self.T_MAX < iapws95_light.Tc)
    # Compute the max density to sample to (pure liquid at max p = p(T_I_min))
    rhol_max = SL_rhol(self.p_at_T_I_min, self.T_MAX,
                       init=820, num_newton_iter=8)
    # Compute the min density to sample to (saturated liquid at TMAX)
    rhol_min = iapws95_light.prho_sat(self.T_MAX)["rho_satl"]
    rhol_range = np.linspace(rhol_min, rhol_max, N_samples)
    # Compute specific energy u at TMAX
    u = np.array([all_pure_lv(rhol, self.T_MAX)[2] for rhol in rhol_range])

    return Boundary(rhol_range, u,
                    rhol_range.min(),
                    rhol_range.max())

  def boundary_L_PMAX(self, N_samples=100):
    ''' Outside boundary of L region at max pressure.
    Curve is parametrized by temperature in range (T_I_min, T_max).
    Constraint is p = p_melt(T_I_min).

    At p_melt(T_I_min), the liquid phase is valid for T > T_I_min. Here
    T_I_min < T_n (below the triple point), as the melting point drops as
    pressure increases above the triple point.
    '''

    T_range = np.linspace(self.T_I_min, self.T_MAX, N_samples)
    rho = np.zeros_like(T_range)
    u = np.zeros_like(T_range)
    for i, T in enumerate(T_range):
      # Compute rhol(p,T) iteratively, fixing p = p_melt(T_I_min)
      rho[i] = SL_rhol(self.p_at_T_I_min, T,
                       num_newton_iter=20, init=1000)
      u[i] = all_pure_lv(rho[i], T)[2]

    return Boundary(rho, u,
                    rho.min(),
                    rho.max())

  def boundary_SV_LSV(self, N_samples=100):
    ''' Boundary between SV and LSV.
    Curve is parametrized by solid mass fraction y_s.
    Constraint is the triple point (p_n and T_n).
    '''

    y_s_range = np.linspace(0, 1, N_samples)
    # Compute specific volume and energy at triple point
    v = (v_s_n * y_s_range + v_v_n * (1 - y_s_range))
    u = (u_s_n * y_s_range + u_v_n * (1 - y_s_range))
    return Boundary(1.0/v, u, (1.0/v).min(), (1.0/v).max())

  def boundary_LV_LSV(self, N_samples=100):
    ''' Boundary between LV and LSV.
    Curve is parametrized by liquid mass fraction y_l.
    Constraint is the triple point (p_n and T_n).
    '''

    y_l_range = np.linspace(0, 1, N_samples)
    # Compute specific volume and energy at triple point
    v = (v_l_n * y_l_range + v_v_n * (1 - y_l_range))
    u = (u_l_n * y_l_range + u_v_n * (1 - y_l_range))
    return Boundary(1.0/v, u, (1.0/v).min(), (1.0/v).max())

  def boundary_LS_LSV(self, N_samples=100):
    ''' Boundary between LS and LSV.
    Curve is parametrized by liquid mass fraction y_l.
    Constraint is the triple point (p_n and T_n).
    '''

    y_l_range = np.linspace(0, 1, N_samples)
    u = np.zeros_like(y_l_range)
    v = np.zeros_like(y_l_range)

    # Compute (rho, u) of liquid at triple point
    rhol_T_n = SL_rhol(IAPWS_ice.ice_I_melting_p(T_n), T_n)
    ul_T_n = all_pure_lv(rhol_T_n, T_n)[2]
    # Compute (rho, u) of solid at triple point
    rhos_T_n = IAPWS_ice.rho_iceI(IAPWS_ice.ice_I_melting_p(T_n), T_n)
    us_T_n = IAPWS_ice.u_iceI(IAPWS_ice.ice_I_melting_p(T_n), T_n)
    for i, y_l in enumerate(y_l_range):
      u[i] = y_l * ul_T_n + (1.0 - y_l) * us_T_n
      v[i] = y_l / rhol_T_n + (1.0 - y_l) / rhos_T_n
    return Boundary(1.0/v, u, (1.0/v).min(), (1.0/v).max())

  def boundary_V_LV(self, N_samples=100):
    ''' Boundary between V and LV.
    Curve is parametrized by temperature in range (T_n, T_max).
    Constraint is the liquid-vapor saturation curve.
    '''

    # Compute pure vapor over T range
    T_range = np.linspace(T_n, self.T_MAX, N_samples)
    uv_range = np.zeros_like(T_range)
    rhov_range = np.zeros_like(T_range)
    for i, T in enumerate(T_range):
      # Compute saturation vapor values at temperature T
      sat_triple = iapws95_light.prho_sat(T)
      s_v, h_v, u_v, w_v = all_pure_lv(sat_triple["rho_satv"], T)
      uv_range[i] = u_v
      rhov_range[i] = sat_triple["rho_satv"]
    return Boundary(rhov_range, uv_range, rhov_range.min(), rhov_range.max())

  def boundary_LV_TMAX(self, N_samples=100):
    ''' Outside boundary of LV region at max temperature.
    Curve is parametrized by liquid mass fraction y_l.
    Constraint is the temperature T = T_MAX.
    Asserts that max temperature is below the critical point.

    Additional info: while the melting curve is not directly relevant to this
    boundary, the largest pressure this model samples at is the melting
    pressure at temperature T_I_min.
    '''

    # Check that the model specifies a temperature lower than the critical pt
    assert(self.T_MAX < iapws95_light.Tc)

    y_l_range = np.linspace(0, 1, N_samples)
    u = np.zeros_like(y_l_range)
    v = np.zeros_like(y_l_range)
    # Compute (rho, u) of saturated liquid and vapor
    sat_triple = iapws95_light.prho_sat(self.T_MAX)
    _, rhol, rhov = sat_triple["psat"], sat_triple["rho_satl"], sat_triple["rho_satv"]
    _, _, u_l, _ = all_pure_lv(rhol, self.T_MAX)
    _, _, u_v, _ = all_pure_lv(rhov, self.T_MAX)
    for i, y_l in enumerate(y_l_range):
      u[i] = y_l * u_l + (1.0 - y_l) * u_v
      v[i] = y_l / rhol + (1.0 - y_l) / rhov
    return Boundary(1.0/v, u, (1.0/v).min(), (1.0/v).max())

  def boundary_V_SV(self, N_samples=100):
    ''' Boundary between V and SV.
    Curve is parametrized by temperature in range (T_min, T_n).
    Constraint is the solid-vapor sublimation curve.
    '''

    T_range = np.linspace(self.T_MIN, T_n, N_samples)
    # Compute density rho(p(T)) along sublimation curve
    rho = np.array([
      SV_rhov(IAPWS_ice.ice_I_sublimation_p(T), T)
      for T in T_range])
    u = np.array([float(all_pure_lv(rhov, T)[2])
      for rhov, T in zip(rho, T_range)])
    return Boundary(rho, u, rho.min(), rho.max())

  def boundary_SV_TMIN(self, N_samples=100):
    ''' Outside boundary of SV region at minimum temperature.
    Curve is parametrized by solid mass fraction y_s.
    Constraint is the temperature T = T_MIN.
    '''

    y_s_range = np.linspace(0, 1, N_samples)
    # Compute state at TMIN along sublimation curve
    p_MIN = IAPWS_ice.ice_I_sublimation_p(self.T_MIN)
    rhos_TMIN = IAPWS_ice.rho_iceI(p_MIN, self.T_MIN)
    us_TMIN = IAPWS_ice.u_iceI(p_MIN, self.T_MIN)
    rhov_TMIN = SV_rhov(p_MIN, self.T_MIN)
    uv_TMIN = all_pure_lv(rhov_TMIN, self.T_MIN)[2]
    # Compute mixture (u, v)
    u = y_s_range * us_TMIN + (1 - y_s_range) * uv_TMIN
    v = y_s_range / rhos_TMIN + (1 - y_s_range) / rhov_TMIN
    return Boundary(1.0/v, u, (1.0/v).min(), (1.0/v).max())

  def boundary_S_SV(self, N_samples=100):
    ''' Boundary between S and SV.
    Curve is parametrized by temperature in range (T_min, T_n).
    Constraint is the solid-vapor sublimation curve.
    '''

    T_range = np.linspace(self.T_MIN, T_n, N_samples)
    # Compute p(T) along sublimation curve
    p = IAPWS_ice.ice_I_sublimation_p(T_range)
    rho = np.array([IAPWS_ice.rho_iceI(p, T)
                    for (p, T) in zip(p, T_range)])
    u = np.array([IAPWS_ice.u_iceI(p, T)
                  for (p, T) in zip(p, T_range)])
    return Boundary(rho, u,
                    rho.min(),
                    rho.max())
  
  def boundary_L_LS(self, N_samples=100):
    ''' Boundary between L and LS.
    Curve is parametrized by temperature in range (T_I_min, T_n).
    Constraint is the melting curve.
    '''

    T_range = np.linspace(self.T_I_min, T_n, N_samples)
    p_melt_curve = IAPWS_ice.ice_I_melting_p(T_range)
    rho = np.array([SL_rhol(p, T)
                    for p, T in zip(p_melt_curve, T_range)])
    u = np.array([all_pure_lv(rhov, T)[2]
                    for rhov, T in zip(rho, T_range)])
    return Boundary(rho, u, rho.min(), rho.max())
   
  def boundary_S_LS(self, N_samples=100):
    ''' Boundary between S and LS.
    Curve is parametrized by temperature in range (T_I_min, T_n).
    Constraint is the melting curve.
    '''

    T_range = np.linspace(self.T_I_min, T_n, N_samples)
    p_melt_curve = IAPWS_ice.ice_I_melting_p(T_range)
    rho = np.array([IAPWS_ice.rho_iceI(p,T)
                    for p, T in zip(p_melt_curve, T_range)])
    u = np.array([IAPWS_ice.u_iceI(p,T)
                  for p, T in zip(p_melt_curve, T_range)])
    return Boundary(rho, u, rho.min(), rho.max())

  def boundary_LS_T_I_MIN(self, N_samples=100):
    ''' Outside boundary of LS region at temperature T_I_MIN.
    Curve is parametrized by liquid mass fraction y_l.
    Constraint is the temperature T = T_I_MIN.

    T_I_MIN is the lowest temperature valid for the melting curve, not the
    global minimum temperature. T_I_MIN can be chosen higher to provide a
    numerical buffer.
    '''

    y_l_range = np.linspace(0, 1, N_samples)
    u = np.zeros_like(y_l_range)
    v = np.zeros_like(y_l_range)
    # Compute properties along saturation curve, at temperature T_MIN
    p = IAPWS_ice.ice_I_melting_p(self.T_I_min)
    rhol_T_min = SL_rhol(p, self.T_I_min)
    ul_T_min = all_pure_lv(rhol_T_min, self.T_I_min)[2]
    rhos_T_min = IAPWS_ice.rho_iceI(p, self.T_I_min)
    us_T_min = IAPWS_ice.u_iceI(p, self.T_I_min)
    for i, y_l in enumerate(y_l_range):
      u[i] = y_l * ul_T_min + (1.0 - y_l) * us_T_min
      v[i] = y_l / rhol_T_min + (1.0 - y_l) / rhos_T_min
    return Boundary((1.0/v), u, (1.0/v).min(), (1.0/v).max())

  def plot_boundaries(self):
    ''' Plots boundaries between phase or phase assemblies. For debug use. '''

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1, figsize=(5,8))

    for boundary in self.boundaries:
      axes[0].plot(boundary.rho, boundary.u)
      axes[0].set_xlabel("density (kg/m^3)")
      axes[0].set_ylabel("specific energy (J/kg)")
      axes[1].plot(boundary.rho, boundary.u)
      axes[1].set_xlim(800, 1150)
      axes[1].set_ylim(-.6e6, 1e6)
      axes[1].set_xlabel("density (kg/m^3)")
      axes[1].set_ylabel("specific energy (J/kg)")
      axes[2].semilogx(1/boundary.rho, boundary.u)
      axes[2].set_xlabel("specific volume (m^3/g)")
      axes[2].set_ylabel("specific energy (J/kg)")

    fig.set_tight_layout(True)
    return fig, axes
  
if __name__ == "__main__":
  Model()

  # Below is used to compute the density-temperature inversion point in L
  # import scipy.optimize
  # scipy.optimize.minimize(
  #   lambda T: -float(iapws95_light.prho_sat(T)["rho_satl"]),
  # 277.14968116)