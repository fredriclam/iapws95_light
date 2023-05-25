#cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np
include "float_phi_functions.pyx"
# Inclusion replaces:
# ctypedef double DTYPE_t

'''
TODO: Return temperature as well as pressure
* Proof Cython implementation, add no-checks for runtime opt
Reduce redunant phi computation
Add initial guess low-order model
Optimize slowest path
Tolerance as an arg,
Try with, without trust region
'''

cdef extern from "math.h":
  double exp(double x)
  double log(double x)
  double fabs(double x)
  double min(double x, double y)
  double max(double x, double y)

''' Estimate initial temperature '''
# Set liquid linearization point
cdef DTYPE_t rho_w0 = 997.0
cdef DTYPE_t T_w0 = 300.0
cdef DTYPE_t p_w0 = 1089221.889890747 # p(rho_w0, T_w0)
cdef DTYPE_t e_w0 = 112471.63726846411 # u(rho_w0, T_w0)
cdef DTYPE_t c_v_w0 = 4126.9245339247345 # c_v(rho_w0, T_w0)
# Set saturated vapour linearization point
cdef DTYPE_t T_gas_ref = 100+273.15
cdef DTYPE_t p_gas_ref = 101417.99665995548 # prho_sat(273.15+100)["psat"]
cdef DTYPE_t rho_gas_ref = 0.5981697919259701 # prho_sat(273.15+100)["rho_satv"]
cdef DTYPE_t e_gas_ref = 2506022.7123141396 # u(rho_gas_ref, 273.15+100)
cdef DTYPE_t c_v_gas_ref = 1555.8307465765565 # c_v(rho_gas_ref, 273.15+100)
# Define output struct (rhow, p, T)
cdef struct TriplerhopT:
  DTYPE_t rhow
  DTYPE_t p
  DTYPE_t T
# Experimental feature: regression weights for est. T(yw, vol_energy, rho_mix)
cdef DTYPE_t[12] static_regression_weights = [
  115.921763310165275, -316.916654834257770, 124.876068728982972,
  -111.538568414603972, 68.024491357232435, -211.349261918711562,
  12.208660020549521, 60.443940920122770, 183.657082274465807,
  37.811498814731365, 72.941538305957451, -7.341630723542266,]
# Parameter packing for WLMA model
cdef struct WLMAParams:
  DTYPE_t vol_energy
  DTYPE_t rho_mix
  DTYPE_t yw
  DTYPE_t ya
  DTYPE_t K
  DTYPE_t p_m0
  DTYPE_t rho_m0
  DTYPE_t c_v_m0
  DTYPE_t R_a
  DTYPE_t gamma_a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t poly_T_residual_est(
    DTYPE_t yw, DTYPE_t vol_energy, DTYPE_t rho_mix) noexcept:  
  ''' Estimate residual based on a posteriori regression for reference magma
  parameters. '''
  cdef DTYPE_t x0 = yw
  cdef DTYPE_t x1 = vol_energy/rho_mix / 1e6 
  cdef DTYPE_t x2 = rho_mix / 1e3
  return static_regression_weights[0]\
    + static_regression_weights[1]*x0 \
    + static_regression_weights[2]*x1 \
    + static_regression_weights[3]*x2 \
    + static_regression_weights[4]*x0*x0 \
    + static_regression_weights[5]*x1*x1 \
    + static_regression_weights[6]*x2*x2 \
    + static_regression_weights[7]*x0*x1 \
    + static_regression_weights[8]*x0*x2 \
    + static_regression_weights[9]*x1*x2 \
    + static_regression_weights[10]*x1*x1*x1 \
    + static_regression_weights[11]*x1*x1*x1*x1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t linear_T_est(DTYPE_t yw, DTYPE_t vol_energy, DTYPE_t rho_mix,
    DTYPE_t c_v_m0):
  cdef DTYPE_t ym = 1.0 - yw
  cdef DTYPE_t T
  if rho_mix < 10.0:
    # Due to an issue with flip-flopping for a low-density gas:
    T = ((vol_energy / rho_mix) - yw * (e_gas_ref - c_v_gas_ref * T_gas_ref)) \
    / (yw * c_v_gas_ref + ym * c_v_m0)
  else:
    # Estimate temperature
    T = ((vol_energy / rho_mix) - yw * (e_w0 - c_v_w0 * T_w0)) \
      / (yw * c_v_w0 + ym * c_v_m0)
  return T

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t magma_mech_energy(DTYPE_t p, 
    DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0):
  ''' Mechanical component of internal energy as integral -p dv.
  First term is -(p-p0)dv, and second is -p0 dv, third is shift. '''
  # Nondimensional variable (for zero pressure)
  cdef DTYPE_t u0 = p_m0 / (K - p_m0)
  # Evaluate energy at zero pressure
  cdef DTYPE_t e_m0 = K / rho_m0 * (u0 - log(1.0 + u0)) - p_m0 / rho_m0 * u0
  # Nondimensional variable for pressure p
  cdef DTYPE_t u = (p_m0 - p)/(p + K - p_m0)
  # Return energy shifted by energy at zero pressure
  return K / rho_m0 * (u - log(1.0 + u)) - p_m0 / rho_m0 * u - e_m0

cpdef DTYPE_t mix_sound_speed(DTYPE_t rhow, DTYPE_t p, DTYPE_t T,
    WLMAParams params):
  ''' Isentropic sound speed.
  Intermediate computations are done through partials of phasic volume in
  Gibbs (p,T) coordinates. For Helmholtz variables (v, T), one can obtain
    dv/dT = - (dv/dp) * (dp/dT)
  from the cyclic relation, where dv/dp = 1/(dp/dv).
  ''' # TODO: check python interactions in .html

  # TODO: HACK: this is an approximation (weigh out water since inaccuracy in
  # rhow can yield exponentially incorrect c_w)
  if params.yw <= 1e-6:
    params.yw = 0.0

  cdef unsigned short i
  # Compute intermediates
  cdef DTYPE_t ym = 1.0 - (params.ya + params.yw)
  cdef DTYPE_t rho_m = params.rho_m0 * (1.0 + (p - params.p_m0) / params.K)
  cdef DTYPE_t d = rhow / rhoc
  # TODO: document triple point clipping
  # Set reciprocal reduced temperature for water
  cdef DTYPE_t t = Tc / max(T, 273.16)

  # Check for water phase equilibrium, with x = 0 default
  cdef DTYPE_t x = 0.0
  cdef DTYPE_t _c0, _c1, _c2
  cdef DTYPE_t dsatl = 1.0
  cdef DTYPE_t dsatv = 0
  cdef DTYPE_t sat_atol = 0.5e-2
  cdef Pair sat_pair
  cdef Derivatives_phir_0_1_2 _phirall0, _phirall1
  if t > 1.0:
    _c0 = 1.0-1.0/t
    _c1 = _c0**(1.0/6.0)
    _c2 = _c1 * _c1
    # Check auxiliary equation for saturation curve
    for i in range(6):
      dsatl += satl_coeffsb[i] * pow_fd(_c2, satl_powsb_times3[i])
      dsatv += satv_coeffsc[i] * pow_fd(_c1, satv_powsc_times6[i])
    dsatv = exp(dsatv)
    # Check if in or near phase equilibrium region
    if d < dsatl + sat_atol and d > dsatv - sat_atol:
      # Compute precise saturation curve and saturation pressure
      sat_pair = rho_sat(Tc / t)
      dsatl = sat_pair.first / rhoc
      dsatv = sat_pair.second / rhoc
      if d <= dsatl and d >= dsatv:
        # Compute vapour mass fraction
        x = (1.0 / rhow - 1.0 / sat_pair.first) \
             / (1.0 / sat_pair.second - 1.0 / sat_pair.first)
  
  # Compute partials for water phase(s)
  cdef DTYPE_t dvdp_w0, dpdT_w0, dvdT_w0, c_v_w0
  cdef DTYPE_t dvdp_w1, dpdT_w1, dvdT_w1, c_v_w1
  if x <= 0.0 or x >= 1.0:
    # Compute single-phase water partials(dv/dp)_T, (dv/dT)_p
    _phirall0 = fused_phir_all(d, t)
    dvdp_w0 = -1.0 / (rhow * rhow * R * Tc / t * (1.0
      + 2.0 * d * _phirall0.phir_d + d * d * _phirall0.phir_dd))
    dpdT_w0 = rhow * R * (
      1.0 + d * _phirall0.phir_d - t * d * _phirall0.phir_dt)
    dvdT_w0 = -dvdp_w0 * dpdT_w0
    c_v_w0  = c_v(rhow, Tc / t)
    # Second phase is weighted out with zero mass fraction
    dvdp_w1 = 1.0 # In denominator
    dpdT_w1 = 0.0
    dvdT_w1 = 0.0
    c_v_w1  = 0.0
  else:
    # Compute partials in both phases of water
    _phirall0 = fused_phir_all(dsatl, t)
    _phirall1 = fused_phir_all(dsatv, t)
    # Liquid
    dvdp_w0 = -1.0 / (sat_pair.first * sat_pair.first * R * Tc / t * (1.0
      + 2.0 * dsatl * _phirall0.phir_d + dsatl * dsatl * _phirall0.phir_dd))
    dpdT_w0 = sat_pair.first * R * (
      1.0 + dsatl * _phirall0.phir_d - t * dsatl * _phirall0.phir_dt)
    dvdT_w0 = - dvdp_w0 * dpdT_w0
    c_v_w0  = c_v(sat_pair.first, Tc / t)
    # Vapour
    dvdp_w1 = -1.0 / (sat_pair.second * sat_pair.second * R * Tc / t * (1.0
      + 2.0 * dsatv * _phirall1.phir_d + dsatv * dsatv * _phirall1.phir_dd))
    dpdT_w1 = sat_pair.second * R * (
      1.0 + dsatv * _phirall1.phir_d - t * dsatv * _phirall1.phir_dt)
    dvdT_w1 = - dvdp_w1 * dpdT_w1
    c_v_w1  = c_v(sat_pair.second, Tc / t)

  # Assemble array of mass fractions
  cdef DTYPE_t[4] arr_y = [params.ya, params.yw*(1.0-x), params.yw*x, ym]
  # Assemble array of phasic (dv/dp)_T
  cdef DTYPE_t[4] arr_vp = [
    -params.R_a * T / (p * p),
    dvdp_w0,
    dvdp_w1,
    -1.0 / (rho_m * rho_m) * params.rho_m0 / params.K,
  ]
  # Assemble array of phasic (dv/dT)_p
  cdef DTYPE_t[4] arr_vT = [
    params.R_a / p,
    dvdT_w0,
    dvdT_w1,
    0.0,
  ]
  # Assemble array of phasic isochoric heat capacities
  cdef DTYPE_t[4] arr_c_v = [
    params.R_a / (params.gamma_a - 1.0),
    c_v_w0,
    c_v_w1,
    params.c_v_m0,
  ]

  # Compute dot products of y and phasic derivatives
  cdef DTYPE_t weighted_c_v = 0.0
  cdef DTYPE_t weighted_vp = 0.0
  cdef DTYPE_t weighted_vT = 0.0
  cdef DTYPE_t weighted_vT2_vp = 0.0
  for i in range(4):
    weighted_c_v += arr_y[i] * arr_c_v[i]
    weighted_vp += arr_y[i] * arr_vp[i]
    weighted_vT += arr_y[i] * arr_vT[i]
    weighted_vT2_vp += arr_y[i] * arr_vT[i] * arr_vT[i] / arr_vp[i]

  # Compute mixture (dv/dp)_s
  cdef DTYPE_t dv_dp_s = weighted_vp + \
    T * weighted_vT * weighted_vT / (weighted_c_v - T * weighted_vT2_vp)
  # Convert to sound speed (drho/dp)_s and return
  if -1.0/(params.rho_mix * params.rho_mix * dv_dp_s) <= 0.0:
    return 0.0
  return sqrt(-1.0/(params.rho_mix * params.rho_mix * dv_dp_s))

def cons_derivatives_pT(rhow, p, T, params:WLMAParams):
  ''' (p,T)-derivatives of volume, energy weighted by mass fractions. '''

  cdef unsigned short i
  # Compute intermediates
  cdef DTYPE_t ym = 1.0 - (params.ya + params.yw)
  cdef DTYPE_t rho_m = params.rho_m0 * (1.0 + (p - params.p_m0) / params.K)
  cdef DTYPE_t d = rhow / rhoc
  cdef DTYPE_t t = Tc / T

  # Check for water phase equilibrium, with x = 0 default
  cdef DTYPE_t x = 0.0
  cdef DTYPE_t _c0, _c1, _c2
  cdef DTYPE_t dsatl = 1.0
  cdef DTYPE_t dsatv = 0
  cdef DTYPE_t sat_atol = 0.5e-2
  cdef Pair sat_pair
  cdef Derivatives_phir_0_1_2 _phirall0, _phirall1
  if t > 1.0:
    _c0 = 1.0-1.0/t
    _c1 = _c0**(1.0/6.0)
    _c2 = _c1 * _c1
    # Check auxiliary equation for saturation curve
    for i in range(6):
      dsatl += satl_coeffsb[i] * pow_fd(_c2, satl_powsb_times3[i])
      dsatv += satv_coeffsc[i] * pow_fd(_c1, satv_powsc_times6[i])
    dsatv = exp(dsatv)
    # Check if in or near phase equilibrium region
    if d < dsatl + sat_atol and d > dsatv - sat_atol:
      # Compute precise saturation curve and saturation pressure
      sat_pair = rho_sat(T)
      dsatl = sat_pair.first / rhoc
      dsatv = sat_pair.second / rhoc
      if d <= dsatl and d >= dsatv:
        # Compute vapour mass fraction
        x = (1.0 / rhow - 1.0 / sat_pair.first) \
             / (1.0 / sat_pair.second - 1.0 / sat_pair.first)
  
  # Compute partials for water phase(s)
  cdef DTYPE_t dvdp_w0, dpdT_w0, dvdT_w0, c_v_w0
  cdef DTYPE_t dedp_w0, dedT_w0
  cdef DTYPE_t _gibbs_dedrho, _gibbs_dedT, _gibbs_dpdrho   # coords (rho, T)
  cdef DTYPE_t dvdp_w1, dpdT_w1, dvdT_w1, c_v_w1
  cdef DTYPE_t dedp_w1, dedT_w1

  if x <= 0.0 or x >= 1.0:
    # Compute single-phase water partials(dv/dp)_T, (dv/dT)_p
    _phirall0 = fused_phir_all(d, t)
    dvdp_w0 = -1.0 / (rhow * rhow * R * T * (1.0
      + 2.0 * d * _phirall0.phir_d + d * d * _phirall0.phir_dd))
    dpdT_w0 = rhow * R * (
      1.0 + d * _phirall0.phir_d - t * d * _phirall0.phir_dt)
    # Isobaric derivative
    dvdT_w0 = -dvdp_w0 * dpdT_w0
    # Compute in gibbs coordinates
    _gibbs_dedrho = R * T / rhoc * _phirall0.phir_dt
    _gibbs_dedT   = c_v(rhow, T)
    _gibbs_dpdrho = -1.0 / (rhow * rhow * dvdp_w0)
    dedp_w0 = _gibbs_dedrho / _gibbs_dpdrho
    dedT_w0 = _gibbs_dedT - dedp_w0 * dpdT_w0
    # Second phase is weighted out with zero mass fraction
    dvdp_w1 = 1.0 # In denominator
    dpdT_w1 = 0.0
    dvdT_w1 = 0.0
    dedp_w1 = 0.0
    dedT_w1 = 0.0
  else:
    # Compute partials in both phases of water
    _phirall0 = fused_phir_all(dsatl, t)
    _phirall1 = fused_phir_all(dsatv, t)
    # Liquid
    dvdp_w0 = -1.0 / (sat_pair.first * sat_pair.first * R * T * (1.0
      + 2.0 * dsatl * _phirall0.phir_d + dsatl * dsatl * _phirall0.phir_dd))
    dpdT_w0 = sat_pair.first * R * (
      1.0 + dsatl * _phirall0.phir_d - t * dsatl * _phirall0.phir_dt)
    dvdT_w0 = - dvdp_w0 * dpdT_w0
    # Compute in gibbs coordinates
    _gibbs_dedrho = R * T / rhoc * _phirall0.phir_dt
    _gibbs_dedT   = c_v(sat_pair.first, T)
    _gibbs_dpdrho = -1.0 / (sat_pair.first * sat_pair.first * dvdp_w0)
    dedp_w0 = _gibbs_dedrho / _gibbs_dpdrho
    dedT_w0 = _gibbs_dedT - dedp_w0 * dpdT_w0

    # Vapour
    dvdp_w1 = -1.0 / (sat_pair.second * sat_pair.second * R * T * (1.0
      + 2.0 * dsatv * _phirall1.phir_d + dsatv * dsatv * _phirall1.phir_dd))
    dpdT_w1 = sat_pair.second * R * (
      1.0 + dsatv * _phirall1.phir_d - t * dsatv * _phirall1.phir_dt)
    dvdT_w1 = - dvdp_w1 * dpdT_w1
    # Compute in gibbs coordinates
    _gibbs_dedrho = R * T / rhoc * _phirall1.phir_dt
    _gibbs_dedT   = c_v(sat_pair.second, T)
    _gibbs_dpdrho = -1.0 / (sat_pair.second * sat_pair.second * dvdp_w1)
    dedp_w1 = _gibbs_dedrho / _gibbs_dpdrho
    dedT_w1 = _gibbs_dedT - dedp_w1 * dpdT_w1

  # Assemble array of mass fractions
  cdef DTYPE_t[4] arr_y = [params.ya, params.yw*(1.0-x), params.yw*x, ym]
  # Assemble array of phasic (dv/dT)_p
  cdef DTYPE_t[4] arr_vT = [
    params.R_a / p,
    dvdT_w0,
    dvdT_w1,
    0.0,
  ]
  # Assemble array of phasic (dv/dp)_T
  cdef DTYPE_t[4] arr_vp = [
    -params.R_a * T / (p * p),
    dvdp_w0,
    dvdp_w1,
    -1.0 / (rho_m * rho_m) * params.rho_m0 / params.K,
  ]
  # Assemble array of phasic (de/dT)_p
  cdef DTYPE_t[4] arr_eT = [
    params.R_a / (params.gamma_a - 1.0),
    dedT_w0,
    dedT_w1,
    params.c_v_m0,
  ]
  # Assemble array of phasic (de/dp)_T
  cdef DTYPE_t[4] arr_ep = [
    0.0,
    dedp_w0,
    dedp_w1,
    p / params.K * params.rho_m0 / (rho_m * rho_m),
  ]

  # Compute dot products of y and phasic derivatives
  cdef DTYPE_t weighted_vT = 0.0
  cdef DTYPE_t weighted_vp = 0.0
  cdef DTYPE_t weighted_eT = 0.0
  cdef DTYPE_t weighted_ep = 0.0
  for i in range(4):
    weighted_vT += arr_y[i] * arr_vT[i]
    weighted_vp += arr_y[i] * arr_vp[i]
    weighted_eT += arr_y[i] * arr_eT[i]
    weighted_ep += arr_y[i] * arr_ep[i]

  return weighted_vT, weighted_vp, weighted_eT, weighted_ep

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef Pair pure_phase_newton_pair(DTYPE_t d, DTYPE_t T, DTYPE_t rho_mix,
  DTYPE_t yw, DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0):
  ''' Returns Pair<value, slope> for Newton iteration of the pure phase volume-
  pressure equilibrium condition. '''
  # Compute coefficients
  cdef DTYPE_t t = Tc / T
  cdef DTYPE_t a = rhoc / rho_mix
  cdef DTYPE_t b = -yw
  # Define polynomial coefficients
  cdef DTYPE_t c2 = a
  cdef DTYPE_t c1 = (1.0 / rho_mix * (K - p_m0) - K * (1.0 - yw) / rho_m0) \
    / (R * T) - yw
  cdef DTYPE_t c0 = -yw / (rhoc * R * T) * (K - p_m0)
  ''' Return value of scaled pressure difference and slope '''
  cdef Pair out = fused_phir_d_phir_dd(d,t)
  cdef DTYPE_t phir_d = out.first
  cdef DTYPE_t phir_dd = out.second
  cdef DTYPE_t val = (a*d + b)*d*d*phir_d + ((c2 * d) + c1)*d + c0
  cdef DTYPE_t slope = (3*a*d + 2*b)*d*phir_d \
    + (a*d + b)*d*d*phir_dd \
    + 2*c2*d + c1
  return Pair(val, slope)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef Pair pure_phase_newton_pair_WLMA(DTYPE_t d, DTYPE_t T, DTYPE_t rho_mix,
  DTYPE_t yw, DTYPE_t ya, DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0,
  DTYPE_t R_a, DTYPE_t gamma_a):
  ''' Returns Pair<value, slope> for Newton iteration of the pure phase volume-
  pressure equilibrium condition. '''
  cdef Pair out = fused_phir_d_phir_dd(d, Tc / T)
  cdef DTYPE_t phir_d = out.first
  cdef DTYPE_t phir_dd = out.second
  cdef DTYPE_t ym = 1.0 - yw - ya
  cdef DTYPE_t Z = 1.0 + d * phir_d
  cdef DTYPE_t Z_d = phir_d + d * phir_dd
  cdef DTYPE_t val = rhoc / rho_mix * d * d * Z \
    - (yw * Z + ya * R_a / R) * d \
    + (-(-K + p_m0) / rho_mix - K / rho_m0 * ym) / (R * T) * d \
    - (K - p_m0) / (rhoc * R * T) * (yw + ya * R_a / R / Z)
  cdef DTYPE_t slope = rhoc / rho_mix * (2.0 * d * Z + d * d * Z_d) \
    - (yw * Z + ya * R_a / R) - yw * d * Z_d \
    + (-(-K + p_m0) / rho_mix - K / rho_m0 * ym) / (R * T) \
    - (K - p_m0) / (rhoc * R * T) * (-ya * R_a / R / (Z * Z) * Z_d)
  '''cdef DTYPE_t val = rhoc / rho_mix * d * d * Z * Z \
    - (yw * Z + ya * R_a / R) * d * Z \
    + (-(-K + p_m0) / rho_mix - K / rho_m0 * ym) / (R * T) * d * Z \
    - (K - p_m0) / (rhoc * R * T) * (yw * Z + ya * R_a / R)
  cdef DTYPE_t slope = rhoc / rho_mix * 2.0 * d * Z * (Z + d * Z_d) \
    - (yw * Z + ya * R_a / R) * (Z + d * Z_d) - yw * d * Z * Z_d \
    + (-(-K + p_m0) / rho_mix - K / rho_m0 * ym) / (R * T) * (Z + d * Z_d) \
    - (K - p_m0) / (rhoc * R * T) * (yw * Z_d)'''
  return Pair(val, slope)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t linear_energy_model(DTYPE_t T, DTYPE_t yw, DTYPE_t ym,
  DTYPE_t c_v_m0) noexcept:
  ''' Linear pressure-independent energy model '''
  return yw * (c_v_w0 * (T - T_w0) + e_w0) + ym * (c_v_m0 * T)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t rho_l_pt(DTYPE_t p, DTYPE_t T):
  ''' Compute liquid-phase density from pressure, temperature. '''
  cdef SatTriple sat_triple = prho_sat(T)
  # Force liquid representation
  if p <= sat_triple.psat:
    return sat_triple.rho_satl
  cdef DTYPE_t d = sat_triple.rho_satl / rhoc
  cdef DTYPE_t f, df, step
  cdef Pair out_pair
  # Newton's method for pressure
  for i in range(12):
    # Compute (phir_d, phir_dd)
    out_pair = fused_phir_d_phir_dd(d, Tc/T)
    f = d + d * d * out_pair.first - p / (rhoc * R * T)
    df = 1.0 + 2.0 * d * out_pair.first + d * d * out_pair.second
    step = -f/df
    d += step
    if step*step < 1e-16:
      break
  return d*rhoc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef TriplerhopT conservative_to_pT_WLM(DTYPE_t vol_energy, DTYPE_t rho_mix,
  DTYPE_t yw, DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0, DTYPE_t c_v_m0):
  ''' Map conservative WLM mixture variables to pressure, temperature.
  Production version. '''
  cdef DTYPE_t ym = 1.0 - yw
  cdef DTYPE_t v_mix = 1.0/rho_mix
  cdef DTYPE_t d = 1.0
  cdef DTYPE_t pmix, T, t, rhow, x, dT
  cdef DTYPE_t psat, rho_satl, rho_satv, d0, d1
  cdef Pair out_pair
  cdef bint is_supercritical, is_mixed
  cdef SatTriple sat_triple
  cdef Derivatives_phir_0_1_2 _phirall_0, _phirall_1
  cdef Derivatives_phi0_0_1_2 _phi0all_0, _phi0all_1
  cdef Derivatives_phir_0_1_2 _phirall
  cdef DTYPE_t dG0dt, dG1dt, dG0dd0, dG1dd0, dG0dd1, dG1dd1, _det, dd0dt, dd1dt
  cdef DTYPE_t dv0dT, dv1dT, du0dT, du1dT, v, vl, vv, dvdT, dT_to_v_bdry
  cdef DTYPE_t partial_dxdT, partial_dxdv, dvdp, dpsatdT, dxdT 
  cdef DTYPE_t uv, ul, dewdT, dedT, demdT, curr_energy, _c_v_w, _u
  cdef DTYPE_t _c1, Z, Z_d, d1psi, d2psi, drhowdT, rhom, drhomdT
  # Root-finding parameters
  cdef unsigned int i = 0
  cdef unsigned int MAX_NEWTON_ITERS = 64
  cdef DTYPE_t trust_region_size = 1e8
  # TODO: Check validity of inputs (lower and upper bounds)

  ''' Estimate initial temperature T_init '''
  if rho_mix < 10.0:
    # Due to an issue with flip-flopping for a low-density gas:
    T = ((vol_energy / rho_mix) - yw * (e_gas_ref - c_v_gas_ref * T_gas_ref)) \
    / (yw * c_v_gas_ref + ym * c_v_m0)
  else:
    # Estimate temperature
    T = ((vol_energy / rho_mix) - yw * (e_w0 - c_v_w0 * T_w0)) \
      / (yw * c_v_w0 + ym * c_v_m0)
    # if yw >= 0.2:
    #   # Refine estimate using a posteriori approximation of residual
    #   T += poly_T_residual_est(yw, vol_energy, rho_mix)
  ''' Estimate d(T_c) '''
  # One-step Newton approximation of critical-temperature value
  d = 1.0
  out_pair = pure_phase_newton_pair(d, Tc, rho_mix, yw, K, p_m0, rho_m0)
  d -= out_pair.first/out_pair.second
  ''' Estimate supercriticality based on energy at Tc '''
  # Quantify approximately supercriticality (rho is approximate, and magma
  #   mechanical energy is neglected)
  is_supercritical = yw * u(d*rhoc, Tc) \
    + ym * (c_v_m0 * Tc) < vol_energy/rho_mix
  ''' Clip T_init based on supercriticality estimate '''
  # Clip initial temperature guess to above or below supercritical
  if is_supercritical and T < Tc + 1.0:
    T = Tc + 1.0
  elif not is_supercritical and T > Tc - 1.0:
    T = Tc - 1.0

  ''' Cold-start compute pressure, water phasic density '''
  if not is_supercritical:
    # Compute saturation properties
    sat_triple = prho_sat(T)
    psat, rho_satl, rho_satv = \
      sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
    # Compute tentative density value
    rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0))
    if rho_satv <= rhow and rhow <= rho_satl:
      # start_case = "start-LV"
      # Accept tentative mixed-phase value of density
      d = rhow / rhoc
      # Compute pressure from sat vapor side
      pmix = psat
    else:
      # start_case = "start-subcrit"
      # Select pure vapor or pure liquid phase density
      d = rho_satv/rhoc if rhow < rho_satv else rho_satl/rhoc
      # Refine estimate of d from one Newton iteration
      out_pair = pure_phase_newton_pair(d, T, rho_mix, yw, K, p_m0, rho_m0)
      d -= out_pair.first/out_pair.second
      # Compute pressure
      pmix = p(d*rhoc, T)
  else:
    # start_case = "start-supercrit"
    # Refine estimate of d from one Newton iteration
    out_pair = pure_phase_newton_pair(d, T, rho_mix, yw, K, p_m0, rho_m0)
    d -= out_pair.first/out_pair.second
    # Compute pressure
    pmix = p(d*rhoc, T)
  # Clip temperatures below the triple point temperature
  if T < 273.16:
    T = 273.16
    
  ''' Perform iteration for total energy condition '''
  for i in range(MAX_NEWTON_ITERS):
    t = Tc/T
    is_mixed = False
    # Set default case string
    # case_str = "iter-supercrit"
    
    if T < Tc:
      # Check saturation curves
      sat_triple = prho_sat(T)
      psat, rho_satl, rho_satv = \
        sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
      # Compute volume-sum constrained density value if pm = psat
      rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0))
      if rho_satv <= rhow and rhow <= rho_satl:
        # case_str = "iter-LV"
        is_mixed = True
        # Accept tentative mixed-phase value of pressure, density
        pmix = psat
        d = rhow / rhoc
        # Compute steam fraction (fraction vapor mass per total water mass)
        x = (1.0 / rhow - 1.0 / rho_satl) / (1.0 / rho_satv - 1.0 / rho_satl)
        # Compute temperature update using saturation relation
        d0 = rho_satl/rhoc
        d1 = rho_satv/rhoc
        _phirall_0 = fused_phir_all(d0, Tc/T)
        _phirall_1 = fused_phir_all(d1, Tc/T)
        _phi0all_0 = fused_phi0_all(d0, Tc/T)
        _phi0all_1 = fused_phi0_all(d1, Tc/T)
        ''' Derivatives along Maxwell-constr. level set G(t, d0, d1) = 0 '''
        # Vector components of partial dG/dt = (dG0/dt; dG1/dt)
        dG0dt = d1*_phirall_1.phir_dt + _phirall_1.phir_t + _phi0all_1.phi0_t \
          - d0*_phirall_0.phir_dt - _phirall_0.phir_t - _phi0all_0.phi0_t
        dG1dt = d0*d0*_phirall_0.phir_dt - d1*d1*_phirall_1.phir_dt
        # Vector components of partial dG/d0
        dG0dd0 = -2.0*_phirall_0.phir_d \
          - d0*_phirall_0.phir_dd - _phi0all_0.phi0_d
        dG1dd0 = 1.0 + 2.0 * d0 * _phirall_0.phir_d + d0*d0*_phirall_0.phir_dd
        # Vector components of partial dG/d1
        dG0dd1 = 2.0*_phirall_1.phir_d \
          + d1*_phirall_1.phir_dd + _phi0all_1.phi0_d
        dG1dd1 = -1.0 - 2.0 * d1 * _phirall_1.phir_d - d1*d1*_phirall_1.phir_dd
        # Compute d(d0)/dt and d(d1)/dt constrainted to level set of G
        _det = dG0dd0 * dG1dd1 - dG0dd1 * dG1dd0
        dd0dt = -(dG1dd1 * dG0dt - dG0dd1 * dG1dt) / _det
        dd1dt = -(-dG1dd0 * dG0dt + dG0dd0 * dG1dt) / _det
        # Construct derivatives of volume:
        #   dv0/dt = partial(v0, t) + partial(v0, d) * dd0/dt
        #   dv0/dT = dv0/dt * dt/dT
        dv0dT = -1.0/(rhoc * d0 * d0) * dd0dt * (-Tc/(T*T))
        dv1dT = -1.0/(rhoc * d1 * d1) * dd1dt * (-Tc/(T*T))
        # Construct derivatives of internal energy
        du0dT = R * Tc * (
          (_phirall_0.phir_dt + _phi0all_0.phi0_dt) * dd0dt
          + (_phirall_0.phir_tt + _phi0all_0.phi0_tt)
        ) * (-Tc/(T*T))
        du1dT = R * Tc * (
          (_phirall_1.phir_dt + _phi0all_1.phi0_dt) * dd1dt
          + (_phirall_1.phir_tt + _phi0all_1.phi0_tt)
        ) * (-Tc/(T*T))
        # Construct dx/dT (change in steam fraction subject to mixture
        #   conditions) as partial(x, T) + partial(x, v) * dv/dT
        v = 1.0/(d*rhoc)
        vl = 1.0/(d0*rhoc)
        vv = 1.0/(d1*rhoc)
        partial_dxdT = ((v - vv) * dv0dT - (v - vl) * dv1dT) \
          / ((vv - vl) * (vv - vl))
        partial_dxdv =  1.0 / (vv - vl) # add partial (x, v)
        dvdp = (1.0 - yw) / yw * K / rho_m0 / (
          (psat + K - p_m0)*(psat + K - p_m0))
        # Compute saturation-pressure-temperature slope
        dpsatdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
          * R * (log(rho_satv / rho_satl) + _phirall_1.phir - _phirall_0.phir \
            - t * (_phirall_1.phir_t - _phirall_0.phir_t))
        dxdT = partial_dxdT + partial_dxdv * dvdp * dpsatdT
        # Construct de/dT for water:
        #   de/dT = partial(e,x) * dx/dT + de/dT
        uv = R * Tc * (_phirall_1.phir_t + _phi0all_1.phi0_t)
        ul = R * Tc * (_phirall_0.phir_t + _phi0all_0.phi0_t)
        dewdT = (uv - ul) * dxdT + x*du1dT + (1.0-x)*du0dT
        # Construct de/dT for mixture, neglecting pdv work of magma
        dedT = yw * dewdT + ym * c_v_m0

        ''' Compute Newton step for temperature '''
        curr_energy = yw * (x * uv + (1.0 - x) * ul) \
          + ym * (c_v_m0 * T + magma_mech_energy(psat, K, p_m0, rho_m0))
        dT = -(curr_energy - vol_energy/rho_mix)/dedT
        if rho_mix < 10: # Size limiter for sparse gas
          trust_region_size = 1e2
          dT = -(curr_energy - vol_energy/rho_mix)/(dedT + 1e6/1e2)

        ''' Estimate saturation state in destination temperature '''
        # Vapor-boundary overshooting correction
        dvdT = (vv - vl) * dxdT + x*dv1dT + (1.0-x)*dv0dT
        dT_to_v_bdry = (vv - v) / (dvdT - dv1dT)
        if dT > 0 and dT_to_v_bdry > 0 and dT > dT_to_v_bdry:
          dT = 0.5 * (dT + dT_to_v_bdry)
        # Newton-step temperature
        T += dT
      else:
        # Annotate subcritical but pure-phase case
        # case_str = "iter-subcrit"
        # Update d based on computed rhow value
        # if rhow > 0:
        #   d = rhow / rhoc
        pass
    if not is_mixed:
      # Compute phasic states using current T and previous iteration d
      rhow = d*rhoc
      rhom = ym / (1.0 / rho_mix - yw / rhow)
      # Evaluate phir derivatives (cost-critical)
      _phirall = fused_phir_all(d, Tc/T)
      # Compute pure-phase heat capacity
      _c_v_w = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
      # Compute pure-phase energy
      _u = t * R * T * (_phirall.phir_t + phi0_t(d, t))

      ''' Compute slopes '''
      # Compute intermediates
      _c1 = (v_mix * d - yw / rhoc)
      # Compute compressibility, partial of compressibility
      Z = 1.0 + d * _phirall.phir_d
      Z_d = _phirall.phir_d + d * _phirall.phir_dd
      # Compute derivatives of pressure equilibrium level set function
      #  R * T *psi(d, T) = 0 constrains d(t)
      #  Here we use the form of equation with T appearing only once
      #  d1psi is d/dd and d2psi is d/dT of (R * T * psi(d,T)).
      d1psi = v_mix * (Z * d * rhoc * R * T + K - p_m0) - K * ym / rho_m0 \
        + _c1 * (rhoc * R * T * (Z + d * Z_d))
      d2psi = _c1 * Z * d * rhoc * R
      # Compute density-temperature slope under the volume addition constraint
      drhowdT = -d2psi / d1psi * rhoc
      # Compute density-temperature slope for m state
      drhomdT = -yw / ym * (rhom / rhow) * (rhom / rhow) * drhowdT
      # Compute water energy-temperature slope
      dewdT = _c_v_w + R * Tc / rhoc * _phirall.phir_dt * drhowdT
      # Compute magma energy-temperature slope
      demdT = c_v_m0 \
        + pmix / (rhom*rhom) * drhomdT
      # Compute mixture energy-temperature slope
      dedT = yw * dewdT + ym * demdT
      ''' Compute Newton step for temperature '''
      curr_energy = yw * _u \
        + ym * (c_v_m0 * T + magma_mech_energy(pmix, K, p_m0, rho_m0))
      # Temperature migration
      dT = -(curr_energy - vol_energy/rho_mix)/dedT
      # Limit to dT <= 100
      dT *= min(1.0, 50.0/(1e-16+fabs(dT)))
      # Newton-step temperature
      T += dT
      # Newton-step d (cross term in Jacobian)
      d += drhowdT/rhoc * dT
      out_pair = pure_phase_newton_pair(d, T, rho_mix, yw, K, p_m0, rho_m0)
      d -= out_pair.first/out_pair.second
      # Update pressure
      pmix = p(d*rhoc, T)
      # Update water phasic density
      rhow = d * rhoc
    # Clip temperatures below the triple point temperature
    T = max(T, 273.16)
    if dT * dT < 1e-9 * 1e-9:
      break
  return TriplerhopT(rhow, pmix, T)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef TriplerhopT conservative_to_pT_WLMA(DTYPE_t vol_energy, DTYPE_t rho_mix,
  DTYPE_t yw, DTYPE_t ya, DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0,
  DTYPE_t c_v_m0, DTYPE_t R_a, DTYPE_t gamma_a):
  ''' Map conservative WLMA mixture variables to pressure, temperature.
  Production version. WLMA model = {water, linearized magma, air}. '''
  cdef DTYPE_t ym = 1.0 - yw - ya
  cdef DTYPE_t v_mix = 1.0/rho_mix
  cdef DTYPE_t d = 1.0
  cdef DTYPE_t pmix, T, t, rhow, x, dT
  cdef DTYPE_t psat, rho_satl, rho_satv, d0, d1
  cdef Pair out_pair
  cdef bint is_supercritical, is_mixed
  cdef SatTriple sat_triple
  cdef Derivatives_phir_0_1_2 _phirall_0, _phirall_1
  cdef Derivatives_phi0_0_1_2 _phi0all_0, _phi0all_1
  cdef Derivatives_phir_0_1_2 _phirall
  cdef DTYPE_t dG0dt, dG1dt, dG0dd0, dG1dd0, dG0dd1, dG1dd1, _det, dd0dt, dd1dt
  cdef DTYPE_t dv0dT, dv1dT, du0dT, du1dT, v, vl, vv, dvdT, dT_to_v_bdry
  cdef DTYPE_t partial_dxdT, partial_dxdv, dvdp, dpsatdT, dxdT 
  cdef DTYPE_t uv, ul, dewdT, dedT, demdT, curr_energy, _c_v_w, _u
  cdef DTYPE_t _c1, Z, Z_d, Z_T, d1psi, d2psi, drhowdT, rhom, drhomdT, drhomadT
  cdef DTYPE_t d_inner_step
  cdef DTYPE_t c_v_a = R_a / (gamma_a - 1.0)
  # Root-finding parameters
  cdef unsigned int i = 0
  cdef unsigned int MAX_NEWTON_ITERS = 64
  cdef DTYPE_t trust_region_size = 1e8
  # TODO: Check validity of inputs (lower and upper bounds)

  ''' Estimate initial temperature T_init '''
  if rho_mix < 10.0:
    # Due to an issue with flip-flopping for a low-density gas:
    T = ((vol_energy / rho_mix) - yw * (e_gas_ref - c_v_gas_ref * T_gas_ref)) \
    / (yw * c_v_gas_ref + ym * c_v_m0 + ya * c_v_a)
  else:
    # Estimate temperature
    T = ((vol_energy / rho_mix) - yw * (e_w0 - c_v_w0 * T_w0)) \
      / (yw * c_v_w0 + ym * c_v_m0 + ya * c_v_a)
    # if yw >= 0.2:
    #   # Refine estimate using a posteriori approximation of residual
    #   T += poly_T_residual_est(yw, vol_energy, rho_mix)
  ''' Estimate d(T_c) '''
  # One-step Newton approximation of critical-temperature value
  d = 1.0
  out_pair = pure_phase_newton_pair_WLMA(d, Tc, rho_mix, yw, ya, K, p_m0,
    rho_m0, R_a, gamma_a)
  d -= out_pair.first/out_pair.second
  if d <= 0:
    # Recovery
    d = 3.0
    out_pair = pure_phase_newton_pair_WLMA(d, Tc, rho_mix, yw, ya, K, p_m0,
      rho_m0, R_a, gamma_a)
    d -= out_pair.first/out_pair.second
  ''' Estimate supercriticality based on energy at Tc '''
  # Quantify approximately supercriticality (rho is approximate, and magma
  #   mechanical energy is neglected)
  is_supercritical = yw * u(d*rhoc, Tc) \
    + ym * (c_v_m0 * Tc) + ya * (c_v_a * Tc) < vol_energy/rho_mix
  ''' Clip T_init based on supercriticality estimate '''
  # Clip initial temperature guess to above or below supercritical
  if is_supercritical and T < Tc + 1.0:
    T = Tc + 1.0
  elif not is_supercritical and T > Tc - 1.0:
    T = Tc - 1.0

  ''' Cold-start compute pressure, water phasic density '''
  if not is_supercritical:
    # Compute saturation properties
    sat_triple = prho_sat(T)
    psat, rho_satl, rho_satv = \
      sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
    # Compute tentative density value
    rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0) \
           - ya * R_a * T / psat)
    if rho_satv <= rhow and rhow <= rho_satl:
      # start_case = "start-LV"
      # Accept tentative mixed-phase value of density
      d = rhow / rhoc
      # Compute pressure from sat vapor side
      pmix = psat
    else:
      # start_case = "start-subcrit"
      # Select pure vapor or pure liquid phase density
      d = rho_satv/rhoc if (0 < rhow and rhow < rho_satv) else rho_satl/rhoc
      # Refine estimate of d from one Newton iteration
      out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
        rho_m0, R_a, gamma_a)
      d -= out_pair.first/out_pair.second
      # Compute pressure
      pmix = p(d*rhoc, T)
  else:
    # start_case = "start-supercrit"
    # Refine estimate of d from one Newton iteration
    out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
      rho_m0, R_a, gamma_a)
    d -= out_pair.first/out_pair.second
    # Compute pressure
    pmix = p(d*rhoc, T)
  # Clip temperatures below the triple point temperature
  if T < 273.16:
    T = 273.16
  if d <= 0:
    # Second recovery
    d = 3.0
  for i in range(20):
    out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
      rho_m0, R_a, gamma_a)
    d_inner_step = -out_pair.first/out_pair.second
    d += d_inner_step
    if d_inner_step * d_inner_step < 1e-4 * d * d or d < 0:
      break
  if d < 0:
    d = 2.0
  # Compute pressure
  pmix = p(d*rhoc, T)  

    
  ''' Perform iteration for total energy condition '''
  for i in range(MAX_NEWTON_ITERS):
    t = Tc/T
    is_mixed = False
    # Set default case string
    # case_str = "iter-supercrit"
    
    if T < Tc:
      # Check saturation curves
      sat_triple = prho_sat(T)
      psat, rho_satl, rho_satv = \
        sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
      # Compute volume-sum constrained density value if pm = psat
      rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0) \
           - ya * R_a * T / psat)
      if rho_satv <= rhow and rhow <= rho_satl:
        # case_str = "iter-LV"
        is_mixed = True
        # Accept tentative mixed-phase value of pressure, density
        pmix = psat
        d = rhow / rhoc
        # Compute steam fraction (fraction vapor mass per total water mass)
        x = (1.0 / rhow - 1.0 / rho_satl) / (1.0 / rho_satv - 1.0 / rho_satl)
        # Compute temperature update using saturation relation
        d0 = rho_satl/rhoc
        d1 = rho_satv/rhoc
        _phirall_0 = fused_phir_all(d0, Tc/T)
        _phirall_1 = fused_phir_all(d1, Tc/T)
        _phi0all_0 = fused_phi0_all(d0, Tc/T)
        _phi0all_1 = fused_phi0_all(d1, Tc/T)
        ''' Derivatives along Maxwell-constr. level set G(t, d0, d1) = 0 '''
        # Vector components of partial dG/dt = (dG0/dt; dG1/dt)
        dG0dt = d1*_phirall_1.phir_dt + _phirall_1.phir_t + _phi0all_1.phi0_t \
          - d0*_phirall_0.phir_dt - _phirall_0.phir_t - _phi0all_0.phi0_t
        dG1dt = d0*d0*_phirall_0.phir_dt - d1*d1*_phirall_1.phir_dt
        # Vector components of partial dG/d0
        dG0dd0 = -2.0*_phirall_0.phir_d \
          - d0*_phirall_0.phir_dd - _phi0all_0.phi0_d
        dG1dd0 = 1.0 + 2.0 * d0 * _phirall_0.phir_d + d0*d0*_phirall_0.phir_dd
        # Vector components of partial dG/d1
        dG0dd1 = 2.0*_phirall_1.phir_d \
          + d1*_phirall_1.phir_dd + _phi0all_1.phi0_d
        dG1dd1 = -1.0 - 2.0 * d1 * _phirall_1.phir_d - d1*d1*_phirall_1.phir_dd
        # Compute d(d0)/dt and d(d1)/dt constrainted to level set of G
        _det = dG0dd0 * dG1dd1 - dG0dd1 * dG1dd0
        dd0dt = -(dG1dd1 * dG0dt - dG0dd1 * dG1dt) / _det
        dd1dt = -(-dG1dd0 * dG0dt + dG0dd0 * dG1dt) / _det
        # Construct derivatives of volume:
        #   dv0/dt = partial(v0, t) + partial(v0, d) * dd0/dt
        #   dv0/dT = dv0/dt * dt/dT
        dv0dT = -1.0/(rhoc * d0 * d0) * dd0dt * (-Tc/(T*T))
        dv1dT = -1.0/(rhoc * d1 * d1) * dd1dt * (-Tc/(T*T))
        # Construct derivatives of internal energy
        du0dT = R * Tc * (
          (_phirall_0.phir_dt + _phi0all_0.phi0_dt) * dd0dt
          + (_phirall_0.phir_tt + _phi0all_0.phi0_tt)
        ) * (-Tc/(T*T))
        du1dT = R * Tc * (
          (_phirall_1.phir_dt + _phi0all_1.phi0_dt) * dd1dt
          + (_phirall_1.phir_tt + _phi0all_1.phi0_tt)
        ) * (-Tc/(T*T))
        # Construct dx/dT (change in steam fraction subject to mixture
        #   conditions) as partial(x, T) + partial(x, v) * dv/dT
        v  = 1.0/(d*rhoc)
        vl = 1.0/(d0*rhoc)
        vv = 1.0/(d1*rhoc)
        partial_dxdT = ((v - vv) * dv0dT - (v - vl) * dv1dT) \
          / ((vv - vl) * (vv - vl))
        partial_dxdv =  1.0 / (vv - vl) # add partial (x, v)
        # Partial of volume w.r.t. pressure due to volume sum condition
        dvdp = ym / yw * K / rho_m0 / ((psat + K - p_m0)*(psat + K - p_m0)) \
          + ya / yw * R_a * T / (psat*psat)
        # Compute change in allowable volume due to air presence (temp variable)
        #   This is a partial derivative of the volume sum condition
        dvdT = - ya / yw * (R_a / psat)
        # Compute saturation-pressure-temperature slope
        dpsatdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
          * R * (log(rho_satv / rho_satl) + _phirall_1.phir - _phirall_0.phir \
            - t * (_phirall_1.phir_t - _phirall_0.phir_t))
        dxdT = partial_dxdT + partial_dxdv * (dvdp * dpsatdT + dvdT)
        # Construct de/dT for water:
        #   de/dT = partial(e,x) * dx/dT + de/dT
        uv = R * Tc * (_phirall_1.phir_t + _phi0all_1.phi0_t)
        ul = R * Tc * (_phirall_0.phir_t + _phi0all_0.phi0_t)
        dewdT = (uv - ul) * dxdT + x*du1dT + (1.0-x)*du0dT
        # Construct de/dT for mixture, with -pdv of magma as (p * -dvdp * dpdT)
        dedT = yw * dewdT + ym * c_v_m0 + ya * c_v_a \
          + ym * (psat * K / rho_m0 / ((psat + K - p_m0)*(psat + K - p_m0))
            * dpsatdT)

        ''' Compute Newton step for temperature '''
        curr_energy = yw * (x * uv + (1.0 - x) * ul) \
          + ym * (c_v_m0 * T + magma_mech_energy(psat, K, p_m0, rho_m0)) \
          + ya * (c_v_a * T)
        dT = -(curr_energy - vol_energy/rho_mix)/dedT
        if rho_mix < 10: # Size limiter for sparse gas
          trust_region_size = 1e2
          dT = -(curr_energy - vol_energy/rho_mix)/(dedT + 1e6/1e2)

        ''' Estimate saturation state in destination temperature '''
        # Vapor-boundary overshooting correction
        dvdT = (vv - vl) * dxdT + x*dv1dT + (1.0-x)*dv0dT
        dT_to_v_bdry = (vv - v) / (dvdT - dv1dT)
        if dT > 0 and dT_to_v_bdry > 0 and dT > dT_to_v_bdry:
          dT = 0.5 * (dT + dT_to_v_bdry)
        # Newton-step temperature
        T += dT
      else:
        # Annotate subcritical but pure-phase case
        # case_str = "iter-subcrit"
        # Update d based on computed rhow value
        # d = rhow / rhoc
        pass
    if not is_mixed:
      # Compute phasic states using current d, T
      rhow = d*rhoc
      if d <= 0 and T < Tc:
        # Negative density recovery attempt
        sat_triple = prho_sat(T)
        d = sat_triple.rho_satl / rhoc
        rhow = sat_triple.rho_satl
      elif d <= 0:
        # Supercritical recovery attempt (cool mixture to reduce air volume)
        T = Tc - 1.0
        sat_triple = prho_sat(T)
        d = sat_triple.rho_satl / rhoc
        rhow = sat_triple.rho_satl
      rhom = ym / (1.0 / rho_mix - yw / rhow - ya * (R_a * T) / pmix)
      # Evaluate phir derivatives (cost-critical)
      _phirall = fused_phir_all(d, Tc/T)
      # Compute pure-phase heat capacity
      _c_v_w = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
      # Compute pure-phase energy
      _u = t * R * T * (_phirall.phir_t + phi0_t(d, t))

      ''' Compute slopes '''
      # Compute compressibility, partial of compressibility
      Z = 1.0 + d * _phirall.phir_d
      Z_d = _phirall.phir_d + d * _phirall.phir_dd
      Z_T = d * _phirall.phir_dt * (-Tc / (T*T))
      # Compute intermediates
      _c1 = (v_mix * d - (yw + ya*R_a/(Z *R)) / rhoc)
      # Compute derivatives of pressure equilibrium level set function 
      #  R * T *psi(d, T) = 0 constrains d(t).
      #  Note the difference with the Newton slope-pair function.
      #  Here we use the form of equation with T appearing only once
      #  d1psi is d/dd (nondim) and d2psi is d/dT (dim) of (R * T * psi(d,T)).
      d1psi = (v_mix + ya * R_a / R / rhoc * Z_d / (Z * Z)) \
        * (Z * d * rhoc * R * T + K - p_m0) - K * ym / rho_m0 \
        + _c1 * (rhoc * R * T * (Z + d * Z_d))
      d2psi = (ya * R_a / R / rhoc * Z_T / (Z * Z)) \
        * (Z * d * rhoc * R * T + K - p_m0) \
        + _c1 * (Z + T * Z_T) * d * rhoc * R
      # Compute density-temperature slope under the volume addition constraint
      drhowdT = -d2psi / d1psi * rhoc
      # Compute density-temperature slope for m state
      drhomdT = -(rhom*rhom/ ym) * (yw * (drhowdT / (rhow*rhow)) 
        + ya * (R_a * T / (pmix*pmix)) * (
          - pmix / T  +
          rhoc * R * ((1.0 + d * _phirall.phir_d) * d 
            - d * d * _phirall.phir_dt * Tc / T)
          + (1.0 + 2.0 * d * _phirall.phir_d + d * d * _phirall.phir_dd)
            * rhoc * R * T * drhowdT / rhoc))
      # Compute water energy-temperature slope # TODO: checked -> debug,mirror
      dewdT = _c_v_w + R * Tc / rhoc * _phirall.phir_dt * drhowdT
      # Compute magma energy-temperature slope (c_v dT + de/dv * dv, v = v(T))
      demdT = c_v_m0 \
        + pmix / (rhom*rhom) * drhomdT # - p dv = p (drho) / rho^2
      # Compute mixture energy-temperature slope
      dedT = yw * dewdT + ym * demdT + ya * c_v_a
      ''' Compute Newton step for temperature '''
      curr_energy = yw * _u \
        + ym * (c_v_m0 * T + magma_mech_energy(pmix, K, p_m0, rho_m0)) \
        + ya * c_v_a * T
      # Temperature migration
      dT = -(curr_energy - vol_energy/rho_mix)/dedT
      # Limit to dT <= 100
      dT *= min(1.0, 50.0/(1e-16+fabs(dT))) # TODO: generalize using trustregion
      # Newton-step temperature
      T += dT
      # Newton-step d (cross term in Jacobian)
      d += drhowdT/rhoc * dT
      out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
        rho_m0, R_a, gamma_a)
      d_inner_step = -out_pair.first/out_pair.second
      d += d_inner_step
      '''if fabs(d_inner_step) > 0.01*d:
        # Restep
        out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
          rho_m0, R_a, gamma_a)
        d_inner_step = -out_pair.first/out_pair.second
        d += d_inner_step'''
      # Update pressure
      pmix = p(d*rhoc, T)
      # Update water phasic density
      rhow = d * rhoc
    # Clip temperatures below the triple point temperature
    T = max(T, 273.16)
    if dT * dT < 1e-9 * 1e-9:
      break
  return TriplerhopT(rhow, pmix, T)

''' Isolated equation sets '''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef kernel4_WLMA(DTYPE_t d, DTYPE_t T, DTYPE_t pr,  DTYPE_t Tr,
  DTYPE_t vol_energy, DTYPE_t rho_mix, DTYPE_t yw, DTYPE_t ya, DTYPE_t K,
  DTYPE_t p_m0, DTYPE_t rho_m0, DTYPE_t c_v_m0, DTYPE_t R_a, DTYPE_t gamma_a):
  ''' Return compatibility equations and derivatives. '''
  cdef DTYPE_t ym = 1.0 - yw - ya
  cdef DTYPE_t v_mix = 1.0/rho_mix
  cdef DTYPE_t pmix, t, rhow, x, dT
  cdef DTYPE_t psat, rho_satl, rho_satv, d0, d1, e_mix
  cdef Pair out_pair
  cdef bint is_supercritical, is_mixed
  cdef SatTriple sat_triple
  cdef Derivatives_phir_0_1_2 _phirall_0, _phirall_1
  cdef Derivatives_phi0_0_1_2 _phi0all_0, _phi0all_1
  cdef Derivatives_phir_0_1_2 _phirall
  cdef DTYPE_t dG0dt, dG1dt, dG0dd0, dG1dd0, dG0dd1, dG1dd1, _det, dd0dt, dd1dt
  cdef DTYPE_t dv0dT, dv1dT, du0dT, du1dT, v, vl, vv, dvdT, dT_to_v_bdry
  cdef DTYPE_t partial_dxdT, partial_dxdv, dvdp, dpsatdT, dxdT 
  cdef DTYPE_t uv, ul, dewdT, dedT, demdT, curr_energy, _c_v_w, _u
  cdef DTYPE_t _c1, Z, Z_d, Z_T, d1psi, d2psi, drhowdT, rhom, drhomdT, drhomadT
  cdef DTYPE_t d_inner_step
  cdef DTYPE_t c_v_a = R_a / (gamma_a - 1.0)

  e_mix = vol_energy/rho_mix
  rhow = rhoc * d
  t = Tc / T
  
  ''' The full relaxation system:
  let ef_w = y_w * e_w/e_mix: energy fraction of water
  let ef_r = y_r * e_r/e_mix: energy fraction of residual components

  ef_w + ef_r - 1 == 0
  yw/vmix / (rhoc*d) + yr/vmix * vr(pr, Tr) - 1 == 0
  p - pr == 0
  T - Tr == 0

  with Jacobian w.r.t [d, Tw, pr, Tr]

  [d(ef_w)dd, d(ef_w)d(Tw), d(ef_r)d(pr), d(ef_r)d(Tr)]
  [-yw/(vmix*rhoc*d*d), 0,  yr/vmix*d(v_r)/d(pr), yr/vmix * d(v_r)/d(Tr)]
  [d(p_w)dd, d(p_w)/dT, -1, 0]
  [0, 1, 0, -1]
  '''

  ''' Allocate residual vector '''
  f = np.ones((4,))
  f[3] = T - Tr

  ''' Compute coefficient matrix C, derivative matrix D such that elementwise
  J_ij = C_ij D_ij, C is dependent only on yw, yr, e_mix, v_mix, and D_ij
  disjointly dependent on everything else. '''
  C = np.ones((4,4))
  D = np.ones((4,4))
  yr = 1.0 - yw
  C[0,0] = yw/e_mix
  C[0,1] = yw/e_mix
  C[0,2] = yr/e_mix
  C[0,3] = yr/e_mix
  C[1,0] = yw/v_mix
  C[1,2] = yr/v_mix
  C[1,3] = yr/v_mix

  # dew/dd
  # dew/dT
  # der/dpr
  # der/Tr
  D[1,0] = -1.0/(d*d*rhoc)
  D[1,1] = 0.0
  # dvr/dpr
  # dvr/Tr
  D[2,2] = -1.0
  D[2,3] = 0.0
  D[3,0] = 0.0
  D[3,1] = 1.0
  D[3,2] = 0.0
  D[3,3] = -1.0

  ''' Compute residual components. '''
  rhom = rho_m0 * (1.0 + (pr - p_m0) / K)
  derdpr = ym / (ym + ya) * pr / (rhom*rhom) * rho_m0 / K
  derdTr = (ym * c_v_m0 + ya * R_a / (gamma_a - 1.0)) / (ym + ya)
  dvrdpr = - (ym * rho_m0 / K / (rhom * rhom) + ya * R_a * Tr / (pr * pr)) \
    / (ym + ya)
  dvrdTr = ya / (ym + ya) * R_a / pr

  D[0,2] = derdpr
  D[0,3] = derdTr
  D[1,2] = dvrdpr
  D[1,3] = dvrdTr

  ''' Compute saturation hypothetical at exact pressure. '''

  if T < Tc:
    # Compute saturation curves
    sat_triple = prho_sat(T)
    psat, rho_satl, rho_satv = \
      sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv

    pmix = psat
    # Compute steam fraction (fraction vapor mass per total water mass)
    x = (1.0 / rhow - 1.0 / rho_satl) / (1.0 / rho_satv - 1.0 / rho_satl)
    # Compute temperature update using saturation relation
    d0 = rho_satl/rhoc
    d1 = rho_satv/rhoc
    _phirall_0 = fused_phir_all(d0, Tc/T)
    _phirall_1 = fused_phir_all(d1, Tc/T)
    _phi0all_0 = fused_phi0_all(d0, Tc/T)
    _phi0all_1 = fused_phi0_all(d1, Tc/T)
    ''' Derivatives along Maxwell-constr. level set G(t, d0, d1) = 0 '''
    # Vector components of partial dG/dt = (dG0/dt; dG1/dt)
    dG0dt = d1*_phirall_1.phir_dt + _phirall_1.phir_t + _phi0all_1.phi0_t \
      - d0*_phirall_0.phir_dt - _phirall_0.phir_t - _phi0all_0.phi0_t
    dG1dt = d0*d0*_phirall_0.phir_dt - d1*d1*_phirall_1.phir_dt
    # Vector components of partial dG/d0
    dG0dd0 = -2.0*_phirall_0.phir_d \
      - d0*_phirall_0.phir_dd - _phi0all_0.phi0_d
    dG1dd0 = 1.0 + 2.0 * d0 * _phirall_0.phir_d + d0*d0*_phirall_0.phir_dd
    # Vector components of partial dG/d1
    dG0dd1 = 2.0*_phirall_1.phir_d \
      + d1*_phirall_1.phir_dd + _phi0all_1.phi0_d
    dG1dd1 = -1.0 - 2.0 * d1 * _phirall_1.phir_d - d1*d1*_phirall_1.phir_dd
    # Compute d(d0)/dt and d(d1)/dt constrainted to level set of G
    _det = dG0dd0 * dG1dd1 - dG0dd1 * dG1dd0
    dd0dt = -(dG1dd1 * dG0dt - dG0dd1 * dG1dt) / _det
    dd1dt = -(-dG1dd0 * dG0dt + dG0dd0 * dG1dt) / _det
    # Construct derivatives of volume:
    #   dv0/dt = partial(v0, t) + partial(v0, d) * dd0/dt
    #   dv0/dT = dv0/dt * dt/dT
    dv0dT = -1.0/(rhoc * d0 * d0) * dd0dt * (-Tc/(T*T))
    dv1dT = -1.0/(rhoc * d1 * d1) * dd1dt * (-Tc/(T*T))
    # Construct derivatives of internal energy
    du0dT = R * Tc * (
      (_phirall_0.phir_dt + _phi0all_0.phi0_dt) * dd0dt
      + (_phirall_0.phir_tt + _phi0all_0.phi0_tt)
    ) * (-Tc/(T*T))
    du1dT = R * Tc * (
      (_phirall_1.phir_dt + _phi0all_1.phi0_dt) * dd1dt
      + (_phirall_1.phir_tt + _phi0all_1.phi0_tt)
    ) * (-Tc/(T*T))
    # Construct dx/dT (change in steam fraction subject to mixture
    #   conditions) as partial(x, T) + partial(x, v) * dv/dT
    v  = 1.0/(d*rhoc)
    vl = 1.0/(d0*rhoc)
    vv = 1.0/(d1*rhoc)

    # Compute partials
    partial_dxdT = ((v - vv) * dv0dT - (v - vl) * dv1dT) \
      / ((vv - vl) * (vv - vl))
    partial_dxdv =  1.0 / (vv - vl)
    partial_dxdd = -partial_dxdv / (d * d * rhoc)
    # Compute saturation-pressure-temperature slope
    dpsatdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
      * R * (log(rho_satv / rho_satl) + _phirall_1.phir - _phirall_0.phir \
        - t * (_phirall_1.phir_t - _phirall_0.phir_t))
    uv = R * Tc * (_phirall_1.phir_t + _phi0all_1.phi0_t)
    ul = R * Tc * (_phirall_0.phir_t + _phi0all_0.phi0_t)

    ew = x * uv + (1.0 - x) * ul
    dewdT = (uv - ul) * partial_dxdT + x*du1dT + (1.0-x)*du0dT
    dewdd = (uv - ul) * partial_dxdd

    lv_hypothetical = {
      "x": x,
      "ew": ew,
      "v": v,
      "ul": ul,
      "uv": uv,
      "vl": vl,
      "vv": vv,
      "dewdT": dewdT,
      "dpsatdT": dpsatdT,
      "partial_dxdT": partial_dxdT,
      "partial_dxdv": partial_dxdv,
      "partial_dxdd": partial_dxdd,
      "dv0dT": dv0dT,
      "dv1dT": dv1dT,
      "du0dT": du0dT,
      "du1dT": du1dT,
    }
    
    # Vapor-boundary overshooting correction
    # dvdT = (vv - vl) * dxdT + x*dv1dT + (1.0-x)*dv0dT
    # dT_to_v_bdry = (vv - v) / (dvdT - dv1dT)
  else:
    lv_hypothetical = {}

  ''' Compute dew/dd, dew/dT '''

  if T < Tc:
    if rho_satv < d * rhoc < rho_satl:
      # Load LV hypotheticals
      dewdd = dewdd
      dewdT = dewdT
      ew = ew
      p = psat
      dpdd = 0.0
      dpdT = dpsatdT
    else:
      # Pure phase
      _phirall = fused_phir_all(d, t)
      dewdd = R * Tc * _phirall.phir_dt
      ew = R * Tc * (_phirall.phir_t + phi0_t(d, t))
      dewdT = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
      p = d * rhoc * R * T * (1.0 + d * _phirall.phir_d)
      dpdd = rhoc * R * T * (1.0 + 2.0 * d * _phirall.phir_d
             + d * d * _phirall.phir_dd)
      dpdT = d * rhoc * R + d * d * rhoc * R * T * (
        _phirall.phir_dt * (-Tc/(T*T)))
  else:
    # Pure phase
    _phirall = fused_phir_all(d, t)
    dewdd = R * Tc * _phirall.phir_dt
    ew = R * Tc * (_phirall.phir_t + phi0_t(d, t))
    dewdT = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
    p = d * rhoc * R * T * (1.0 + d * _phirall.phir_d)
    dpdd = rhoc * R * T * (1.0 + 2.0 * d * _phirall.phir_d
             + d * d * _phirall.phir_dd)
    dpdT = d * rhoc * R + d * d * rhoc * R * T * (
        _phirall.phir_dt * (-Tc/(T*T)))

  # Complete fill-in
  em = c_v_m0 * Tr + magma_mech_energy(pr, K, p_m0, rho_m0)
  ea = R_a / (gamma_a - 1.0) * Tr
  f[0] = (yw * ew + ym * em + ya * ea)/ e_mix  - 1.0
  f[1] = (yw / (rhoc * d) \
        + ym / rhom \
        + ya * R_a * Tr / pr) / v_mix - 1.0
  f[2] = p - pr
  D[0,0] = dewdd
  D[0,1] = dewdT
  D[2,0] = dpdd
  D[2,1] = dpdT

  return f, C, D, lv_hypothetical, p

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def kernel2_WLMA_debug(DTYPE_t alpha_w, DTYPE_t T,
  DTYPE_t vol_energy, DTYPE_t rho_mix, DTYPE_t yw, DTYPE_t ya, DTYPE_t K,
  DTYPE_t p_m0, DTYPE_t rho_m0, DTYPE_t c_v_m0, DTYPE_t R_a, DTYPE_t gamma_a):
  ''' Return compatibility equations and derivatives. '''
  cdef DTYPE_t ym = 1.0 - yw - ya
  cdef DTYPE_t v_mix = 1.0/rho_mix
  cdef DTYPE_t pmix, t, rhow, x, dT
  cdef DTYPE_t psat, rho_satl, rho_satv, d0, d1, e_mix
  cdef Pair out_pair
  cdef bint is_supercritical, is_mixed
  cdef SatTriple sat_triple
  cdef Derivatives_phir_0_1_2 _phirall_0, _phirall_1
  cdef Derivatives_phi0_0_1_2 _phi0all_0, _phi0all_1
  cdef Derivatives_phir_0_1_2 _phirall
  cdef DTYPE_t dG0dt, dG1dt, dG0dd0, dG1dd0, dG0dd1, dG1dd1, _det, dd0dt, dd1dt
  cdef DTYPE_t dv0dT, dv1dT, du0dT, du1dT, v, vl, vv, dvdT, dT_to_v_bdry
  cdef DTYPE_t partial_dxdT, partial_dxdv, dvdp, dpsatdT, dxdT 
  cdef DTYPE_t uv, ul, dewdT, dedT, demdT, curr_energy, _c_v_w, _u
  cdef DTYPE_t _c1, Z, Z_d, Z_T, d1psi, d2psi, drhowdT, rhom, drhomdT, drhomadT
  cdef DTYPE_t d_inner_step
  cdef DTYPE_t c_v_a = R_a / (gamma_a - 1.0)
  # Enumerate flag types
  cdef int _FLAG_L = 0
  cdef int _FLAG_LV = 1
  cdef int _FLAG_V = 2
  cdef int _FLAG_C = 3

  ''' The energy-volume system:
  let ef_w = y_w * e_w/e_mix: energy fraction of water
  let ef_r = y_r * e_r/e_mix: energy fraction of residual components

  ef_w + ef_r - 1 == 0
  yw/vmix / (rhoc*d) + yr/vmix * vr(pr, Tr) - 1 == 0

  with Jacobian w.r.t [alpha_w, T].
  '''
  # Compute dependents
  t = Tc / T
  ym = 1.0 - yw - ya
  # Compute water phase density as dependent
  rhow = rho_mix * yw / alpha_w
  # Set bound on rhow based on pressure bound [, 1 GPa]
  rhow = np.clip(rhow, 1e-9, 1260)
  d = rhow / rhoc

  ''' Allocate matrices '''
  f = np.ones((2,))
  J = np.ones((2,2))

  ''' Compute saturation hypothetical at exact pressure. '''
  if T < Tc:
    # Compute saturation curves
    sat_triple = prho_sat(T)
    psat, rho_satl, rho_satv = \
      sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
    # Compute steam fraction (fraction vapor mass per total water mass)
    x = (1.0 / rhow - 1.0 / rho_satl) / (1.0 / rho_satv - 1.0 / rho_satl)
    # Compute temperature update using saturation relation
    d0 = rho_satl/rhoc
    d1 = rho_satv/rhoc
    _phirall_0 = fused_phir_all(d0, Tc/T)
    _phirall_1 = fused_phir_all(d1, Tc/T)
    _phi0all_0 = fused_phi0_all(d0, Tc/T)
    _phi0all_1 = fused_phi0_all(d1, Tc/T)
    ''' Derivatives along Maxwell-constr. level set G(t, d0, d1) = 0 '''
    # Vector components of partial dG/dt = (dG0/dt; dG1/dt)
    dG0dt = d1*_phirall_1.phir_dt + _phirall_1.phir_t + _phi0all_1.phi0_t \
      - d0*_phirall_0.phir_dt - _phirall_0.phir_t - _phi0all_0.phi0_t
    dG1dt = d0*d0*_phirall_0.phir_dt - d1*d1*_phirall_1.phir_dt
    # Vector components of partial dG/d0
    dG0dd0 = -2.0*_phirall_0.phir_d \
      - d0*_phirall_0.phir_dd - _phi0all_0.phi0_d
    dG1dd0 = 1.0 + 2.0 * d0 * _phirall_0.phir_d + d0*d0*_phirall_0.phir_dd
    # Vector components of partial dG/d1
    dG0dd1 = 2.0*_phirall_1.phir_d \
      + d1*_phirall_1.phir_dd + _phi0all_1.phi0_d
    dG1dd1 = -1.0 - 2.0 * d1 * _phirall_1.phir_d - d1*d1*_phirall_1.phir_dd
    # Compute d(d0)/dt and d(d1)/dt constrainted to level set of G
    _det = dG0dd0 * dG1dd1 - dG0dd1 * dG1dd0
    dd0dt = -(dG1dd1 * dG0dt - dG0dd1 * dG1dt) / _det
    dd1dt = -(-dG1dd0 * dG0dt + dG0dd0 * dG1dt) / _det
    # Construct derivatives of volume:
    #   dv0/dt = partial(v0, t) + partial(v0, d) * dd0/dt
    #   dv0/dT = dv0/dt * dt/dT
    dv0dT = -1.0/(rhoc * d0 * d0) * dd0dt * (-Tc/(T*T))
    dv1dT = -1.0/(rhoc * d1 * d1) * dd1dt * (-Tc/(T*T))
    # Construct derivatives of internal energy
    du0dT = R * Tc * (
      (_phirall_0.phir_dt + _phi0all_0.phi0_dt) * dd0dt
      + (_phirall_0.phir_tt + _phi0all_0.phi0_tt)
    ) * (-Tc/(T*T))
    du1dT = R * Tc * (
      (_phirall_1.phir_dt + _phi0all_1.phi0_dt) * dd1dt
      + (_phirall_1.phir_tt + _phi0all_1.phi0_tt)
    ) * (-Tc/(T*T))
    # Construct dx/dT (change in steam fraction subject to mixture
    #   conditions) as partial(x, T) + partial(x, v) * dv/dT
    v  = 1.0/(d*rhoc)
    vl = 1.0/(d0*rhoc)
    vv = 1.0/(d1*rhoc)

    # Compute partials
    partial_dxdT = ((v - vv) * dv0dT - (v - vl) * dv1dT) \
      / ((vv - vl) * (vv - vl))
    partial_dxdv =  1.0 / (vv - vl)
    partial_dxdd = -partial_dxdv / (d * d * rhoc)
    # Compute saturation-pressure-temperature slope
    dpsatdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
      * R * (log(rho_satv / rho_satl) + _phirall_1.phir - _phirall_0.phir \
        - t * (_phirall_1.phir_t - _phirall_0.phir_t))
    uv = R * Tc * (_phirall_1.phir_t + _phi0all_1.phi0_t)
    ul = R * Tc * (_phirall_0.phir_t + _phi0all_0.phi0_t)

    ew = x * uv + (1.0 - x) * ul
    dewdT = (uv - ul) * partial_dxdT + x*du1dT + (1.0-x)*du0dT
    dewdd = (uv - ul) * partial_dxdd

    lv_hypothetical = {
      "x": x,
      "ew": ew,
      "v": v,
      "ul": ul,
      "uv": uv,
      "vl": vl,
      "vv": vv,
      "dewdT": dewdT,
      "dpsatdT": dpsatdT,
      "partial_dxdT": partial_dxdT,
      "partial_dxdv": partial_dxdv,
      "partial_dxdd": partial_dxdd,
      "dv0dT": dv0dT,
      "dv1dT": dv1dT,
      "du0dT": du0dT,
      "du1dT": du1dT,
    }
  else:
    lv_hypothetical = {}

  ''' Compute dew/dd, dew/dT '''
  if T < Tc:
    if rho_satv < d * rhoc < rho_satl:
      # Load LV hypotheticals
      dewdd = dewdd
      dewdT = dewdT
      ew = ew
      pmix = psat
      dpdd = 0.0
      dpdT = dpsatdT
      region_type = _FLAG_LV
    else:
      # Pure phase
      _phirall = fused_phir_all(d, t)
      dewdd = R * Tc * _phirall.phir_dt
      ew = R * Tc * (_phirall.phir_t + phi0_t(d, t))
      dewdT = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
      pmix = d * rhoc * R * T * (1.0 + d * _phirall.phir_d)
      dpdd = rhoc * R * T * (1.0 + 2.0 * d * _phirall.phir_d
             + d * d * _phirall.phir_dd)
      dpdT = d * rhoc * R * (
        1.0 + d * _phirall.phir_d - t * d * _phirall.phir_dt)
      region_type = _FLAG_V if d * rhoc <= rho_satv else _FLAG_L
  else:
    # Pure phase
    _phirall = fused_phir_all(d, t)
    dewdd = R * Tc * _phirall.phir_dt
    ew = R * Tc * (_phirall.phir_t + phi0_t(d, t))
    dewdT = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
    pmix = d * rhoc * R * T * (1.0 + d * _phirall.phir_d)
    dpdd = rhoc * R * T * (1.0 + 2.0 * d * _phirall.phir_d
             + d * d * _phirall.phir_dd)
    dpdT = d * rhoc * R * (
      1.0 + d * _phirall.phir_d - t * d * _phirall.phir_dt)
    region_type = _FLAG_C

  ''' Compute output '''
  # Compute dependents, with water pressure as mixture pressure
  e_mix = vol_energy / rho_mix
  rhoa = pmix / (R_a * T)
  rhom = rho_m0 * (1.0 + (pmix - p_m0) / K)
  alphaa = rho_mix * ya / rhoa
  alpham = rho_mix * ym / rhom
  # Clip water energy
  if ew > 5e6:
    ew = 5e6
  elif ew < 0.0:
    ew = 0.0
  ea = R_a / (gamma_a - 1.0) * T
  em = c_v_m0 * T + magma_mech_energy(pmix, K, p_m0, rho_m0)
  # Compute energy fractions (symbol eps)
  epsw = yw * ew / e_mix
  epsa = ya * ea / e_mix
  epsm = ym * em / e_mix
  # Chain rule for d pressure w.r.t. alpha_w
  dpdaw = -1.0 / rhoc * dpdd * rho_mix * yw / (alpha_w * alpha_w)
  # Gradient of energy condition w.r.t. [alpha_w, T]
  J[0,1] = 1.0 / e_mix * (yw * dewdT + ya * R_a / (gamma_a - 1.0) + ym * c_v_m0
    + ym * pmix * rho_m0 / (K * rhom * rhom) * dpdT)
  J[0,0] = 1.0 / e_mix * (-yw / rhoc * dewdd * rho_mix * yw / (alpha_w*alpha_w)
    + ym * pmix * rho_m0 / (K * rhom * rhom) * dpdaw)
  # Gradient of volume condition w.r.t. [alpha_w, T]
  J[1,1] = rho_mix * ya * R_a / pmix \
    + rho_mix * (ya * (-R_a * T / (pmix*pmix))  + ym * (-rho_m0 / (K * rhom * rhom))) * dpdT
  J[1,0] = 1 - rho_mix * ya * R_a * T /(pmix * pmix)* dpdaw \
    - rho_mix * ym * rho_m0 / (K * rhom * rhom) * dpdaw
  f[0] = epsw + epsa + epsm - 1.0
  f[1] = alpha_w + alphaa + alpham - 1.0
  
  return f, J, lv_hypothetical, pmix, region_type

cdef struct OutKernel2:
  DTYPE_t f0
  DTYPE_t f1
  DTYPE_t step0
  DTYPE_t step1
  DTYPE_t pmix
  int region_type
# cdef struct WLMAParams:
#   DTYPE_t vol_energy
#   DTYPE_t rho_mix
#   DTYPE_t yw
#   DTYPE_t ya
#   DTYPE_t K
#   DTYPE_t p_m0
#   DTYPE_t rho_m0
#   DTYPE_t c_v_m0
#   DTYPE_t R_a
#   DTYPE_t gamma_a
# Enumerate flag types
cdef int _FLAG_L = 0
cdef int _FLAG_LV = 1
cdef int _FLAG_V = 2
cdef int _FLAG_C = 3

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef OutKernel2 kernel2_WLMA(DTYPE_t alpha_w, DTYPE_t T, WLMAParams params):
  ''' Returns residuals of the system of two equations for energy fraction and
  volume fraction summation conditions.
  Returns:
    {f0, f1, step0, step1, pmix, region_type}
  
  The energy-volume system is as follows.
  let ef_w = y_w * e_w/e_mix: energy fraction of water
  let ef_r = y_r * e_r/e_mix: energy fraction of residual components
  ef_w + ef_r - 1 == 0
  yw/vmix / (rhoc*d) + yr/vmix * vr(pr, Tr) - 1 == 0
  '''

  # Zealous unpacking
  cdef DTYPE_t vol_energy, rho_mix, yw, ya, K, \
    p_m0, rho_m0, c_v_m0, R_a, gamma_a
  vol_energy, rho_mix, yw, ya, K, p_m0, rho_m0, c_v_m0, R_a, gamma_a = \
    params.vol_energy, params.rho_mix, params.yw, params.ya, params.K, \
    params.p_m0, params.rho_m0, params.c_v_m0, params.R_a, params.gamma_a

  # Compute dependents
  cdef DTYPE_t t = Tc / T
  cdef DTYPE_t ym = 1.0 - (yw + ya)
  # Compute water phase density as dependent
  cdef DTYPE_t rhow = rho_mix * yw / alpha_w
  # Set bound on rhow based on pressure bound [, 1 GPa]
  rhow = max(1e-9, rhow)
  rhow = min(1260.0, rhow)
  cdef DTYPE_t d = rhow / rhoc

  ''' Compute saturation state at exact pressure. '''
  cdef SatTriple sat_triple
  cdef DTYPE_t psat, x, rho_satl, rho_satv, d0, d1
  cdef Derivatives_phir_0_1_2 _phirall_0, _phirall_1
  cdef Derivatives_phi0_0_1_2 _phi0all_0, _phi0all_1
  cdef DTYPE_t dG0dt, dG1dt, dG0dd0, dG1dd0, dG0dd1, dG1dd1, _det, dd0dt, dd1dt
  cdef DTYPE_t dv0dT, dv1dT, du0dT, du1dT, v, vl, vv, ul, uv
  cdef DTYPE_t partial_dxdT, partial_dxdv, partial_dxdd, dvdp, dpsatdT 
  cdef DTYPE_t ew, dewdT, dewdd
  cdef int region_type
  cdef DTYPE_t dpdd, dpdT, pmix
  cdef Derivatives_phir_0_1_2 _phirall
  if T < Tc:
    # Compute saturation curves
    sat_triple = prho_sat(T)
    psat, rho_satl, rho_satv = \
      sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
    if rho_satv < d * rhoc < rho_satl:
      ''' Compute mixed-phase state '''
      # Compute steam fraction (fraction vapor mass per total water mass)
      x = (1.0 / rhow - 1.0 / rho_satl) / (1.0 / rho_satv - 1.0 / rho_satl)
      # Compute temperature update using saturation relation
      d0 = rho_satl/rhoc
      d1 = rho_satv/rhoc
      _phirall_0 = fused_phir_all(d0, Tc/T)
      _phirall_1 = fused_phir_all(d1, Tc/T)
      _phi0all_0 = fused_phi0_all(d0, Tc/T)
      _phi0all_1 = fused_phi0_all(d1, Tc/T)
      ''' Derivatives along Maxwell-constr. level set G(t, d0, d1) = 0 '''
      # Vector components of partial dG/dt = (dG0/dt; dG1/dt)
      dG0dt = d1*_phirall_1.phir_dt + _phirall_1.phir_t + _phi0all_1.phi0_t \
        - d0*_phirall_0.phir_dt - _phirall_0.phir_t - _phi0all_0.phi0_t
      dG1dt = d0*d0*_phirall_0.phir_dt - d1*d1*_phirall_1.phir_dt
      # Vector components of partial dG/d0
      dG0dd0 = -2.0*_phirall_0.phir_d \
        - d0*_phirall_0.phir_dd - _phi0all_0.phi0_d
      dG1dd0 = 1.0 + 2.0 * d0 * _phirall_0.phir_d + d0*d0*_phirall_0.phir_dd
      # Vector components of partial dG/d1
      dG0dd1 = 2.0*_phirall_1.phir_d \
        + d1*_phirall_1.phir_dd + _phi0all_1.phi0_d
      dG1dd1 = -1.0 - 2.0 * d1 * _phirall_1.phir_d - d1*d1*_phirall_1.phir_dd
      # Compute d(d0)/dt and d(d1)/dt constrainted to level set of G
      _det = dG0dd0 * dG1dd1 - dG0dd1 * dG1dd0
      dd0dt = -(dG1dd1 * dG0dt - dG0dd1 * dG1dt) / _det
      dd1dt = -(-dG1dd0 * dG0dt + dG0dd0 * dG1dt) / _det
      # Construct derivatives of volume:
      #   dv0/dt = partial(v0, t) + partial(v0, d) * dd0/dt
      #   dv0/dT = dv0/dt * dt/dT
      dv0dT = -1.0/(rhoc * d0 * d0) * dd0dt * (-Tc/(T*T))
      dv1dT = -1.0/(rhoc * d1 * d1) * dd1dt * (-Tc/(T*T))
      # Construct derivatives of internal energy
      du0dT = R * Tc * (
        (_phirall_0.phir_dt + _phi0all_0.phi0_dt) * dd0dt
        + (_phirall_0.phir_tt + _phi0all_0.phi0_tt)
      ) * (-Tc/(T*T))
      du1dT = R * Tc * (
        (_phirall_1.phir_dt + _phi0all_1.phi0_dt) * dd1dt
        + (_phirall_1.phir_tt + _phi0all_1.phi0_tt)
      ) * (-Tc/(T*T))
      # Construct dx/dT (change in steam fraction subject to mixture
      #   conditions) as partial(x, T) + partial(x, v) * dv/dT
      v  = 1.0/(d*rhoc)
      vl = 1.0/(d0*rhoc)
      vv = 1.0/(d1*rhoc)
      # Compute partials
      partial_dxdT = ((v - vv) * dv0dT - (v - vl) * dv1dT) \
        / ((vv - vl) * (vv - vl))
      partial_dxdv =  1.0 / (vv - vl)
      partial_dxdd = -partial_dxdv / (d * d * rhoc)
      # Compute saturation-pressure-temperature slope
      dpsatdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
        * R * (log(rho_satv / rho_satl) + _phirall_1.phir - _phirall_0.phir \
          - t * (_phirall_1.phir_t - _phirall_0.phir_t))
      uv = R * Tc * (_phirall_1.phir_t + _phi0all_1.phi0_t)
      ul = R * Tc * (_phirall_0.phir_t + _phi0all_0.phi0_t)
      # Compute energy and its derivatives
      ew = x * uv + (1.0 - x) * ul
      dewdd = (uv - ul) * partial_dxdd
      dewdT = (uv - ul) * partial_dxdT + x*du1dT + (1.0-x)*du0dT
      # Compute commons
      pmix = psat
      dpdd = 0.0
      dpdT = dpsatdT
      region_type = _FLAG_LV
    else:
      # Compute common quantities for pure phase subcritical
      _phirall = fused_phir_all(d, t)
      ew = R * Tc * (_phirall.phir_t + phi0_t(d, t))
      dewdd = R * Tc * _phirall.phir_dt
      dewdT = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
      pmix = d * rhoc * R * T * (1.0 + d * _phirall.phir_d)
      dpdd = rhoc * R * T * (1.0 + 2.0 * d * _phirall.phir_d
             + d * d * _phirall.phir_dd)
      dpdT = d * rhoc * R * (
        1.0 + d * _phirall.phir_d - t * d * _phirall.phir_dt)
      region_type = _FLAG_V if d * rhoc <= rho_satv else _FLAG_L
  else:
    # Compute common quantities for pure phase supercritical
    _phirall = fused_phir_all(d, t)
    ew = R * Tc * (_phirall.phir_t + phi0_t(d, t))
    dewdd = R * Tc * _phirall.phir_dt
    dewdT = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
    pmix = d * rhoc * R * T * (1.0 + d * _phirall.phir_d)
    dpdd = rhoc * R * T * (1.0 + 2.0 * d * _phirall.phir_d
             + d * d * _phirall.phir_dd)
    dpdT = d * rhoc * R * (
      1.0 + d * _phirall.phir_d - t * d * _phirall.phir_dt)
    region_type = _FLAG_C

  ''' Compute output '''
  # Compute dependents, with water pressure as mixture pressure
  cdef DTYPE_t e_mix = vol_energy / rho_mix
  cdef DTYPE_t rhoa = pmix / (R_a * T)
  cdef DTYPE_t rhom = rho_m0 * (1.0 + (pmix - p_m0) / K)
  cdef DTYPE_t alphaa = rho_mix * ya / rhoa
  cdef DTYPE_t alpham = rho_mix * ym / rhom
  # Clip water energy for range of validity (avoids issue if energy is evaluated
  #   from the Helmholtz potential in the phase equilibrium region)
  ew = min(ew, 5e6)
  ew = max(ew, 1e-6)
  cdef DTYPE_t ea = R_a / (gamma_a - 1.0) * T
  cdef DTYPE_t em = c_v_m0 * T + magma_mech_energy(pmix, K, p_m0, rho_m0)
  # Chain rule for d pressure w.r.t. alpha_w
  cdef DTYPE_t dpdaw = -1.0 / rhoc * dpdd * rho_mix * yw / (alpha_w * alpha_w)

  ''' Assemble Jacobian and residual vector '''
  cdef DTYPE_t _f0, _f1
  cdef DTYPE_t _J00, _J01, _J10, _J11
  # Gradient of energy condition w.r.t. [alpha_w, T]
  _J00 = 1.0 / e_mix * (-yw / rhoc * dewdd * rho_mix * yw / (alpha_w * alpha_w)
    + ym * pmix * rho_m0 / (K * rhom * rhom) * dpdaw)
  _J01 = 1.0 / e_mix * (yw * dewdT + ya * R_a / (gamma_a - 1.0) + ym * c_v_m0
    + ym * pmix * rho_m0 / (K * rhom * rhom) * dpdT)
  # Gradient of volume condition w.r.t. [alpha_w, T]
  _J10 = 1.0 - rho_mix * ya * R_a * T /(pmix * pmix) * dpdaw \
    - rho_mix * ym * rho_m0 / (K * rhom * rhom) * dpdaw
  _J11 = rho_mix * ya * R_a / pmix \
    + rho_mix * (ya * (-R_a * T / (pmix * pmix)) \
    + ym * (-rho_m0 / (K * rhom * rhom))) * dpdT
  # Energy condition
  _f0 = (yw * ew + ya * ea + ym * em) / e_mix - 1.0
  # Volume condition
  _f1 = alpha_w + alphaa + alpham - 1.0
  # Compute step
  _det = _J00 * _J11 - _J01 * _J10
  cdef DTYPE_t _step0 = -(_J11 * _f0 - _J01 * _f1) / _det
  cdef DTYPE_t _step1 = -(-_J10 * _f0 + _J00 * _f1) / _det
  
  return OutKernel2(_f0, _f1, _step0, _step1, pmix, region_type)

''' Robust WLMA pT function '''

# Set numerics constants
cdef unsigned int NUM_ITER_BACKTRACK = 16
cdef unsigned int NUM_ITER_PHASE_BACKTRACK = 11
# Define output type of iterate_backtrack_box
cdef struct OutIterate:
  DTYPE_t U0
  DTYPE_t U1
  DTYPE_t fnorm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef OutIterate iterate_backtrack_box(DTYPE_t U0, DTYPE_t U1, WLMAParams params,
    logger=False):
  ''' Backtracked Newton with box bounds, mapping
    U == {alpha_w, T} -> {U0, U1, fnorm}.
  If logger is passed, its method log is called with signature
    log(level:str, data:dict).
  Vector quantities U0, U1 are passed as scalars. '''
  # Compute Newton step
  cdef OutKernel2 _out_step = kernel2_WLMA(U0, U1, params)
  cdef int region_type_curr = _out_step.region_type
  cdef DTYPE_t _fnorm = sqrt(_out_step.f0*_out_step.f0 + _out_step.f1*_out_step.f1)
  # Define convenience stack arrays
  cdef DTYPE_t[2] U = [U0, U1]
  cdef DTYPE_t[2] step = [_out_step.step0, _out_step.step1]

  if logger:
    # logger.info(f"Init   | f: {f[0]}, {f[1]}, U:{U[0]}, {U[1]}, step: {step[0]}, {step[1]} ")
    logger.log("info", {"stage": "init",
      "f0": _out_step.f0, "f1": _out_step.f1,
      "U0": U[0], "U1": U[1],
      "step0": step[0], "step1": step[1],
      "region_type_curr": region_type_curr})

  ''' Scaling for box constraints '''
  # Define box bounds
  cdef DTYPE_t[2] U_min = [1e-9, 273.16]
  cdef DTYPE_t[2] U_max = [1.0, 2273.15]
  # Scale step to box bounds
  cdef DTYPE_t step_size_factor = 1.0
  cdef unsigned short i
  for i in range(2):
    if U[i] + step[i] < U_min[i]:
      # i-th min bound is active; scale to step hitting just the boundary
      step_size_factor = min(step_size_factor, (U_min[i] - U[i])/ step[i])
    if U[i] + step[i] > U_max[i]:
      # i-th max bound is active
      step_size_factor = min(step_size_factor, (U_max[i] - U[i])/ step[i])
  # Clip step size factor to [0, 1.0]
  step_size_factor = 0.0 if step_size_factor < 0.0 else step_size_factor
  step_size_factor = 1.0 if step_size_factor > 1.0 else step_size_factor
  # Apply step scaling
  step[0] *= step_size_factor
  step[1] *= step_size_factor
  # Floating point precision correction for alphaw_min
  for i in range(2):
    if U[i] + step[i] < U_min[i]:
      step[i] = U_min[i] - U[i]
  if logger:
    logger.log("info", {"stage": "bounds",
      "U0": U[0] + step[0], "U1": U[1] + step[1],
      "step0": step[0], "step1": step[1]})

  ''' Armijo backtracking line search '''
  # Define greed ratio in (0,1) for Armijo conditions
  cdef DTYPE_t armijo_c = 0.1
  # Define step reduction factor in (0, 1) used each iteration
  cdef DTYPE_t step_reduction_factor = 0.5
  # Set step size factor (==1.0 the first time Armijo condition is checked)
  cdef DTYPE_t _a = 1.0 / step_reduction_factor
  # Set minimum decrement of the objective value. Note that
  #   c * grad(norm(f)) dot J^{-1} f = c || f ||, since Newton sends model to 0
  cdef DTYPE_t min_decrement = armijo_c * _fnorm
  # Backtracking
  cdef DTYPE_t[2] U_next
  cdef DTYPE_t f_along_line
  for i in range(NUM_ITER_BACKTRACK):
    _a *= step_reduction_factor
    U_next[0] = U[0] + _a * step[0]
    U_next[1] = U[1] + _a * step[1]
    # Compute value function along search direction
    _out_step = kernel2_WLMA(U_next[0], U_next[1], params)
    f_along_line = sqrt(_out_step.f0*_out_step.f0 + _out_step.f1*_out_step.f1)
    if logger:
      logger.log("info", {"stage": "backtrack",
        "f0": _out_step.f0, "f1": _out_step.f1,
        "U0": U_next[0], "U1": U_next[1],
        "step0": _a*step[0], "step1": _a*step[1],
        "fnorm": f_along_line, "freq": (1.0 - _a * armijo_c) * _fnorm,
        "region_type": _out_step.region_type})
    if _a * _a * (step[0] * step[0] + step[1] * step[1]) < 1e-12:
      if logger:
        logger.log("warning", {"stage": "steptoosmall",
          "squarestepsize": _a * _a * (step[0] * step[0] + step[1] * step[1]),})
      # Early return with f = 0 sentinel value
      return OutIterate(U_next[0], U_next[1], 0.0)

    # Armijo condition: improvement is at least a portion of a * |f|
    if f_along_line <= (1.0 - _a * armijo_c) * _fnorm:
      break
  else:
    if logger:
      logger.log("critical",
        {"message": "Could not backtrack to a satisfactory step."})
  
  ''' Step damping near f-discontinuity due to phase boundary. '''
  cdef DTYPE_t f_at_t_max = f_along_line
  cdef DTYPE_t t_min = 0.0
  cdef DTYPE_t t_max = 1.0
  cdef DTYPE_t _m
  cdef DTYPE_t[2] U_m
  if _out_step.region_type != region_type_curr:
    # Identify phase boundary for b in (0, 1) with bisection
    for i in range(NUM_ITER_PHASE_BACKTRACK):
      _m = 0.5 * (t_min + t_max)
      U_m[0] = U[0] + _m * _a * step[0]
      U_m[1] = U[1] + _m * _a * step[1]
      # Compute midpoint values
      _out_step = kernel2_WLMA(U_m[0], U_m[1], params)
      # Evaluate objective function norm
      f_along_line = sqrt(_out_step.f0*_out_step.f0 + _out_step.f1*_out_step.f1)
      if logger:
        logger.log("info", {"stage": "phaseboundarybacktrack",
          "f0": _out_step.f0, "f1": _out_step.f1,
          "U0": U_m[0], "U1": U_m[1],
          "step0": _m*_a*step[0], "step1": _m*_a*step[1],
          "fnorm": f_along_line,
          "region_type": _out_step.region_type,
          "region_type_curr": region_type_curr})
      if _out_step.region_type == region_type_curr:
        t_min = _m
      else:
        t_max = _m
        # Cache |f| at t_max
        f_at_t_max = f_along_line
    # Conclude backtracking by using smallest known step that switches phases
    U_next[0] = U[0] + t_max * _a * step[0]
    U_next[1] = U[1] + t_max * _a * step[1]
    if logger:
      logger.log("info", {"stage": "finalstep",
        "f0": _out_step.f0, "f1": _out_step.f1,
        "U0": U_next[0], "U1": U_next[1],
        "step0": t_max*_a*step[0], "step1": t_max*_a*step[1],
        "fnorm": f_at_t_max,
        "region_type": _out_step.region_type,
        "region_type_curr": region_type_curr})
    
  return OutIterate(U_next[0], U_next[1], f_at_t_max)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef DTYPE_t p_LMA(DTYPE_t T, DTYPE_t ya, WLMAParams params):
  ''' Compute pressure in an ideal gas + LM system. '''
  cdef DTYPE_t ym = 1.0 - ya
  # Define useful quantities: additive constant to magma EOS
  cdef DTYPE_t sym1 = params.K - params.p_m0
  # Define partial pressure of gas mixture
  cdef DTYPE_t sym2 = (ya * params.R_a) * params.rho_mix * T
  # Define b (from quadratic formula)
  cdef DTYPE_t b = -(sym1 - sym2 
    - params.K / params.rho_m0 * ym * params.rho_mix)
  # Define total gas volume fraction
  cdef DTYPE_t phi =  0.5 / sym1 * (-b + sqrt(b*b + 4.0*sym1*sym2))
  return (ya * params.R_a) * params.rho_mix * T \
    + (1.0 - phi) * (params.p_m0 - params.K) \
    + (params.K / params.rho_m0 * ym * params.rho_mix)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef DTYPE_t phi_LMA(DTYPE_t T, DTYPE_t ya, WLMAParams params):
  ''' Compute gas volume fraction in an ideal gas + LM system. '''
  cdef DTYPE_t ym = 1.0 - ya
  # Define useful quantities: additive constant to magma EOS
  cdef DTYPE_t sym1 = params.K - params.p_m0
  # Define partial pressure of gas mixture
  cdef DTYPE_t sym2 = (ya * params.R_a) * params.rho_mix * T
  # Define b (from quadratic formula)
  cdef DTYPE_t b = -(sym1 - sym2 
    - params.K / params.rho_m0 * ym * params.rho_mix)
  return  0.5 / sym1 * (-b + sqrt(b*b + 4.0*sym1*sym2))

# Set numerics for finding initial guesses
cdef unsigned int NUM_ITER_BISECTION_INIT = 16
# Set numerics for solving energy/volume fraction equations
cdef unsigned int NUM_ITER_NEWTON = 32
cdef DTYPE_t FTOL_NEWTON = 1e-12

# Define linearization point at 300 K, 1 bar
cdef DTYPE_t ref_T_atm = 300.0
cdef DTYPE_t ref_rhow_atm = 996.55634039
cdef DTYPE_t ref_ew_atm = u(ref_rhow_atm, ref_T_atm)
cdef DTYPE_t ref_c_v_atm = c_v(ref_rhow_atm, ref_T_atm)
# Define linearization point at 1000 K, 50 MPa
cdef DTYPE_t ref_T_hot0 = 1000.0
cdef DTYPE_t ref_rhow_hot0 = 123.48077675
cdef DTYPE_t ref_ew_hot0 = u(ref_rhow_hot0, ref_T_hot0)
cdef DTYPE_t ref_c_v_hot0 = c_v(ref_rhow_hot0, ref_T_hot0)

@cython.boundscheck(True)
@cython.wraparound(True)
@cython.nonecheck(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
cpdef TriplerhopT conservative_to_pT_WLMA_bn(
  DTYPE_t vol_energy, DTYPE_t rho_mix, DTYPE_t yw, DTYPE_t ya,
  DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0, DTYPE_t c_v_m0,
  DTYPE_t R_a, DTYPE_t gamma_a, logger=False):
  ''' Map from conservative to pressure-temperature variables for WLMA model.
  Uses bisection + Newton (bn) to compute initial guesses based on the
  approximate regime of the inputs (approximate the phase of water) followed
  by a backtracking Newton search for (p, T) solving the energy fraction and
  volume fraction equations. The range of validity is designed in terms of mass
  fractions, water density, and temperature as
    1e-7 <= yw <= 1-1e-7,
    1e-7 <= ya <= 1-1e-7,
    0.1 <= rhow <= 1260
    278.16 <= T <= 1500.
  The two constraint equations are as follows. 
    let ef_w = y_w * e_w/e_mix: energy fraction of water
    let ef_r = y_r * e_r/e_mix: energy fraction of residual components
    eqn ef_w + ef_r - 1 == 0
    eqn yw/vmix / (rhoc*d) + yr/vmix * vr(pr, Tr) - 1 == 0.
  '''
  # Compute magma mass fraction (using ordering stable for yw, ya ~ 0)
  cdef DTYPE_t ym = 1.0 - (yw + ya)
  # Parameter Packing
  cdef WLMAParams params = WLMAParams(vol_energy, rho_mix, yw, ya,
    K, p_m0, rho_m0, c_v_m0, R_a, gamma_a)

  ''' Check for endmember/limit cases '''
  if vol_energy * rho_mix == 0:
    if logger:
      logger.log("info", {
        "message": "vol_energy or rho_mix is zero. Returning zeros."})
    return TriplerhopT(0.0, 0.0, 0.0)

  cdef DTYPE_t min_y = 1e-14
  # Clip yw, ym to minimum values
  yw = max(min_y, yw)
  ym = max(min_y, ym)
  # Remove excess from air (best precision order of operations)
  ya = 1.0 - (yw + ym) if yw + ym + ya > 1.0 else ya

  ''' Low-water case '''
  cdef DTYPE_t ya_replaced, T_approx, p_as_ideal, rhoa, R_gas_eff, R_w_max, e_w
  cdef DTYPE_t[7] R_vec = [0.125*R_a, 0.25*R_a, 0.5*R_a,
    R_a, 2*R_a, 4*R_a, 8*R_a]
  cdef unsigned int i
  if yw <= 1e-6:
    # Water -> air replacement
    ya_replaced = 1.0 - ym
    # Approximate temperature without water, magma mech energy
    T_approx = (vol_energy / rho_mix) / (
      ya_replaced * R_a / (gamma_a - 1.0) + ym * c_v_m0)
    params.ya = ya_replaced
    params.yw = 0.0
    # Compute LM+A equilibrium pressure and air-density
    p_as_ideal = p_LMA(T_approx, ya_replaced, params)
    rhoa = p_as_ideal / (R_a * T_approx)
    # One-step correction for water, using air density as water density
    #   (avoid using vol frac formulation, which can explode pw)
    e_w = u(rhoa, max(T_approx, 273.16))
    T_approx = (vol_energy / rho_mix - yw * e_w
                - ym * magma_mech_energy(p_as_ideal, K, p_m0, rho_m0))\
                / (ya * R_a / (gamma_a - 1.0) + ym * c_v_m0)
    # Compute effective gas constant using air + water mixture
    # experimental: modified gas stiffness (only good for ya >> yw)
    # _R_w = float_mix_functions.p(_rhoa, _T_approx) / (_rhoa * _T_approx)
    params.R_a = (ya * R_a + yw * R) / (ya + yw)
    p_as_ideal = p_LMA(T_approx, ya_replaced, params)

    # # Mixture gas constant correction
    # for i in range(7):
    #   params.R_a = R_vec[i]
    #   # Compute volume criterion
    #   kernel2_WLMA()
    #   kernel2_WLMA(alpha_w, T, params).f1

    # TODO: replace returned bogus rhow value
    if logger:
      logger.log("info", {
        "stage": "dryapprox",
        "yw": yw,
        "T_approx": T_approx,
        "p_approx": p_as_ideal,
      })
    return TriplerhopT(rhoa, p_LMA(T_approx, ya_replaced, params), T_approx)

  ''' Compute initial guesses for (alphaw, T) heuristically '''
  # Estimate energy using reference liquid water heat capacity and neglecting
  #   LM strain
  cdef DTYPE_t _e_diff = vol_energy/rho_mix - yw * (ref_ew_atm) \
    - ya * R_a / (gamma_a - 1.0) * ref_T_atm - ym * c_v_m0 * ref_T_atm
  cdef DTYPE_t _T_init = ref_T_atm + _e_diff \
    / (yw * ref_c_v_atm + ya * R_a / (gamma_a - 1.0) + ym * c_v_m0)
  if _T_init > 900:
    # Use higher linearization point
    _e_diff = vol_energy/rho_mix - yw * (ref_ew_hot0) \
      - ya * R_a / (gamma_a - 1.0) * ref_T_hot0 - ym * c_v_m0 * ref_T_hot0
    _T_init = ref_T_hot0 + _e_diff \
      / (yw * ref_c_v_hot0 + ya * R_a / (gamma_a - 1.0) + ym * c_v_m0)
  # Clip temperature values below triple point
  _T_init = max(273.16, _T_init)
  cdef SatTriple sat_triple
  cdef DTYPE_t psat, hyp_rhoa, hyp_rhom, hyp_vw, hyp_x, _alphaw_initLV
  # (1/3) Compute saturation-aware initial guess
  if _T_init < Tc:
    # Compute saturation curves
    sat_triple = prho_sat(_T_init)
    # Compute hypothetical saturation state (at p_sat(_T_init))
    psat = sat_triple.psat
    hyp_rhoa = psat / (R_a * _T_init)
    hyp_rhom = rho_m0 * (1.0 + (psat - p_m0) / K)
    # Compute water volume from volume fraction condition
    hyp_vw = (1.0 / rho_mix - ya / hyp_rhoa + ym / hyp_rhom) / yw
    # Compute hypothetical steam fraction
    hyp_x = (hyp_vw - 1.0 / sat_triple.rho_satl) \
            / (1.0/sat_triple.rho_satv - 1.0/sat_triple.rho_satl)
    if 0 <= hyp_x and hyp_x <= 1:
      # Accept saturation state
      _alphaw_initLV =  yw * rho_mix * hyp_vw
    elif hyp_x < 0:
      # Liquid-like
      _alphaw_initLV = yw * rho_mix / 1000.0
    else:
      # Vapor-like extrapolation from superheated vapour (likely inaccurate)
      _alphaw_initLV = yw * rho_mix / sat_triple.rho_satv
  else:
    # Critical density extrapolation (likely inaccurate)
    _alphaw_initLV = yw * rho_mix / rhoc   
  # Replace invalid initial guesses with an independent initial guess
  if _alphaw_initLV < 0 or _alphaw_initLV > 1:    
    _alphaw_initLV = 0.75
  # (2/3) Compute volume-fraction-accurate initial guess using bisection;
  #   if no root is bracketed, _m converges to an endpoint (_a or _b).
  cdef DTYPE_t _a = 1e-7
  cdef DTYPE_t _b = 1.0
  cdef DTYPE_t _m, _alphaw_initVF
  cdef OutKernel2 kernel_out
  for i in range(NUM_ITER_BISECTION_INIT):
    _m = 0.5 * (_a + _b)
    # Compare volume residual sum(vol fracs) - 1.0
    if kernel2_WLMA(_m, _T_init, params).f1 > 0:
      # Decrease water volume fraction to mitigate excess volume
      _b = _m
    else:
      # Increase water volume fraction to mitigate insufficient volume
      _a = _m
  # Select lower volume fraction bound (heuristic: better landscape along
  #   search dir)
  _alphaw_initVF = _a
  # (3/3) Compute water-density-accurate initial guess (needed for stiff
  #   liquid water) using bisection
  _a, _b = 0.01, 1160
  # Set default initial guess (used if T >= Tc)
  cdef DTYPE_t _alphaw_initrhow = 0.25
  cdef DTYPE_t f_vol_m, f_vol_a
  # Use initial T guess, and bisection search over water density
  # Compute volume residual for endpoint _a
  f_vol_a = kernel2_WLMA(yw * rho_mix / _a, _T_init, params).f1    
  # Bisect for volume condition using water density as the decision variable
  for i in range(NUM_ITER_BISECTION_INIT):
    _m = 0.5 * (_a + _b)
    # Compute volume residual at rho_w = _m
    f_vol_m = kernel2_WLMA(yw * rho_mix / _m, _T_init, params).f1
    if f_vol_m * f_vol_a < 0:
      # m & a different sign; replace b
      _b = _m
    else:
      _a = _m
      f_vol_a = f_vol_m
  # Select volfrac based on midpoint water density
  _alphaw_initrhow = yw * rho_mix / (0.5 * (_a + _b))
  # Assemble and select from candidates for water volume fraction initial guess
  cdef DTYPE_t[8] _alpha_w_candidates = [
      _alphaw_initLV, # 1.0 - _alphaw_initLV,
      1e-5, 0.01, 0.5, 0.99, 1.0 - 1e-5,
      _alphaw_initVF, _alphaw_initrhow]
  # Select best candidate
  cdef DTYPE_t[8] _candidate_performance
  cdef unsigned int _argmin_candidate = 0
  cdef DTYPE_t _min_candidate = 1e32
  for i in range(8):
    kernel_out = kernel2_WLMA(_alpha_w_candidates[i], _T_init, params)
    # Compute squared norm of residuals as performance metric
    _candidate_performance[i] = kernel_out.f0 * kernel_out.f0 \
      + kernel_out.f1 * kernel_out.f1
    if _candidate_performance[i] < _min_candidate:
      _argmin_candidate = i
      _min_candidate = _candidate_performance[i]
  # Selection confidence heuristic
  if _min_candidate > 0.5:
    # No-confidence choice
    _argmin_candidate = 7

  # Finalize initial guess
  cdef DTYPE_t _alphaw_init = _alpha_w_candidates[_argmin_candidate]
  if logger:
    logger.log("info", {
      "stage": "initialguess",
      "alphaw_candidates": _alpha_w_candidates,
      "alphaw_candidate_performance": _candidate_performance,
      "alphaw_argmin": _argmin_candidate,
      "T_init": _T_init,
      "is_low_confidence": _min_candidate > 0.5,
    })



  ''' Backtracking Newton iteration '''
  cdef DTYPE_t _fnorm, U0, U1
  cdef OutIterate out_iterate_backtrack
  
  # Newton iterations
  U0, U1 = _alphaw_init, _T_init
  for i in range(NUM_ITER_NEWTON):
    # Take Newton step with backtracking
    out_iterate_backtrack = iterate_backtrack_box(U0, U1, params, logger=logger)
    U0, U1 = out_iterate_backtrack.U0, out_iterate_backtrack.U1
    if out_iterate_backtrack.fnorm < FTOL_NEWTON:
      break
  else:
    if logger:
      logger.log("critical", {
        "message": "Newton iteration failed.",
        "fnorm_last": out_iterate_backtrack.fnorm,
        "fnorm_req": FTOL_NEWTON,
      })
  
  ''' Finalize solve '''
  cdef DTYPE_t rhow   = rho_mix * yw / U0
  cdef DTYPE_t T_calc = U1
  cdef DTYPE_t p_calc = p(rhow, U1)
  if logger:
    logger.log("info", {
      "stage": "finalize",
      "rhow": rhow,
      "p": p_calc,
      "T": T_calc
    })
  return TriplerhopT(rhow, p_calc, T_calc)

''' Vectorizing wrappers '''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# @cython.cdivision(True)
def vec_conservative_to_pT_WLMA(np.ndarray vol_energy, np.ndarray rho_mix,
  np.ndarray yw, np.ndarray ya, DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0,
  DTYPE_t c_v_m0, DTYPE_t R_a, DTYPE_t gamma_a, logger=False) -> np.ndarray:
  ''' Python interface for the mapping
    (E, rho_mix, yw, ya; ...params...) -> (rhow, pmix, T, c).
  Maps arrays to arrays. Output is interlaced as (rhow, pmix, T, c).
  For logging, pass through logger with method
  log(self, level:str, data:dict). '''
  cdef int i = 0
  cdef int N = yw.size
  cdef TriplerhopT out_triple
  cdef DTYPE_t p
  # Memory management
  cdef np.ndarray[DTYPE_t] data = \
    np.ascontiguousarray(np.stack((
      vol_energy,
      rho_mix,
      yw,
      ya), axis=-1).ravel())
  # cdef np.ndarray[DTYPE_t] rhow = np.empty_like(yw)
  # cdef np.ndarray[DTYPE_t] pmix = np.empty_like(yw)
  # cdef np.ndarray[DTYPE_t] T = np.empty_like(yw)

  if data.size != 4*N:
    raise ValueError("Size of vector inputs are either not the same.")

  cdef WLMAParams params
  for i in range(N):
    params = WLMAParams(data[4*i], data[4*i+1], data[4*i+2], data[4*i+3],
      K, p_m0, rho_m0, c_v_m0, R_a, gamma_a)
    # Use packed inputs (vol_energy, rho_mix, yw, ya)
    out_triple = conservative_to_pT_WLMA_bn(
      data[4*i], data[4*i+1], data[4*i+2], data[4*i+3],
      K, p_m0, rho_m0, c_v_m0, R_a, gamma_a, logger=logger)
    # Reuse data track as output
    data[4*i]   = out_triple.rhow
    data[4*i+1] = out_triple.p
    data[4*i+2] = out_triple.T
    # TODO: replace with direct postprocess with access to last phir_*
    # TODO: replace sound speed computation for mixture
    data[4*i+3] = mix_sound_speed(out_triple.rhow, out_triple.p, out_triple.T,
      params)
  return data # [...,:] = [rhow, pmix, T, 0.0]

''' Legacy/test functions '''

def conservative_to_pT_WLMA_debug(DTYPE_t vol_energy, DTYPE_t rho_mix,
  DTYPE_t yw, DTYPE_t ya, DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0,
  DTYPE_t c_v_m0, DTYPE_t R_a, DTYPE_t gamma_a):
  ''' Mirror of conservative_to_pT_WLMA with debug info '''
  cdef DTYPE_t ym = 1.0 - yw - ya
  cdef DTYPE_t v_mix = 1.0/rho_mix
  cdef DTYPE_t d = 1.0
  cdef DTYPE_t pmix, T, t, rhow, x, dT
  cdef DTYPE_t psat, rho_satl, rho_satv, d0, d1
  cdef Pair out_pair
  cdef bint is_supercritical, is_mixed
  cdef SatTriple sat_triple
  cdef Derivatives_phir_0_1_2 _phirall_0, _phirall_1
  cdef Derivatives_phi0_0_1_2 _phi0all_0, _phi0all_1
  cdef Derivatives_phir_0_1_2 _phirall
  cdef DTYPE_t dG0dt, dG1dt, dG0dd0, dG1dd0, dG0dd1, dG1dd1, _det, dd0dt, dd1dt
  cdef DTYPE_t dv0dT, dv1dT, du0dT, du1dT, v, vl, vv, dvdT, dT_to_v_bdry
  cdef DTYPE_t partial_dxdT, partial_dxdv, dvdp, dpsatdT, dxdT 
  cdef DTYPE_t uv, ul, dewdT, dedT, demdT, curr_energy, _c_v_w, _u
  cdef DTYPE_t _c1, Z, Z_d, Z_T, d1psi, d2psi, drhowdT, rhom, drhomdT, drhomadT
  cdef DTYPE_t c_v_a = R_a / (gamma_a - 1.0)
  cdef DTYPE_t min_vw = 1e-5
  # Root-finding parameters
  cdef unsigned int i = 0
  cdef unsigned int MAX_NEWTON_ITERS = 64
  cdef DTYPE_t trust_region_size = 1e80
  # Iteration path
  iteration_path = []

  ''' Estimate initial temperature T_init '''
  if rho_mix < 10.0:
    # Due to an issue with flip-flopping for a low-density gas:
    T = ((vol_energy / rho_mix) - yw * (e_gas_ref - c_v_gas_ref * T_gas_ref)) \
    / (yw * c_v_gas_ref + ym * c_v_m0 + ya * c_v_a)
  else:
    # Estimate temperature
    T = ((vol_energy / rho_mix) - yw * (e_w0 - c_v_w0 * T_w0)) \
      / (yw * c_v_w0 + ym * c_v_m0 + ya * c_v_a)
    # if yw >= 0.2:
    #   # Refine estimate using a posteriori approximation of residual
    #   T += poly_T_residual_est(yw, vol_energy, rho_mix)
  ''' Estimate d(T_c) '''
  # One-step Newton approximation of critical-temperature value
  d = 1.0
  out_pair = pure_phase_newton_pair_WLMA(d, Tc, rho_mix, yw, ya, K, p_m0,
    rho_m0, R_a, gamma_a)
  d -= out_pair.first/out_pair.second
  if d <= 0:
    # Recovery
    d = 3.0
    out_pair = pure_phase_newton_pair_WLMA(d, Tc, rho_mix, yw, ya, K, p_m0,
      rho_m0, R_a, gamma_a)
    d -= out_pair.first/out_pair.second
    iteration_path.append((None, d*rhoc, T, "init-recovery"))
  ''' Estimate supercriticality based on energy at Tc '''
  # Quantify approximately supercriticality (rho is approximate, and magma
  #   mechanical energy is neglected)
  is_supercritical = yw * u(d*rhoc, Tc) \
    + ym * (c_v_m0 * Tc) + ya * (c_v_a * Tc) < vol_energy/rho_mix
  ''' Clip T_init based on supercriticality estimate '''
  # Clip initial temperature guess to above or below supercritical
  if is_supercritical and T < Tc + 1.0:
    T = Tc + 1.0
  elif not is_supercritical and T > Tc - 1.0:
    T = Tc - 1.0
  
  iteration_path.append((None, d*rhoc, T, "init"))

  ''' Cold-start compute pressure, water phasic density '''
  if not is_supercritical:
    # Compute saturation properties
    sat_triple = prho_sat(T)
    psat, rho_satl, rho_satv = \
      sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
    # Compute tentative density value
    rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0) \
           - ya * R_a * T / psat)
    # Check rhow > 0 positivity constraint
    if rhow <= 0:
      # Pressurize above saturated liquid
      rhow = rho_satl*(1+1e-6)
      '''# Soft-bound Newton to bring volume close to zero
      iteration_path.append((psat, rhow, T, 'negative-rhow'))
      for i in range(1):
        t = Tc/T
        _phirall_0 = fused_phir_all(rho_satl/rhoc, t)
        _phirall_1 = fused_phir_all(rho_satv/rhoc, t)
        _phi0all_0 = fused_phi0_all(rho_satl/rhoc, t)
        _phi0all_1 = fused_phi0_all(rho_satv/rhoc, t)
        dpsatdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
          * R * (log(rho_satv / rho_satl) + _phirall_1.phir - _phirall_0.phir \
            - t * (_phirall_1.phir_t - _phirall_0.phir_t))
        # Approximate dT using Newton and neglecting dvm/dp * dp/dT 
        min_vw = 1e-5
        dT = -(T - psat / (ya * R_a) 
          * (v_mix - ym * K / rho_m0 / (psat + K - p_m0) - min_vw)) \
          / (1 - dpsatdT / (ya * R_a) * \
          (v_mix - ym * K / rho_m0 / (psat + K - p_m0) - min_vw))
        T += dT
        # Recompute saturation curve
        sat_triple = prho_sat(T)
        psat, rho_satl, rho_satv = \
          sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
        iteration_path.append((psat, rhow, T, 'negative-rhow-corr'))
    if rhow <= 0:
      rhow = 0.1
      iteration_path.append((psat, rhow, T, 'negative-rhow-force'))'''

    if rho_satv <= rhow and rhow <= rho_satl:
      start_case = "start-LV"
      # Accept tentative mixed-phase value of density
      d = rhow / rhoc
      # Compute pressure from sat vapor side
      pmix = psat
    else:
      start_case = "start-subcrit"
      # Select pure vapor or pure liquid phase density
      d = rho_satv/rhoc if (0 < rhow and rhow < rho_satv) else rho_satl/rhoc
      # Refine estimate of d from one Newton iteration
      out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
        rho_m0, R_a, gamma_a)
      d -= out_pair.first/out_pair.second
      # Compute pressure
      pmix = p(d*rhoc, T)
  else:
    start_case = "start-supercrit"
    # Refine estimate of d from one Newton iteration
    out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
      rho_m0, R_a, gamma_a)
    d -= out_pair.first/out_pair.second
    # Compute pressure
    pmix = p(d*rhoc, T)
  # Clip temperatures below the triple point temperature
  if T < 273.16:
    T = 273.16
  
  iteration_path.append((pmix, d*rhoc, T, start_case))

  if d <= 0:
    # Second recovery
    d = 3.0
  for i in range(20):
    out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
      rho_m0, R_a, gamma_a)
    d_inner_step = -out_pair.first/out_pair.second
    d += d_inner_step
    if d_inner_step * d_inner_step < 1e-4 * d * d or d < 0:
      break
  if d < 0:
    d = 2.0
  # Compute pressure
  pmix = p(d*rhoc, T)
  iteration_path.append((None, d*rhoc, T, "start-recovery"))
    
  ''' Perform iteration for total energy condition '''
  for i in range(MAX_NEWTON_ITERS):
    t = Tc/T
    is_mixed = False
    # Set default case string
    case_str = "iter-supercrit"
    
    if T < Tc:
      # Check saturation curves
      sat_triple = prho_sat(T)
      psat, rho_satl, rho_satv = \
        sat_triple.psat, sat_triple.rho_satl, sat_triple.rho_satv
      # Compute volume-sum constrained density value if pm = psat
      rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0) \
           - ya * R_a * T / psat)
      if rho_satv <= rhow and rhow <= rho_satl:
        case_str = "iter-LV"
        is_mixed = True
        # Accept tentative mixed-phase value of pressure, density
        pmix = psat
        d = rhow / rhoc
        # Compute steam fraction (fraction vapor mass per total water mass)
        x = (1.0 / rhow - 1.0 / rho_satl) / (1.0 / rho_satv - 1.0 / rho_satl)
        # Compute temperature update using saturation relation
        d0 = rho_satl/rhoc
        d1 = rho_satv/rhoc
        _phirall_0 = fused_phir_all(d0, Tc/T)
        _phirall_1 = fused_phir_all(d1, Tc/T)
        _phi0all_0 = fused_phi0_all(d0, Tc/T)
        _phi0all_1 = fused_phi0_all(d1, Tc/T)
        ''' Derivatives along Maxwell-constr. level set G(t, d0, d1) = 0 '''
        # Vector components of partial dG/dt = (dG0/dt; dG1/dt)
        dG0dt = d1*_phirall_1.phir_dt + _phirall_1.phir_t + _phi0all_1.phi0_t \
          - d0*_phirall_0.phir_dt - _phirall_0.phir_t - _phi0all_0.phi0_t
        dG1dt = d0*d0*_phirall_0.phir_dt - d1*d1*_phirall_1.phir_dt
        # Vector components of partial dG/d0
        dG0dd0 = -2.0*_phirall_0.phir_d \
          - d0*_phirall_0.phir_dd - _phi0all_0.phi0_d
        dG1dd0 = 1.0 + 2.0 * d0 * _phirall_0.phir_d + d0*d0*_phirall_0.phir_dd
        # Vector components of partial dG/d1
        dG0dd1 = 2.0*_phirall_1.phir_d \
          + d1*_phirall_1.phir_dd + _phi0all_1.phi0_d
        dG1dd1 = -1.0 - 2.0 * d1 * _phirall_1.phir_d - d1*d1*_phirall_1.phir_dd
        # Compute d(d0)/dt and d(d1)/dt constrainted to level set of G
        _det = dG0dd0 * dG1dd1 - dG0dd1 * dG1dd0
        dd0dt = -(dG1dd1 * dG0dt - dG0dd1 * dG1dt) / _det
        dd1dt = -(-dG1dd0 * dG0dt + dG0dd0 * dG1dt) / _det
        # Construct derivatives of volume:
        #   dv0/dt = partial(v0, t) + partial(v0, d) * dd0/dt
        #   dv0/dT = dv0/dt * dt/dT
        dv0dT = -1.0/(rhoc * d0 * d0) * dd0dt * (-Tc/(T*T))
        dv1dT = -1.0/(rhoc * d1 * d1) * dd1dt * (-Tc/(T*T))
        # Construct derivatives of internal energy
        du0dT = R * Tc * (
          (_phirall_0.phir_dt + _phi0all_0.phi0_dt) * dd0dt
          + (_phirall_0.phir_tt + _phi0all_0.phi0_tt)
        ) * (-Tc/(T*T))
        du1dT = R * Tc * (
          (_phirall_1.phir_dt + _phi0all_1.phi0_dt) * dd1dt
          + (_phirall_1.phir_tt + _phi0all_1.phi0_tt)
        ) * (-Tc/(T*T))
        # Construct dx/dT (change in steam fraction subject to mixture
        #   conditions) as partial(x, T) + partial(x, v) * dv/dT
        v  = 1.0/(d*rhoc)
        vl = 1.0/(d0*rhoc)
        vv = 1.0/(d1*rhoc)
        partial_dxdT = ((v - vv) * dv0dT - (v - vl) * dv1dT) \
          / ((vv - vl) * (vv - vl))
        partial_dxdv =  1.0 / (vv - vl) # add partial (x, v)
        # Partial of volume w.r.t. pressure due to volume sum condition
        dvdp = ym / yw * K / rho_m0 / ((psat + K - p_m0)*(psat + K - p_m0)) \
          + ya / yw * R_a * T / (psat*psat)
        # Compute change in allowable volume due to air presence (temp variable)
        #   This is a partial derivative of the volume sum condition
        dvdT = - ya / yw * (R_a / psat)
        # Compute saturation-pressure-temperature slope
        dpsatdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
          * R * (log(rho_satv / rho_satl) + _phirall_1.phir - _phirall_0.phir \
            - t * (_phirall_1.phir_t - _phirall_0.phir_t))
        dxdT = partial_dxdT + partial_dxdv * (dvdp * dpsatdT + dvdT)
        # Construct de/dT for water:
        #   de/dT = partial(e,x) * dx/dT + de/dT
        uv = R * Tc * (_phirall_1.phir_t + _phi0all_1.phi0_t)
        ul = R * Tc * (_phirall_0.phir_t + _phi0all_0.phi0_t)
        dewdT = (uv - ul) * dxdT + x*du1dT + (1.0-x)*du0dT
        # Construct de/dT for mixture, with -pdv of magma as (p * -dvdp * dpdT)
        dedT = yw * dewdT + ym * c_v_m0 + ya * c_v_a \
          + ym * (psat * K / rho_m0 / ((psat + K - p_m0)*(psat + K - p_m0)) * dpsatdT)

        ''' Compute Newton step for temperature '''
        curr_energy = yw * (x * uv + (1.0 - x) * ul) \
          + ym * (c_v_m0 * T + magma_mech_energy(psat, K, p_m0, rho_m0)) \
          + ya * (c_v_a * T)
        dT = -(curr_energy - vol_energy/rho_mix)/dedT
        if rho_mix < 10: # Size limiter for sparse gas
          trust_region_size = 1e2
          dT = -(curr_energy - vol_energy/rho_mix)/(dedT + 1e6/1e2)

        ''' Estimate saturation state in destination temperature '''
        # Vapor-boundary overshooting correction
        dvdT = (vv - vl) * dxdT + x*dv1dT + (1.0-x)*dv0dT
        dT_to_v_bdry = (vv - v) / (dvdT - dv1dT)
        if dT > 0 and dT_to_v_bdry > 0 and dT > dT_to_v_bdry:
          dT = 0.5 * (dT + dT_to_v_bdry)
        # Newton-step temperature
        T += dT
      else:
        # Annotate subcritical but pure-phase case
        case_str = "iter-subcrit"
        pass
        '''if d < 0:
          case_str = "iter-subcrit-forced-positivity"
          d = rho_satl*(1+1e-6) / rhoc
        else:
          # Update d based on computed rhow value
          d = rhow / rhoc'''
    if not is_mixed:
      # Compute phasic states using current d, T
      rhow = d*rhoc
      if d <= 0 and T < Tc:
        # Negative density recovery attempt
        sat_triple = prho_sat(T)
        d = sat_triple.rho_satl / rhoc
        rhow = sat_triple.rho_satl
      elif d <= 0:
        # Supercritical recovery attempt (cool mixture to reduce air volume)
        T = Tc - 1.0
        sat_triple = prho_sat(T)
        d = sat_triple.rho_satl / rhoc
        rhow = sat_triple.rho_satl
      rhom = ym / (1.0 / rho_mix - yw / rhow - ya * (R_a * T) / pmix)
      # Evaluate phir derivatives (cost-critical)
      _phirall = fused_phir_all(d, Tc/T)
      # Compute pure-phase heat capacity
      _c_v_w = -t * t * R * (_phirall.phir_tt + phi0_tt(d, t))
      # Compute pure-phase energy
      _u = t * R * T * (_phirall.phir_t + phi0_t(d, t))

      ''' Compute slopes '''
      # Compute compressibility, partial of compressibility
      Z = 1.0 + d * _phirall.phir_d
      Z_d = _phirall.phir_d + d * _phirall.phir_dd
      Z_T = d * _phirall.phir_dt * (-Tc / (T*T))
      # Compute intermediates
      _c1 = (v_mix * d - (yw + ya*R_a/(Z *R)) / rhoc)
      # Compute derivatives of pressure equilibrium level set function 
      #  R * T *psi(d, T) = 0 constrains d(t).
      #  Note the difference with the Newton slope-pair function.
      #  Here we use the form of equation with T appearing only once
      #  d1psi is d/dd (nondim) and d2psi is d/dT (dim) of (R * T * psi(d,T)).
      d1psi = (v_mix + ya * R_a / R / rhoc * Z_d / (Z * Z)) \
        * (Z * d * rhoc * R * T + K - p_m0) - K * ym / rho_m0 \
        + _c1 * (rhoc * R * T * (Z + d * Z_d))
      d2psi = (ya * R_a / R / rhoc * Z_T / (Z * Z)) \
        * (Z * d * rhoc * R * T + K - p_m0) \
        + _c1 * (Z + T * Z_T) * d * rhoc * R
      # Compute density-temperature slope under the volume addition constraint
      drhowdT = -d2psi / d1psi * rhoc
      # Compute density-temperature slope for m state
      drhomdT = -(rhom*rhom/ ym) * (yw * (drhowdT / (rhow*rhow)) 
        + ya * (R_a * T / (pmix*pmix)) * (
          - pmix / T  +
          rhoc * R * ((1.0 + d * _phirall.phir_d) * d 
            - d * d * _phirall.phir_dt * Tc / T)
          + (1.0 + 2.0 * d * _phirall.phir_d + d * d * _phirall.phir_dd)
            * rhoc * R * T * drhowdT / rhoc))
      # Compute water energy-temperature slope
      dewdT = _c_v_w + R * Tc / rhoc * _phirall.phir_dt * drhowdT
      # Compute magma energy-temperature slope (c_v dT + de/dv * dv, v = v(T))
      demdT = c_v_m0 \
        + pmix / (rhom*rhom) * drhomdT # - p dv = p (drho) / rho^2
      # Compute mixture energy-temperature slope
      dedT = yw * dewdT + ym * demdT + ya * c_v_a
      ''' Compute Newton step for temperature '''
      curr_energy = yw * _u \
        + ym * (c_v_m0 * T + magma_mech_energy(pmix, K, p_m0, rho_m0)) \
        + ya * c_v_a * T
      # Temperature migration
      dT = -(curr_energy - vol_energy/rho_mix)/dedT
      # Limit to dT <= 100
      dT *= min(1.0, 50.0/(1e-16+fabs(dT)))
      # Newton-step temperature
      T += dT
      # Newton-step d (cross term in Jacobian)
      d += drhowdT/rhoc * dT
      out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
        rho_m0, R_a, gamma_a)
      d -= out_pair.first/out_pair.second
      # Update pressure
      pmix = p(d*rhoc, T)
      # Update water phasic density
      rhow = d * rhoc
    # Clip temperatures below the triple point temperature
    T = max(T, 273.16)
    iteration_path.append((pmix, rhow, T, case_str))
    if dT * dT < 1e-9 * 1e-9:
      break
  return iteration_path, TriplerhopT(rhow, pmix, T)

def conservative_to_pT_WLM_debug(DTYPE_t vol_energy, DTYPE_t rho_mix,
  DTYPE_t yw, DTYPE_t K, DTYPE_t p_m0,
  DTYPE_t rho_m0, DTYPE_t c_v_m0) -> DTYPE_t:
  ''' Map conservative mixture variables to pressure, temperature.
  Python-like implementation with debug values. '''
  iteration_path = []

  # Evaluate dependent variables
  ym = 1.0 - yw
  v_mix = 1.0/rho_mix
  
  # Due to an issue with flip-flopping for a low-density gas:
  if rho_mix < 10.0:
    T = ((vol_energy / rho_mix) - yw * (e_gas_ref - c_v_gas_ref * T_gas_ref)) \
    / (yw * c_v_gas_ref + ym * c_v_m0)
  else:
    # Estimate temperature
    T = ((vol_energy / rho_mix) - yw * (e_w0 - c_v_w0 * T_w0)) \
      / (yw * c_v_w0 + ym * c_v_m0)

  ''' Critical quantification: critical or not? '''
  # One-step Newton approximation of critical-temperature value
  d = 1.0
  out_pair = pure_phase_newton_pair(d, Tc, rho_mix, yw, K, p_m0, rho_m0)
  d -= out_pair.first/out_pair.second
  # Quantify approximately supercriticality
  is_supercritical = yw * u(d*rhoc, Tc) \
    + ym * (c_v_m0 * Tc) < vol_energy/rho_mix
  # Clip initial temperature guess from linear energy
  if is_supercritical and T < Tc + 1.0:
    T = Tc + 1.0
  elif not is_supercritical and T > Tc - 1.0:
    T = Tc - 1.0

  # print(f"Supercritical?: {is_supercritical}")
  # print(f"Initial T = {T}")

  ''' Cold-start compute pressure, water phasic density '''
  if not is_supercritical:
    # Compute saturation properties
    sat_info = prho_sat(T)
    psat, rho_satl, rho_satv = \
      sat_info.psat, sat_info.rho_satl, sat_info.rho_satv
    # Compute tentative density value
    rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0))
    if rho_satv <= rhow and rhow <= rho_satl:
      # Accept tentative mixed-phase value of density
      d = rhow / rhoc
      # Compute pressure from sat vapor side
      pmix = psat
      start_case = "start-LV"
    else:
      # Select pure vapor or pure liquid phase density
      d = rho_satv/rhoc if rhow < rho_satv else rho_satl/rhoc
      # Refine estimate of d from one Newton iteration
      out_pair = pure_phase_newton_pair(d, T, rho_mix, yw, K, p_m0, rho_m0)
      d -= out_pair.first/out_pair.second
      # Compute pressure
      pmix = p(d*rhoc, T)
      start_case = "start-subcrit"
  else:
    # Refine estimate of d from one Newton iteration
    out_pair = pure_phase_newton_pair(d, T, rho_mix, yw, K, p_m0, rho_m0)
    d -= out_pair.first/out_pair.second
    # Compute pressure from
    pmix = p(d*rhoc, T)
    start_case = "start-supercrit"

  # Clip temperatures below the triple point temperature
  if T < 273.16:
    T = 273.16
    
  # print(f"Start case is {start_case}")
  iteration_path.append((pmix, d*rhoc, T, start_case))

  ''' Perform iteration for total energy condition '''
  for i in range(64):
    t = Tc/T
    is_mixed = False
    # Set default case string
    case_str = "iter-supercrit"
    
    if T < Tc:
      sat_info = prho_sat(T)
      psat, rho_satl, rho_satv = \
        sat_info.psat, sat_info.rho_satl, sat_info.rho_satv
      # Compute volume-sum constrained density value if pm = psat
      rhow = yw / (v_mix - ym * K / rho_m0 / (psat + K - p_m0))
      # iteration_path.append((psat, rhow, T, "interm"))
      # d = rhow/rhoc
      if rho_satv <= rhow and rhow <= rho_satl:
        is_mixed = True
        pmix = psat
        # Accept tentative mixed-phase value of density
        d = rhow / rhoc
        # Compute steam fraction (fraction vapor mass per total water mass)
        x = (1.0 / rhow - 1.0 / rho_satl) / (1.0 / rho_satv - 1.0 / rho_satl)
        # Compute temperature update using saturation relation
        # _phir_all_v = float_phi_functions.fused_phir_all(rho_satv/rhoc, Tc/T)
        # _phir_all_l = float_phi_functions.fused_phir_all(rho_satl/rhoc, Tc/T)
        d0 = rho_satl/rhoc
        d1 = rho_satv/rhoc
        _phirall_0 = fused_phir_all(d0, Tc/T)
        _phirall_1 = fused_phir_all(d1, Tc/T)
        _phi0all_0 = fused_phi0_all(d0, Tc/T)
        _phi0all_1 = fused_phi0_all(d1, Tc/T)
        # float_phi_functions.prho_sat(500)

        ''' Derivatives along Maxwell-constr. level set G(t, d0, d1) = 0 '''
        # Vector components of partial dG/dt = (dG0/dt; dG1/dt)
        dG0dt = d1*_phirall_1.phir_dt + _phirall_1.phir_t + _phi0all_1.phi0_t \
          - d0*_phirall_0.phir_dt - _phirall_0.phir_t - _phi0all_0.phi0_t
        dG1dt = d0*d0*_phirall_0.phir_dt - d1*d1*_phirall_1.phir_dt
        # Vector components of partial dG/d0
        dG0dd0 = -2.0*_phirall_0.phir_d \
          - d0*_phirall_0.phir_dd - _phi0all_0.phi0_d
        dG1dd0 = 1.0 + 2.0 * d0 * _phirall_0.phir_d + d0*d0*_phirall_0.phir_dd
        # Vector components of partial dG/d1
        dG0dd1 = 2.0*_phirall_1.phir_d \
          + d1*_phirall_1.phir_dd + _phi0all_1.phi0_d
        dG1dd1 = -1.0 - 2.0 * d1 * _phirall_1.phir_d - d1*d1*_phirall_1.phir_dd
        # Compute d(d0)/dt and d(d1)/dt constrainted to level set of G
        _det = dG0dd0 * dG1dd1 - dG0dd1 * dG1dd0
        dd0dt = -(dG1dd1 * dG0dt - dG0dd1 * dG1dt) / _det
        dd1dt = -(-dG1dd0 * dG0dt + dG0dd0 * dG1dt) / _det
        # Construct derivatives of volume:
        #   dv0/dt = partial(v0, t) + partial(v0, d) * dd0/dt
        #   dv0/dT = dv0/dt * dt/dT
        dv0dT = -1.0/(rhoc * d0 * d0) * dd0dt * (-Tc/(T*T))
        dv1dT = -1.0/(rhoc * d1 * d1) * dd1dt * (-Tc/(T*T))
        # Construct derivatives of internal energy
        du0dT = R * Tc * (
          (_phirall_0.phir_dt + _phi0all_0.phi0_dt) * dd0dt
          + (_phirall_0.phir_tt + _phi0all_0.phi0_tt)
        ) * (-Tc/(T*T))
        du1dT = R * Tc * (
          (_phirall_1.phir_dt + _phi0all_1.phi0_dt) * dd1dt
          + (_phirall_1.phir_tt + _phi0all_1.phi0_tt)
        ) * (-Tc/(T*T))
        # Construct dx/dT (change in steam fraction subject to mixture
        #   conditions) as partial(x, T) + partial(x, v) * dv/dT
        v = 1.0/(d*rhoc)
        vl = 1.0/(d0*rhoc)
        vv = 1.0/(d1*rhoc)
        partial_dxdT = ((v - vv) * dv0dT - (v - vl) * dv1dT) \
          / ((vv - vl) * (vv - vl))
        partial_dxdv =  1.0 / (vv - vl) # add partial (x, v)
        dvdp = (1.0 - yw) / yw * K / rho_m0 / (
          (psat + K - p_m0)*(psat + K - p_m0))
        # Compute saturation-pressure-temperature slope
        dpsatdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
          * R * (log(rho_satv / rho_satl) + _phirall_1.phir - _phirall_0.phir \
            - t * (_phirall_1.phir_t - _phirall_0.phir_t))
        dxdT = partial_dxdT + partial_dxdv * dvdp * dpsatdT
        # Construct de/dT for water:
        #   de/dT = partial(e,x) * dx/dT + de/dT
        uv = R * Tc * (_phirall_1.phir_t + _phi0all_1.phi0_t)
        ul = R * Tc * (_phirall_0.phir_t + _phi0all_0.phi0_t)
        dewdT = (uv - ul) * dxdT + x*du1dT + (1.0-x)*du0dT
        # Construct de/dT for mixture, neglecting pdv work of magma
        dedT = yw * dewdT + ym * c_v_m0

        ''' Compute Newton step for temperature '''
        curr_energy = yw * u(rhow, T) \
          + ym * (c_v_m0 * T + magma_mech_energy(psat, K, p_m0, rho_m0))
        dT = -(curr_energy - vol_energy/rho_mix)/dedT
        if rho_mix < 10: # Size limiter for sparse gas
          trust_region_size = 1e2
          dT = -(curr_energy - vol_energy/rho_mix)/(dedT + 1e6/1e2)

        ''' Estimate saturation state in destination temperature '''
        # Vapor-boundary overshooting correction
        dvdT = (vv - vl) * dxdT + x*dv1dT + (1.0-x)*dv0dT
        dT_to_v_bdry = (vv - v) / (dvdT - dv1dT)
        if dT > 0 and dT_to_v_bdry > 0 and dT > dT_to_v_bdry:
          dT = 0.5 * (dT + dT_to_v_bdry)
        # Newton-step temperature
        T += dT

        
        # Compute saturation-pressure-temperature slope
        # dpdT = rho_satv * rho_satl / (rho_satv - rho_satl) \
        #   * R * (log(rho_satv / rho_satl) + _phir_all_v["phir"] \
        #     - _phir_all_l["phir"] \
        #     - t * (_phir_all_v["phir_t"] - _phir_all_l["phir_t"]))
        # # Directly use pressure mismatch as solution target
        # rhom = ym / (1.0 / rho_mix - yw / rhow)
        # pm = p_m0 + K * (rhom - rho_m0) / rho_m0
        # pw = p
        # T += (pm - pw) /dpdT
        # is_mixed = True
        case_str = "iter-LV"
      else:
        # Annotate subcritical but pure-phase case
        case_str = "iter-subcrit"
        # Update d based on computed rhow value
        # d = rhow / rhoc
        pass
    
    if not is_mixed:
      # Compute phasic states using current T and previous iteration d
      rhow = d*rhoc
      rhom = ym / (1.0 / rho_mix - yw / rhow)
      ''' Call cost-critical functions '''
      _phir_all = fused_phir_all(d, Tc/T)
      # Compute heat capacity
      _c_v_w = c_v(rhow, T)

      ''' Compute slopes '''
      # Compute intermediates
      _c1 = (v_mix * d - yw / rhoc)
      # Compute compressibility, partial of compressibility
      Z = 1.0 + d * _phir_all.phir_d
      Z_d = _phir_all.phir_d + d * _phir_all.phir_dd
      # Compute derivatives of pressure equilibrium level set function
      #  R * T *psi(d, T) = 0 constrains d(t)
      #  Here we use the form of equation with T appearing only once
      #  d1psi is d/dd and d2psi is d/dT of (R * T * psi(d,T)).
      d1psi = v_mix * (Z * d * rhoc * R * T + K - p_m0) - K * ym / rho_m0 \
        + _c1 * (rhoc * R * T * (Z + d * Z_d))
      d2psi = _c1 * Z * d * rhoc * R # TODO: update here and in non _debug from Z to (Z + T * Z_T)
      # Compute density-temperature slope under the volume addition constraint
      drhowdT = -d2psi / d1psi * rhoc
      # Compute density-temperature slope for m state
      drhomdT = -yw / ym * (rhom / rhow) * (rhom / rhow) * drhowdT
      # Compute pressure-temperature slope under the volume addition constraint
      # dpdT = rhoc * R * (
      #   (2.0 * d + 2.0 * d * _phir_all["phir_d"] 
      #     + d * d * _phir_all["phir_dd"]) * drhowdT * T
        # + d * d * (1.0 + _phir_all["phir_d"]))
      # Compute water energy-temperature slope
      dewdT = _c_v_w + R * Tc / rhoc * _phir_all.phir_dt * drhowdT
      # Compute magma energy-temperature slope
      demdT = c_v_m0 \
        + pmix / (rhom*rhom) * drhomdT
      # Compute mixture energy-temperature slope
      dedT = yw * dewdT + ym * demdT

      ''' Compute Newton step for temperature '''
      curr_energy = yw * u(rhow, T) \
        + ym * (c_v_m0 * T + magma_mech_energy(pmix, K, p_m0, rho_m0))
      # Temperature migration
      dT = -(curr_energy - vol_energy/rho_mix)/dedT
      # Penalize large steps (trust region)
      trust_region_size = 1e8
      dT = -(curr_energy - vol_energy/rho_mix)/(dedT + 1e6/trust_region_size)
      # Limit to dT <= 100
      dT *= min(1.0, 50.0/(1e-16+fabs(dT)))

      # Extrapolation cut
      # if case_str == "iter-subcrit":

      # if dT*dT > trust_region_size*trust_region_size:
        # dT = trust_region_size*np.sign(dT)
      # Newton-step temperature
      T += dT
      # Newton-step d (cross term in Jacobian)
      d += drhowdT/rhoc * dT
      out_pair = pure_phase_newton_pair(d, T, rho_mix, yw, K, p_m0, rho_m0)
      d -= out_pair.first/out_pair.second
      # Update pressure
      pmix = p(d*rhoc, T)
      # Update water phasic density
      rhow = d * rhoc
    
    # Clip temperatures below the triple point temperature
    T = max(T, 273.16)

    # Append output
    iteration_path.append((pmix, rhow, T, case_str))

    if dT * dT < 1e-9 * 1e-9:
      break

    # Legacy debug
    # d, absstep = mixtureWLM._itersolve_d(rho_mix, yw, T, d_init=d_reinit)
    # print(yw / (rhoc * d) + ym / rhom, 1/rho_mix, val)
    # print()
    # print(Z * rhow * R * T - K/rho_m0 * rhom + K - p_m0)
    # print(rhom)
    # print(ym / rhom + yw / rhow, 1/rho_mix)
    # print(p, p_m0 + K / rho_m0 * (rhom - rho_m0), val,
    #   (v_mix*d - yw / rhoc)*(p - (p_m0 + K / rho_m0 * (rhom - rho_m0))),
    #   (1 + d*phir_d) * d * rhoc * R * T \
    #   - K/rho_m0 * ym / (v_mix - yw / (d * rhoc)) + K - p_m0,
    #   - K/rho_m0 * ym / (v_mix - yw / (d * rhoc)) + K - p_m0)
    # u_mix = yw * float_phi_functions.u(rhow, T) \
    #   + ym * (c_v_m0 * T + magma_mech_energy(p))
    # print(f"Spec energy: {u_mix}, {vol_energy/rho_mix}")
    # print(f"Spec vol: {yw /rhow + ym / rhom}, {1.0/rho_mix}")
    # print(f"Pressure: {p} and {p_m0 + K / rho_m0 * (rhom - rho_m0)}, " +
    #       f"{p_target}")

  return iteration_path