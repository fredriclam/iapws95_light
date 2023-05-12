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
cpdef kernel2_WLMA(DTYPE_t alpha_w, DTYPE_t T,
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

''' Newton-line-search '''

"""@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef Pair iterate_backtrack_box(DTYPE_t U):
  ''' Backtracked Newton with box bounds '''

  # Compute Newton variables
  f, C, D, lv_hypothetical, p = kernel4_WLMA(
    d, T, pr, Tr, vol_energy, rho_mix, yw, ya, K,
    p_m0, rho_m0, c_v_m0, R_a, gamma_a)
  # Solve 4x4 system for Newton step
  step = -np.linalg.solve(C * D, f)

  ''' Box bound scaling '''
  U_min = np.array([1e-5, 273.16, 7e3, 173.16])  # Permit density down to 7 kPa (d = 0.01)
  U_max = np.array([3.882, 2200.0, 1e9, 2200.0]) # Permit density up to 1 GPa

  use_type = "projection"
  if use_type == "scaling": # ''' Scale step to land on boundary '''
    step_size_factor = 1.0
    for i in range(len(U)):
      if U[i] + step[i] < U_min[i]:
        # i-th min bound is active 
        step_size_factor = np.minimum(step_size_factor, (U_min[i] - U[i])/ step[i])
      if U[i] + step[i] > U_max[i]:
        # i-th max bound is active
        step_size_factor = np.minimum(step_size_factor, (U_max[i] - U[i])/ step[i])
    step *= step_size_factor

  # Greed ratio in (0,1) for Armijo conditions
  greed_ratio = 0.9
  # Step reduction factor in (0, 1) used each iteration
  step_reduction_factor = 0.7
  # Set the step size factor (==1.0 the first time Armijo condition is checked)
  a = 1.0 / step_reduction_factor
  # Backtracking
  for i in range(24):
    a *= step_reduction_factor
    # Compute tentative step
    U_next = U + a * step
    # Project tentative step to feasible set
    box_active = False
    if use_type == "projection": # ''' Type B: Projection '''
      for i in range(len(U)):
        if U_next[i] < U_min[i]:
          # i-th min bound is active
          U_next[i] = U_min[i] 
          box_active = True
        if U_next[i] > U_max[i]:
          # i-th max bound is active
          U_next[i] = U_max[i]
          box_active = True
    f_along_line = kern(*U_next)[0]
    f_along_line, _, _, _, _ = kernel4_WLMA(
      d, T, pr, Tr, vol_energy, rho_mix, yw, ya, K,
      p_m0, rho_m0, c_v_m0, R_a, gamma_a)
    
    if not box_active:
      # Check validity
      if np.any(np.isnan(f_along_line)):
        a *= step_reduction_factor
        continue
      # c * grad(norm(f)) dot J^{-1} f = c || f ||, since Newton sends model to 0
      tt = greed_ratio * np.linalg.norm(f)
      # LHS of Armijo condition: improvement is at least a portion of a * |f|
      armijoLHS = np.linalg.norm(f) - np.linalg.norm(f_along_line) - a * tt
      if armijoLHS >= 0:
        break
    else:
      # No backtracking for boundary-projected steps
      break
  return U_next, f_along_line"""

"""@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# type TriplerhopT
cpdef conservative_to_pT_WLMA_NBLS(DTYPE_t vol_energy,
  DTYPE_t rho_mix, DTYPE_t yw, DTYPE_t ya, DTYPE_t K, DTYPE_t p_m0,
  DTYPE_t rho_m0, DTYPE_t c_v_m0, DTYPE_t R_a, DTYPE_t gamma_a):
  ''' Map conservative WLMA mixture variables to pressure, temperature.
  WLMA model = {water, linearized magma, air}.
  NLS = Newton + Backtracking Line Search. '''
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

  ''' Estimate initial temperature T_init:
  Linear estimator stepwise parametrized by mixture density.
    Low density mixture (rho_mix < 10.0): assume water is vapour.
    Else: assume water is liquid. '''
  if rho_mix < 10.0:
    # Due to an issue with flip-flopping for a low-density gas:
    T = ((vol_energy / rho_mix) - yw * (e_gas_ref - c_v_gas_ref * T_gas_ref)) \
    / (yw * c_v_gas_ref + ym * c_v_m0 + ya * c_v_a)
  else:
    # Estimate temperature
    T = ((vol_energy / rho_mix) - yw * (e_w0 - c_v_w0 * T_w0)) \
      / (yw * c_v_w0 + ym * c_v_m0 + ya * c_v_a)

  ''' Estimate d(T_c) '''
  # One-step Newton approximation of critical-temperature value
  d = 1.0
  out_pair = pure_phase_newton_pair_WLMA(d, Tc, rho_mix, yw, ya, K, p_m0,
    rho_m0, R_a, gamma_a)
  d -= out_pair.first/out_pair.second
  # Attempt to recover invalid values of d
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

  return 

  ''' With initial guess, perform Newton with backtracking '''


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
  return TriplerhopT(rhow, pmix, T)"""

"""U, f = iterate_backtrack_box(U)

# Set best seed
U0 = np.array([1, 400, 100e6, 300])

# Redefine kernel
vol_energy = mg_vol_energy[i,j,k,l]
rho_mix = mg_rho_mix[i,j,k,l]
yw = mg_yw[i,j,k,l]
ya = mg_ya[i,j,k,l]
kern = lambda d, T, pr, Tr: float_mix_functions.kernel4_WLMA(d, T, pr, Tr,
  vol_energy, rho_mix, yw, ya, K, p_m0, rho_m0, c_v_m0, R_a, gamma_a)

# Log map
# V = np.log(U0)
U = U0
# Newton iteration
for _k in range(128):
  # Newton step with backtracking
  U, f = iterate_backtrack_box(U)
  fnorm = np.linalg.norm(f)
  if fnorm < 1e-12:
    break
  if np.any(np.isnan(f)):
    fnorm = 1e10
    print(kern(*U), vol_energy, rho_mix, yw)
    raise Exception("Nan encountered, dumping")

# Compute final U = [d, T, pr, Tr]
T_calc = U[1]
p_calc = float_mix_functions.p(U[0] * mixtureWLMA.rhoc, U[1])
mg_p_calc[i,j,k,l], mg_T_calc[i,j,k,l] = p_calc, T_calc"""



''' Vectorizing wrappers '''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# @cython.cdivision(True)
def vec_conservative_to_pT_WLMA(np.ndarray vol_energy, np.ndarray rho_mix,
  np.ndarray yw, np.ndarray ya, DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0,
  DTYPE_t c_v_m0, DTYPE_t R_a, DTYPE_t gamma_a) -> np.ndarray:
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

  for i in range(N):
    # TODO: replace binary decision
    if data[4*i+3] > 1e-4:
      # Treat immediately as air-only
      _p = data[4*i] * (gamma_a - 1.0)        # en / (gamma-1)
      data[4*i+2] = _p / (data[4*i+1] * R_a)  # p / (rho R)
      data[4*i+1] = _p
      data[4*i+3] = sqrt(gamma_a * R_a * data[4*i+2])
      data[4*i] = 0.0
    else:  
      # Use packed inputs (vol_energy, rho_mix, yw, ya)
      out_triple = conservative_to_pT_WLMA(
        data[4*i], data[4*i+1], data[4*i+2], data[4*i+3],
        K, p_m0, rho_m0, c_v_m0, R_a, gamma_a)
      # Reuse data track as output
      data[4*i]   = out_triple.rhow
      data[4*i+1] = out_triple.p
      data[4*i+2] = out_triple.T
      # TODO: replace with direct postprocess with access to last phir_*
      # TODO: replace sound speed computation for mixture
      data[4*i+3] = sound_speed(out_triple.rhow, out_triple.T)
  # return rhow, pmix, T
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