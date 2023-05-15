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
cdef OutKernel2 kernel2_WLMA(DTYPE_t alpha_w, DTYPE_t T, WLMAParams params):
  ''' Return compatibility equations and derivatives. '''

  # Zealous unpacking
  cdef DTYPE_t vol_energy, rho_mix, yw, ya, K, \
    p_m0, rho_m0, c_v_m0, R_a, gamma_a
  vol_energy, rho_mix, yw, ya,  K, p_m0, rho_m0, c_v_m0, R_a, gamma_a = \
    params.vol_energy, params.rho_mix, params.yw, params.ya, params.K, \
    params.p_m0, params.rho_m0, params.c_v_m0, params.R_a, params.gamma_a

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
  # Compute step
  _temp = -np.linalg.solve(J, f)
  
  return OutKernel2(f[0], f[1], _temp[0], _temp[1], pmix, region_type)

''' Robust WLMA pT function '''

# Define linearization point
cdef DTYPE_t ref_T = 300
cdef DTYPE_t ref_ew = u(996, ref_T)
cdef DTYPE_t ref_c_v = c_v(996, ref_T)
cdef int NUM_ITER_BACKTRACK = 16
cdef int NUM_ITER_PHASE_BACKTRACK = 11

cdef struct OutIterate:
  DTYPE_t U0
  DTYPE_t U1
  DTYPE_t fnorm

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

"""@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef TriplerhopT conservative_to_pT_WLMA_bn(DTYPE_t vol_energy, DTYPE_t rho_mix,
  DTYPE_t yw, DTYPE_t K, DTYPE_t p_m0, DTYPE_t rho_m0, DTYPE_t c_v_m0)
  ''' Map from conservative to pressure-temperature variables for WLMA model.
  Uses bisection + Newton (bn) to compute initial guesses based on the
  approximate regime of the inputs (approximate the phase of water) followed
  by a backtracking Newton search for (p, T) solving the energy fraction and
  volume fraction equations. The range of validity is designed in terms of mass
  fractions, water density, and temperature as
    1e-7 <= yw <= 1-1e-7,
    1e-7 <= ya <= 1-1e-7,
    0.1 <= rhow <= 1050,
    280 <= T <= 1500.
  The two constraint equations are as follows. 
    let ef_w = y_w * e_w/e_mix: energy fraction of water
    let ef_r = y_r * e_r/e_mix: energy fraction of residual components
    eqn ef_w + ef_r - 1 == 0
    eqn yw/vmix / (rhoc*d) + yr/vmix * vr(pr, Tr) - 1 == 0.
  '''
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
  
  '''
  # Compute dependents
  t = Tc / T
  ym = 1.0 - yw - ya
  # Compute water phase density as dependent
  rhow = rho_mix * yw / alpha_w
  # Set bound on rhow based on pressure bound [, 1 GPa]
  rhow = np.clip(rhow, 1e-9, 1260)
  d = rhow / rhoc
def iter_solve1(vol_energy, rho_mix, yw, ya, dump_state=False):
  msg_list = []

  if vol_energy * rho_mix == 0:
    return 0.0, 0.0, msg_list

  kern = lambda U0, U1: float_mix_functions.kernel2_WLMA(
    U0, U1,
    vol_energy, rho_mix, yw, ya,
    K, p_m0, rho_m0, c_v_m0, R_a, gamma_a)
  
  # ym = 1.0 - (ya + yw)

  # Lazy boundary snapping
  min_y = 1e-14
  if yw < min_y:
    yw = min_y
  ym = 1.0 - (ya + yw)
  if ym < min_y:
    ym = min_y
  # Remove mass fraction from air
  if yw + ya + ym > 1.0:
    ya = 1.0 - (yw + ym)
  
  def p_LMA(T, yw, ya, ym, get_phi=False):
    ''' Compute volume fraction of sum of gases (also called porosity). '''
    # Define useful quantities: additive constant to magma EOS
    sym1 = K - p_m0
    # Define partial pressure of gas mixture
    sym2 = (ya * R_a + yw * 0 * mixtureWLMA.R) * rho_mix * T
    # Define negative b (from quadratic formula)
    b = (sym1 - sym2 - K / rho_m0 * ym * rho_mix)
    phi =  0.5 / sym1 * (b + np.sqrt(b*b + 4*sym1*sym2))

    if not get_phi:
      # Return pressure
      return (ya * R_a + yw * 0 * mixtureWLMA.R) * rho_mix * T \
        + (1.0 - phi) * (p_m0 - K) + (K / rho_m0 * ym * rho_mix)
    else:
      return (ya * R_a + yw * 0 * mixtureWLMA.R) * rho_mix * T \
        + (1.0 - phi) * (p_m0 - K) + (K / rho_m0 * ym * rho_mix), phi
  
  # if ym < 0:
  #   return 0.0, 0.0, msg_list
  
  # # Include latent heat is liquid state must be rejected
  # if yw * rho_mix <= rhoc:
  #   e_lv = ref_e_lv
  # else:
  #   e_lv = 0

  # Compute temperature guess
  e_lv = 0
  _e_diff = vol_energy/rho_mix - yw * (ref_ew + e_lv) \
    - ya * R_a / (gamma_a - 1.0) * ref_T - ym * c_v_m0 * ref_T
  _T_init = ref_T + _e_diff \
    / (yw * ref_c_v + ya * R_a / (gamma_a - 1.0) + ym * c_v_m0)
  if _T_init < 273.16:
    _T_init = 273.16

  # Sparse water limit solution
  # if yw < 1e-5 and ya > 1e-9:
  #   # Non-iterative pressure
  #   _p_LMA = p_LMA(_T_init, yw, ya, ym)
  #   _vol_occupied = ya * R_a * _T_init / _p_LMA \
  #     + (1.0 - (ya + yw)) / rho_m0 / (1.0 + (_p_LMA - p_m0) / K)
  #   _alphaw_test = np.clip((1.0/rho_mix - _vol_occupied) * rho_mix, 0.0, rho_mix)
  #   if False: # _alphaw_test > 0:
  #     # Add volumefrac-weighted pressure of water
  #     _p_WLMA = _p_LMA * (1.0 - _alphaw_test) \
  #       + _alphaw_test * float_mix_functions.p(
  #         rho_mix * yw / _alphaw_test, _T_init)
  #   else:
  #     _p_WLMA = _p_LMA
  #   return _p_WLMA, _T_init, msg_list
  
  # Weighted pressure
  # alphaw_test = np.clip((1.0/rho_mix - vol_test)*rho_mix, 0.0, rho_mix)
  # rhow_test = rho_mix * yw / alphaw_test
  # kern(alphaw_test, 3.90948974e+02)
  # p_LMA(3.90948974e+02, yw, ya, 1.0 - ya - yw) * (1 - alphaw_test) \
  #   +alphaw_test * float_mix_functions.p(rhow_test, 3.90948974e+02), \
  #   p_LMA(3.90948974e+02, yw, ya, 1.0 - ya - yw), \
  #   solution["p"]

  # Compute stable water volume fraction guess
  if _T_init < Tc:
    _out = float_mix_functions.prho_sat(_T_init)
    # Hypothetical LV coexistence
    psat = _out["psat"]
    h_rhoa = psat / (R_a * _T_init)
    h_rhom = rho_m0 * (1.0 + (psat - p_m0) / K)
    h_vw = (1/rho_mix - ya / h_rhoa + ym / h_rhom) / yw
    h_x = (h_vw - 1.0/_out["rho_satl"]) / (1.0/_out["rho_satv"] - 1.0/_out["rho_satl"])
    if 0 <= h_x and h_x <= 1:
      # Accept saturation
      _alphaw_init =  yw * rho_mix * h_vw
    elif h_x < 0:
      # Liquid-like
      _alphaw_init = yw * rho_mix / 1000.0
    else:
      # Vapor-like
      _alphaw_init = yw * rho_mix / _out["rho_satv"]
  else:
    _alphaw_init = yw * rho_mix / rhoc   
  if _alphaw_init < 0 or _alphaw_init > 1:
    _alphaw_init = 0.5

  # Compute volume constraint minimizer
  _a, _b = 1e-7, 1.0
  _bisection_count = 16

  for i in range(_bisection_count):
    _m = 0.5 * (_a + _b)
    if kern(*np.array([_m, _T_init]))[0][1] > 0:
      _b = _m
    else:
      _a = _m
  # Volume candidates
  _alpha_w_candidates = [
    _alphaw_init, 1e-5, 0.01, .99, 1-1e-5, 0.5, 1-_alphaw_init, _a]
  
  # For highly incompressible, low-water component:
  if _T_init < Tc:
    # Use initial T guess, and bisection search over water density
    _f_from_rho_w = lambda rho_w: kern(yw * rho_mix / rho_w, _T_init)[0]
    _a, _b = 0.01, 1160
    # Bisect for volume condition using water density as a variable
    for i in range(_bisection_count):
      _m = 0.5 * (_a + _b)
      if _f_from_rho_w(_m)[1] * _f_from_rho_w(_a)[1] < 0:
        # m & a different sign; replace b
        _b = _m
      else:
        _a = _m
  dense_water_init = yw * rho_mix / _a
  _alpha_w_candidates.append(yw * rho_mix / _a)

  _alpha_w_candidates = np.unique(_alpha_w_candidates)

  # Select best candidate
  _candidate_performance = [
    np.linalg.norm(kern(*np.array([_alpha_w, _T_init]))[0])
    for _alpha_w in _alpha_w_candidates]
  _alphaw_init = _alpha_w_candidates[np.argmin(_candidate_performance)]
  U = np.array([_alphaw_init, _T_init])

  # Check for low vol frac
  if False and _alphaw_init < 0.001:
    # Non-iterative pressure
    if _phi > 0.01:
      _p_LMA, _phi = p_LMA(_T_init, yw, ya, ym, get_phi=True)
      _vol_occupied = ya * R_a * _T_init / _p_LMA \
        + (1.0 - (ya + yw)) / rho_m0 / (1.0 + (_p_LMA - p_m0) / K)
      _alphaw_test = np.clip((1.0/rho_mix - _vol_occupied) * rho_mix, 0.0, rho_mix)
      if False: # _alphaw_test > 0:
        # Add volumefrac-weighted pressure of water
        _p_WLMA = _p_LMA * (1.0 - _alphaw_test) \
          + _alphaw_test * float_mix_functions.p(
            rho_mix * yw / _alphaw_test, _T_init)
      else:
        _p_WLMA = _p_LMA
      return _p_WLMA, _T_init, msg_list
    # else:
      # raise NotImplementedError
    

  # Newton iterations
  for _k in range(32):
    # Newton step with backtracking
    U, f, msg_list = iterate_backtrack_box(U, kern, append_log=msg_list)
    fnorm = np.linalg.norm(f)
    if fnorm < 1e-12:
      break
    if np.any(np.isnan(f)):
      fnorm = 1e10
      continue
      print(kern(*U), vol_energy, rho_mix, yw)
      raise Exception("Nan encountered, dumping")

  # Compute final U = [alpha_w, T]
  T_calc = U[1]
  rhow = rho_mix * yw / U[0]
  p_calc = float_mix_functions.p(rhow, U[1])
  # if np.any(np.abs(p_calc) > 1e9):
  #   print(kern(*U), vol_energy, rho_mix, yw)
  #   raise Exception("Dumping")

  if not dump_state:
    return p_calc, T_calc, msg_list
  else:
    return p_calc, T_calc, msg_list, (_alpha_w_candidates, _candidate_performance, dense_water_init)"""

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