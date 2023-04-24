#cython: language_level=3

cimport cython
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
      drhowdT = -d2psi / d1psi * rhoc # Units TODO:
      # Compute density-temperature slope for m state
      drhomdT = -yw / ym * (rhom / rhow) * (rhom / rhow) * drhowdT
      # Compute water energy-temperature slope
      dewdT = _c_v_w - R * Tc / rhoc * _phirall.phir_dt * drhowdT
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
      dT *= min(1.0, 100.0/(1e-16+fabs(dT)))
      # Newton-step temperature
      T += dT
      # Newton-step d (cross term in Jacobian)
      d += drhowdT/rhoc
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
      d = rho_satv/rhoc if rhow < rho_satv else rho_satl/rhoc
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
          + psat * K / rho_m0 / ((psat + K - p_m0)*(psat + K - p_m0)) * dpsatdT

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
        pass
    if not is_mixed:
      # Compute phasic states using current d, T
      rhow = d*rhoc
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
        + _c1 * Z * d * rhoc * R
      # Compute density-temperature slope under the volume addition constraint
      drhowdT = -d2psi / d1psi * rhoc
      # Compute density-temperature slope for m state TODO: chain rule may be
      #   missing a term.
      drhomdT = -(rhom*rhom/ ym) * (yw * (drhowdT / (rhow*rhow)) 
        - ya * (R_a / pmix))
      # Compute water energy-temperature slope
      dewdT = _c_v_w - R * Tc / rhoc * _phirall.phir_dt * drhowdT
      # Compute magma energy-temperature slope (c_v dT + de/dv * dv, v = v(T))
      demdT = c_v_m0 \
        + pmix / (rhom*rhom) * drhomdT # - p dv = p (drho) / rho^2
      # Compute air energy-temperature slope
      deadT = c_v_a
      # Compute mixture energy-temperature slope
      dedT = yw * dewdT + ym * demdT + ya * deadT
      ''' Compute Newton step for temperature '''
      curr_energy = yw * _u \
        + ym * (c_v_m0 * T + magma_mech_energy(pmix, K, p_m0, rho_m0)) \
        + ya * c_v_a * T
      # Temperature migration
      dT = -(curr_energy - vol_energy/rho_mix)/dedT
      # Limit to dT <= 100
      dT *= min(1.0, 100.0/(1e-16+fabs(dT)))
      # Newton-step temperature
      T += dT
      # Newton-step d (cross term in Jacobian)
      d += drhowdT/rhoc
      out_pair = pure_phase_newton_pair_WLMA(d, T, rho_mix, yw, ya, K, p_m0,
        rho_m0, R_a, gamma_a)
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

''' Legacy/test functions '''

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
      d2psi = _c1 * Z * d * rhoc * R
      # Compute density-temperature slope under the volume addition constraint
      drhowdT = -d2psi / d1psi * rhoc # Units TODO:
      # Compute density-temperature slope for m state
      drhomdT = -yw / ym * (rhom / rhow) * (rhom / rhow) * drhowdT
      # Compute pressure-temperature slope under the volume addition constraint
      # dpdT = rhoc * R * (
      #   (2.0 * d + 2.0 * d * _phir_all["phir_d"] 
      #     + d * d * _phir_all["phir_dd"]) * drhowdT * T
        # + d * d * (1.0 + _phir_all["phir_d"]))
      # Compute water energy-temperature slope
      dewdT = _c_v_w - R * Tc / rhoc * _phir_all.phir_dt * drhowdT
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
      dT *= min(1.0, 100.0/(1e-16+fabs(dT)))

      # Extrapolation cut
      # if case_str == "iter-subcrit":

      # if dT*dT > trust_region_size*trust_region_size:
        # dT = trust_region_size*np.sign(dT)
      # Newton-step temperature
      T += dT
      # Newton-step d (cross term in Jacobian)
      d += drhowdT/rhoc
      out_pair = pure_phase_newton_pair(d, T, rho_mix, yw, K, p_m0, rho_m0)
      d -= out_pair.first/out_pair.second
      # Update pressure
      pmix = p(d*rhoc, T)
      # Update water phasic density
      rhow = d * rhoc
    
    # Clip temperatures below the triple point temperature
    T = max(T, 273.16)

    # Append output
    iteration_path.append((pmix, d*rhoc, T, case_str))

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