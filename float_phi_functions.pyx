#cython: language_level=3

''' # For cython timing: place at top
# cython: binding=True
# cython: linetrace=True
# cython: profile=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1'''

ctypedef double DTYPE_t
cimport cython

cdef extern from "math.h":
    double exp(double x)
    double log(double x)
    double pow(double x, double y)

cdef DTYPE_t[2] a_res55_56 = [3.5, 3.5]
cdef DTYPE_t[2] A_res55_56 = [0.32, 0.32]
cdef DTYPE_t[3] alpha_res52_54 = [20., 20., 20.]
cdef DTYPE_t[2] b_res55_56 = [0.85, 0.95]
cdef DTYPE_t[2] B_res55_56 = [0.2, 0.2]
cdef DTYPE_t[3] beta_res52_54 = [150., 150., 250.]
cdef DTYPE_t[2] beta_res55_56 = [0.3, 0.3]
# range(7,22) -> 1; range(22,42) -> 2; range(42,46) -> 3; range(46,52)
cdef DTYPE_t[51] c_res1_51 = [
  0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
  1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
  2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 4., 6., 6., 6., 6.]
cdef DTYPE_t[3] c_res52_54 = [0., 0., 0.]
cdef DTYPE_t[2] C_res55_56 = [28., 32.]
cdef DTYPE_t[56] d_res = [
  1.,  1.,  1.,  2.,  2.,  3.,  4.,  1.,  1.,  1.,  2.,  2.,  3.,
  4.,  4.,  5.,  7.,  9., 10., 11., 13., 15.,  1.,  2.,  2.,  2.,
  3.,  4.,  4.,  4.,  5.,  6.,  6.,  7.,  9.,  9.,  9.,  9.,  9.,
  10., 10., 12.,  3.,  4.,  4.,  5., 14.,  3.,  6.,  6.,  6.,  3.,
  3.,  3.,  1.,  1.]
cdef DTYPE_t[2] D_res55_56 = [700., 800.]
cdef DTYPE_t[3] eps_res52_54 = [1., 1., 1.]
cdef DTYPE_t[3] gamma_res52_54 = [1.21, 1.21, 1.25]
cdef DTYPE_t[56] n_res = [
        1.2533547935523e-02,  7.8957634722828e+00, -8.7803203303561e+00,
        3.1802509345418e-01, -2.6145533859358e-01, -7.8199751687981e-03,
        8.8089493102134e-03, -6.6856572307965e-01,  2.0433810950965e-01,
       -6.6212605039687e-05, -1.9232721156002e-01, -2.5709043003438e-01,
        1.6074868486251e-01, -4.0092828925807e-02,  3.9343422603254e-07,
       -7.5941377088144e-06,  5.6250979351888e-04, -1.5608652257135e-05,
        1.1537996422951e-09,  3.6582165144204e-07, -1.3251180074668e-12,
       -6.2639586912454e-10, -1.0793600908932e-01,  1.7611491008752e-02,
        2.2132295167546e-01, -4.0247669763528e-01,  5.8083399985759e-01,
        4.9969146990806e-03, -3.1358700712549e-02, -7.4315929710341e-01,
        4.7807329915480e-01,  2.0527940895948e-02, -1.3636435110343e-01,
        1.4180634400617e-02,  8.3326504880713e-03, -2.9052336009585e-02,
        3.8615085574206e-02, -2.0393486513704e-02, -1.6554050063734e-03,
        1.9955571979541e-03,  1.5870308324157e-04, -1.6388568342530e-05,
        4.3613615723811e-02,  3.4994005463765e-02, -7.6788197844621e-02,
        2.2446277332006e-02, -6.2689710414685e-05, -5.5711118565645e-10,
       -1.9905718354408e-01,  3.1777497330738e-01, -1.1841182425981e-01,
       -3.1306260323435e+01,  3.1546140237781e+01, -2.5213154341695e+03,
       -1.4874640856724e-01,  3.1806110878444e-01]
cdef DTYPE_t[56] t_res = [
  -0.5  ,  0.875,  1.   ,  0.5  ,  0.75 ,  0.375,  1.   ,  4.   ,
  6.   , 12.   ,  1.   ,  5.   ,  4.   ,  2.   , 13.   ,  9.   ,
  3.   ,  4.   , 11.   ,  4.   , 13.   ,  1.   ,  7.   ,  1.   ,
  9.   , 10.   , 10.   ,  3.   ,  7.   , 10.   , 10.   ,  6.   ,
  10.   , 10.   ,  1.   ,  2.   ,  3.   ,  4.   ,  8.   ,  6.   ,
  9.   ,  8.   , 16.   , 22.   , 23.   , 23.   , 10.   , 50.   ,
  44.   , 46.   , 50.   ,  0.   ,  1.   ,  4.   ,  0.   ,  0.   ]
cdef DTYPE_t[2] _exp1_55_56 = [1.6666666666666667, 1.6666666666666667]
# Fused coefficient arrays for 1 to 51 of uniform type.
#   Coefficients (n, d, t, c) are contiguous in memory.
# These arrays are used in some earlier implementations.
cdef DTYPE_t[204] ndtc1_51
for i in range(51):
  ndtc1_51[4*i] = n_res[i]
  ndtc1_51[4*i+1] = d_res[i]
  ndtc1_51[4*i+2] = t_res[i]
  ndtc1_51[4*i+3] = c_res1_51[i]
# Coefficients (n, d, t) are contiguous in memory for int c_coeff optimization
cdef DTYPE_t[153] ndt1_51
for i in range(51):
  ndt1_51[3*i] = n_res[i]
  ndt1_51[3*i+1] = d_res[i]
  ndt1_51[3*i+2] = t_res[i]

# Ideal gas part coefficients: updated values of the 2018 IAPWS Release
cdef DTYPE_t[8] n_ideal = [-8.3204464837497,  6.6832105275932,  3.00632   ,
        0.012436  ,  0.97315   ,  1.2795    ,  0.96956   ,  0.24873   ]
cdef DTYPE_t[8] g_ideal = [ 0.        ,  0.        ,  0.        ,
        1.28728967,  3.53734222,  7.74073708,  9.24437796, 27.5075105 ]
cdef DTYPE_t Tc   = 647.096      # K
cdef DTYPE_t rhoc = 322          # kg / m^3
cdef DTYPE_t R    = 0.46151805e3 # J / kg K
# Saturated liquid density correlation (Saul and Wagner 1987 Eq. 2.3)
cdef DTYPE_t[6] satl_powsb = [
  0.3333333333333333, 0.6666666666666666, 1.6666666666666667,
  5.333333333333333, 14.333333333333334, 36.666666666666664]
cdef DTYPE_t[6] satl_coeffsb = [
  1.99206, 1.10123, -5.12506e-1, -1.75263, -45.4485, -6.75615e5]
# Saturated vapour density correlation (Saul and Wagner 1987 Eq. 2.2)
cdef DTYPE_t[6] satv_powsc = [0.33333333, 0.66666667, 1.33333333, 3.0,
  6.16666667, 11.83333333]
cdef DTYPE_t[6] satv_coeffsc = [-2.02957, -2.68781, -5.38107, -17.3151,
  -44.6384, -64.3486]

''' Output struct types '''
# Generic pair
cdef struct Pair:
  DTYPE_t first
  DTYPE_t second
# Triple of saturation information
cdef struct SatTriple:
  DTYPE_t psat
  DTYPE_t rho_satl
  DTYPE_t rho_satv
# Collection of phir and its derivatives
cdef struct Derivatives_0_1_2:
  DTYPE_t phir
  DTYPE_t phir_d
  DTYPE_t phir_dd
  DTYPE_t phir_t
  DTYPE_t phir_tt
  DTYPE_t phir_dt
# Coefficient triple (more efficient packing and tighter typing)
cdef struct CoeffTriple_ndt_did:
  double n
  int d
  double t
# Coefficient triple (more efficient packing and tighter typing)
cdef struct CoeffTriple_ndt_dii:
  double n
  int d
  int t

# Fill typed ndt arrays (type suffix did->double int double, ...)
cdef CoeffTriple_ndt_did[7] typed_ndt_1_7
cdef CoeffTriple_ndt_dii[44] typed_ndt_8_51
for i in range(7):
  typed_ndt_1_7[i].n = ndtc1_51[4*i]
  typed_ndt_1_7[i].d = int(ndtc1_51[4*i+1])
  typed_ndt_1_7[i].t = ndtc1_51[4*i+2]
for i in range(7,51):
  typed_ndt_8_51[i-7].n = ndtc1_51[4*i]
  typed_ndt_8_51[i-7].d = int(ndtc1_51[4*i+1])
  typed_ndt_8_51[i-7].t = int(ndtc1_51[4*i+2])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double pow_fd(double x, int y) noexcept:
  ''' Exponentiation of type double ** int.
    Input x expected to be strictly positive.'''
  cdef double out = 1.0
  if y < 0:
    x = 1.0 / x
    y = -y
  elif y == 0:
    return 1.0
  while y > 1:
    if y % 2:
      out *= x    # Apply x^(2^i)
    x *= x        # x^(2^i) -> x^(2^(i+1))
    y //= 2       # Bit shift
  return out * x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _dummy(DTYPE_t d, DTYPE_t t):
  ''' Dummy function that returns d (for profiling). '''
  return d

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phi0(DTYPE_t d, DTYPE_t t) noexcept:
  ''' Ideal gas part phi0 of dimless Helmholtz function. '''
  cdef DTYPE_t out = log(d) + n_ideal[0] + n_ideal[1] * t + n_ideal[2] * log(t)
  cdef unsigned short i
  for i in range(3,8):
    out += n_ideal[i] * log(1.0 - exp(-g_ideal[i] * t))
  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phi0_d(DTYPE_t d, DTYPE_t t) noexcept:
  ''' Ideal gas part phi0 of d/dd dimless Helmholtz function. '''
  return 1.0/d

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phi0_dd(DTYPE_t d, DTYPE_t t) noexcept:
  ''' Ideal gas part phi0 of d2/dd2 dimless Helmholtz function. '''
  return -1.0/(d*d)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phi0_t(DTYPE_t d, DTYPE_t t) noexcept:
  ''' Ideal gas part phi0 of d/dt dimless Helmholtz function. '''
  cdef DTYPE_t out = n_ideal[1] + n_ideal[2] / t
  cdef unsigned short i
  for i in range(3,8):
    out += n_ideal[i] * g_ideal[i] * (1.0 / (1.0 - exp(-g_ideal[i] * t)) - 1.0)
  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phi0_tt(DTYPE_t d, DTYPE_t t) noexcept:
  ''' Ideal gas part phi0 of d2/dt2 dimless Helmholtz function. '''
  cdef DTYPE_t out = -n_ideal[2] / (t * t)
  cdef DTYPE_t _temp
  cdef unsigned short i
  for i in range(3,8):
    _exp_result = exp(-g_ideal[i] * t)
    out += -n_ideal[i] * g_ideal[i] * g_ideal[i] \
      * _exp_result/((1.0 - _exp_result)*(1.0 - _exp_result))
  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phi0_dt(DTYPE_t d, DTYPE_t t) noexcept:
  ''' Ideal gas part phi0 of d2/(dd dt) dimless Helmholtz function. '''
  return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phir(DTYPE_t d, DTYPE_t t):
  ''' Residual part of dimless Helmholtz function
      phi = f/(RT).
  Cython implementation for float input. '''
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  cdef DTYPE_t out = 0.0

  # Use strides as below
  # cdef n_coeff ndtc1_51[4*i]
  # cdef d_coeff ndtc1_51[4*i+1]
  # cdef t_coeff ndtc1_51[4*i+2]
  # cdef c_coeff ndtc1_51[4*i+3]

  # Compute uniform coefficients with mixed data (1-indices 1 to 51)
  cdef unsigned short i
  for i in range(7):
    out += ndtc1_51[4*i] \
      * (d ** (ndtc1_51[4*i+1])) * (t ** ndtc1_51[4*i+2])
  for i in range(7, 51):
    out += ndtc1_51[4*i] \
      * (d ** (ndtc1_51[4*i+1])) * (t ** ndtc1_51[4*i+2]) \
      * exp(-d ** ndtc1_51[4*i+3])

  # Declare temporary registers
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _d_shift
  cdef DTYPE_t _t_shift
  # Compute heterogeneous coefficients for 1-indices 52 to 54
  for i in range(51,54):
    _d_shift = d - eps_res52_54[i-51]
    _t_shift = t - gamma_res52_54[i-51]
    out += n_res[i] * (d ** (d_res[i])) * (t ** t_res[i]) \
      * exp(-alpha_res52_54[i-51] * _d_shift * _d_shift \
      -beta_res52_54[i-51] * _t_shift * _t_shift)
  # Compute heterogeneous coefficients for 1-indices 55 to 56
  for i in range(54,56):
    _theta = (1.0 - t) + A_res55_56[i-54] * d_quad ** _exp1_55_56[i-54]
    _Delta = _theta*_theta + B_res55_56[i-54] * d_quad ** a_res55_56[i-54]
    out += n_res[i] * (d ** (d_res[i])) * (t ** t_res[i]) \
      * _Delta ** b_res55_56[i-54] * exp(-C_res55_56[i-54] * d_quad \
      - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))

  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phir_d(DTYPE_t d, DTYPE_t t):
  ''' First delta-derivative of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details.
  Cython implementation for float input. '''
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _neg_dpowc
  cdef DTYPE_t out = 0.0

  # Use strides as below
  # cdef n_coeff ndtc1_51[4*i]
  # cdef d_coeff ndtc1_51[4*i+1]
  # cdef t_coeff ndtc1_51[4*i+2]
  # cdef c_coeff ndtc1_51[4*i+3]

  # Compute uniform coefficients with mixed data (1-indices 1 to 51)
  cdef unsigned short i
  for i in range(7):
    # out += n_coeff * d_coeff * (d ** (d_coeff-1.0)) * (t ** t_coeff)
    out += ndtc1_51[4*i] \
      * ndtc1_51[4*i+1] \
      * (d ** (ndtc1_51[4*i+1]-1.0)) * (t ** ndtc1_51[4*i+2])
  for i in range(7, 51):
    _neg_dpowc = -d ** ndtc1_51[4*i+3]
    out += ndtc1_51[4*i] \
      * (ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _neg_dpowc) \
      * (d ** (ndtc1_51[4*i+1]-1.0)) * (t ** ndtc1_51[4*i+2]) \
      * exp(_neg_dpowc)

  # Compute pre-exponential coefficients
  cdef DTYPE_t _d_shift
  cdef DTYPE_t _t_shift
  cdef DTYPE_t _coeff  
  # Compute heterogeneous coefficients for 1-indices 52 to 54
  for i in range(51,54):
    _coeff = n_res[i] * (d ** (d_res[i]-1.0)) * (t ** t_res[i]) \
      * (d_res[i] - 2.0 * alpha_res52_54[i-51] * d * (d - eps_res52_54[i-51]))
    _d_shift = d - eps_res52_54[i-51]
    _t_shift = t - gamma_res52_54[i-51]
    out += _coeff * exp(-alpha_res52_54[i-51] * _d_shift * _d_shift \
      -beta_res52_54[i-51] * _t_shift * _t_shift)
  # Compute heterogeneous coefficients for 1-indices 55 to 56
  for i in range(54,56):
    _coeff = n_res[i] * (d ** (d_res[i]-1.0)) * (t ** t_res[i])
    _theta = (1.0 - t) + A_res55_56[i-54] * d_quad ** _exp1_55_56[i-54]
    _Delta = _theta*_theta + B_res55_56[i-54] * d_quad ** a_res55_56[i-54]
    _coeff *= (
      _Delta * (1.0 - 2.0 * C_res55_56[i-54] * (d-1.0) * d)
      + b_res55_56[i-54] * d * (d-1.0) * (
        A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54]
        * d_quad**(_exp1_55_56[i-54] - 1.0)
        + 2.0 * B_res55_56[i-54] * a_res55_56[i-54]
        * d_quad**(a_res55_56[i-54] - 1.0)
      )
    )
    if _Delta != 0:
      _Delta = _Delta ** (b_res55_56[i-54]-1.0)
    _coeff *= _Delta
    out += _coeff * exp(-C_res55_56[i-54] * d_quad \
      - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))

  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phir_dd(DTYPE_t d, DTYPE_t t):
  ''' Second delta-derivative of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details.
  Cython implementation for float input.
  '''

  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  cdef DTYPE_t _neg_dpowc
  cdef DTYPE_t out = 0.0

  # Use strides as below
  # cdef n_coeff ndtc1_51[4*i]
  # cdef d_coeff ndtc1_51[4*i+1]
  # cdef t_coeff ndtc1_51[4*i+2]
  # cdef c_coeff ndtc1_51[4*i+3]

  # Compute uniform coefficients with mixed data (1-indices 1 to 51)
  cdef unsigned short i
  for i in range(7):
    out += ndtc1_51[4*i] \
      * ndtc1_51[4*i+1] * (ndtc1_51[4*i+1] - 1.0) \
      * (d ** (ndtc1_51[4*i+1] - 2.0)) * (t ** ndtc1_51[4*i+2])
  for i in range(7, 51):
    _neg_dpowc = -d ** ndtc1_51[4*i+3]
    out += ndtc1_51[4*i] \
      * ((ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _neg_dpowc) \
        * (ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _neg_dpowc - 1.0) \
        + ndtc1_51[4*i+3] * ndtc1_51[4*i+3] * _neg_dpowc ) \
      * (d ** (ndtc1_51[4*i+1] - 2.0)) * (t ** ndtc1_51[4*i+2]) \
      * exp(_neg_dpowc)

  # Compute pre-exponential coefficients
  cdef DTYPE_t _c1
  cdef DTYPE_t _c2
  cdef DTYPE_t _coeff
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _dDelta_div
  cdef DTYPE_t _ddDelta
  # Compute heterogeneous coefficients for 1-indices 52 to 54
  for i in range(51,54):
    _coeff = n_res[i] * (d ** (d_res[i])) * (t ** t_res[i]) \
    * (-2.0 * alpha_res52_54[i-51] \
    + 4.0 * alpha_res52_54[i-51] * alpha_res52_54[i-51] \
      * (d - eps_res52_54[i-51]) * (d - eps_res52_54[i-51]) \
    - 4.0 * d_res[i] * alpha_res52_54[i-51] / d * (d - eps_res52_54[i-51]) \
    + d_res[i] * (d_res[i] - 1.0) / (d * d))
    _c1 = d - eps_res52_54[i-51]
    _c2 = t - gamma_res52_54[i-51]
    out += _coeff * exp(-alpha_res52_54[i-51] * _c1 * _c1 \
      -beta_res52_54[i-51] * _c2 * _c2)
  # Compute heterogeneous coefficients for 1-indices 55 to 56
  for i in range(54,56):
    _theta = (1.0 - t) + A_res55_56[i-54] * d_quad ** _exp1_55_56[i-54]
    _Delta = _theta*_theta + B_res55_56[i-54] * d_quad ** a_res55_56[i-54]
    # Compute d(Delta)/d(delta) divided by (delta - 1.0) for numerical stability
    _dDelta_div = A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54] \
        * d_quad**(_exp1_55_56[i-54] - 1.0) \
      + 2.0 * B_res55_56[i-54] * a_res55_56[i-54] \
        * d_quad**(a_res55_56[i-54] - 1.0)
    # Reuse register
    if d_quad != 0.0:
      _c1 = d_quad ** (_exp1_55_56[i-54] - 2.0)
    else:
      _c1 = 0.0
    _c2 = A_res55_56[i-54] / beta_res55_56[i-54] \
      * d_quad**(_exp1_55_56[i-54] - 1.0)
    _ddDelta = _dDelta_div + ((d-1.0)**2) * (
      4.0 * B_res55_56[i-54] * a_res55_56[i-54] * (a_res55_56[i-54] - 1.0)
        * d_quad**(a_res55_56[i-54] - 2.0)
      + 2.0 * _c2*_c2
      + 4.0 * _theta * A_res55_56[i-54] / beta_res55_56[i-54] \
        * (_exp1_55_56[i-54] - 1.0) * _c1
    )
    # Finish d(Delta)/d(delta) computation in-place
    _dDelta_div *= d - 1.0
    # Replace (t_res is zero, so coeffs[54:56] contains invalid entries) for
    #   1-indices from 55 to 56
    _coeff = _Delta*_Delta * (-4.0 * C_res55_56[i-54] * (d-1.0) 
      + d * (2.0*C_res55_56[i-54]*d_quad - 1.0) * 2.0 * C_res55_56[i-54])
    _coeff += _Delta * 2.0 * b_res55_56[i-54] * _dDelta_div \
      * (1.0 - 2.0 * d * C_res55_56[i-54] * (d - 1.0))
    _coeff += b_res55_56[i-54] * (_Delta * _ddDelta
      + (b_res55_56[i-54] - 1.0) * _dDelta_div * _dDelta_div) * d
    # Reuse register
    if _Delta != 0.0:
      _c1 = _Delta ** (b_res55_56[i-54] - 2.0)
    else:
      _c1 = 0.0
    _coeff *= n_res[i] * _c1
    out += _coeff * exp(-C_res55_56[i-54] * d_quad \
      - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))

  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef Pair fused_phir_d_phir_dd(DTYPE_t d, DTYPE_t t) noexcept:
  ''' Optimized routine for simultaneously computing the first and second
  delta-derivatives of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details.
  Cython implementation for float input.
  '''

  cdef DTYPE_t out_phir_d = 0.0
  cdef DTYPE_t out_phir_dd = 0.0
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  # Declare temporary registers
  cdef DTYPE_t _c1
  cdef DTYPE_t _c2
  cdef DTYPE_t _c_coeff
  cdef DTYPE_t _common

  # Use strides as below
  # cdef n_coeff ndt1_51[3*i]
  # cdef d_coeff ndt1_51[3*i+1]
  # cdef t_coeff ndt1_51[3*i+2]

  # Compute uniform coefficients with mixed coefficients ndt (1-indices 1 to 51)
  cdef unsigned short i
  for i in range(7):
    _common = ndt1_51[3*i] * ndt1_51[3*i+1] \
      * (d ** (ndt1_51[3*i+1]-1.0)) * (t ** ndt1_51[3*i+2])
    out_phir_d += _common
    out_phir_dd += _common * (ndt1_51[3*i+1] - 1.0) / d
  # Integer c_coeff optimization: use c as
  #   range(7,22) -> 1
  #   range(22,42) -> 2
  #   range(42,46) -> 3
  #   range(46,47) -> 4
  #   range(47,51) -> 6
  # and manually exponentiate d**c
  _c1 = -d
  _c2 = exp(_c1)
  _c_coeff = 1.0
  for i in range(7,22):
    _common = ndt1_51[3*i] \
      * (d ** (ndt1_51[3*i+1]-1.0)) * (t ** ndt1_51[3*i+2]) * _c2
    out_phir_d += _common * (ndt1_51[3*i+1] + _c_coeff * _c1)
    out_phir_dd += _common * ((ndt1_51[3*i+1] + _c_coeff * _c1) \
        * (ndt1_51[3*i+1] + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / d
  _c1 = -d * d
  _c2 = exp(_c1)
  _c_coeff = 2.0
  for i in range(22,42):
    _common = ndt1_51[3*i] \
      * (d ** (ndt1_51[3*i+1]-1.0)) * (t ** ndt1_51[3*i+2]) * _c2
    out_phir_d += _common * (ndt1_51[3*i+1] + _c_coeff * _c1)
    out_phir_dd += _common * ((ndt1_51[3*i+1] + _c_coeff * _c1) \
        * (ndt1_51[3*i+1] + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / d
  _c1 = -d * d * d
  _c2 = exp(_c1)
  _c_coeff = 3.0
  for i in range(42,46):
    _common = ndt1_51[3*i] \
      * (d ** (ndt1_51[3*i+1]-1.0)) * (t ** ndt1_51[3*i+2]) * _c2
    out_phir_d += _common * (ndt1_51[3*i+1] + _c_coeff * _c1)
    out_phir_dd += _common * ((ndt1_51[3*i+1] + _c_coeff * _c1) \
        * (ndt1_51[3*i+1] + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / d
  _c1 = d * d
  _c1 *= -_c1
  _c2 = exp(_c1)
  _c_coeff = 4.0
  for i in range(46,47):
    _common = ndt1_51[3*i] \
      * (d ** (ndt1_51[3*i+1]-1.0)) * (t ** ndt1_51[3*i+2]) * _c2
    out_phir_d += _common * (ndt1_51[3*i+1] + _c_coeff * _c1)
    out_phir_dd += _common * ((ndt1_51[3*i+1] + _c_coeff * _c1) \
        * (ndt1_51[3*i+1] + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / d
  _c1 = d * d * d
  _c1 *= -_c1
  _c2 = exp(_c1)
  _c_coeff = 6.0
  for i in range(47,51):
    _common = ndt1_51[3*i] \
      * (d ** (ndt1_51[3*i+1]-1.0)) * (t ** ndt1_51[3*i+2]) * _c2
    out_phir_d += _common * (ndt1_51[3*i+1] + _c_coeff * _c1)
    out_phir_dd += _common * ((ndt1_51[3*i+1] + _c_coeff * _c1) \
        * (ndt1_51[3*i+1] + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / d
  
  # One-loop form of range(7,51) (approx. 40% more load)
  # for i in range(7, 51):
  #   _c1 = -d ** ndtc1_51[4*i+3]
  #   _common = ndtc1_51[4*i] \
  #     * (d ** (ndtc1_51[4*i+1]-1.0)) * (t ** ndtc1_51[4*i+2]) \
  #     * exp(_c1)
  #   out_phir_d += _common * (ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _c1)
  #   out_phir_dd += _common * ((ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _c1) \
  #       * (ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _c1 - 1.0) \
  #       + ndtc1_51[4*i+3] * ndtc1_51[4*i+3] * _c1 ) / d

  # Declare temporary registers
  cdef DTYPE_t _c3
  cdef DTYPE_t _c4
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _dDelta_div
  cdef DTYPE_t _ddDelta
  # Compute heterogeneous coefficients for 1-indices 52 to 54
  for i in range(51,54):
    # Compute commons
    _c1 = d - eps_res52_54[i-51]
    _c2 = t - gamma_res52_54[i-51]
    _common = n_res[i] * exp(-alpha_res52_54[i-51] * _c1 * _c1 \
      -beta_res52_54[i-51] * _c2 * _c2) \
      * (d * d # d ** (d_res[i]-1.0)
        ) * (t ** t_res[i])
    # Compute phir_d term
    out_phir_d += _common * (d_res[i] \
      - 2.0 * alpha_res52_54[i-51] * d * _c1)
    # Compute phir_dd term
    out_phir_dd += _common * d * (-2.0 * alpha_res52_54[i-51]
      + 4.0 * alpha_res52_54[i-51] * alpha_res52_54[i-51]
        * _c1 * _c1
      - 4.0 * d_res[i] * alpha_res52_54[i-51] / d * _c1
      + d_res[i] * (d_res[i] - 1.0) / (d * d))
  # Compute heterogeneous coefficients for 1-indices 55 to 56
  for i in range(54,56):
    # Compute commons
    _c1 = d_quad ** (_exp1_55_56[i-54] - 1.0) # 5/3
    _c2 = d_quad ** (a_res55_56[i-54] - 1.0)
    _theta = (1.0 - t) + A_res55_56[i-54] * _c1 * d_quad
    _Delta = _theta*_theta + B_res55_56[i-54] * _c2 * d_quad
    _common = n_res[i] * exp(-C_res55_56[i-54] * d_quad \
      - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))
    # Replace limiting value if Delta == 0
    if _Delta != 0.0:
      _common *= _Delta ** (b_res55_56[i-54] - 2.0) # 0.85 to 0.95
    else:
      _common = 0.0

    # Compute phir_d term
    _c3 = (
      _Delta * (1.0 - 2.0 * C_res55_56[i-54] * (d-1.0) * d)
      + b_res55_56[i-54] * d * (d-1.0) * (
        A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54] * _c1
        + 2.0 * B_res55_56[i-54] * a_res55_56[i-54] * _c2
      )
    )    
    out_phir_d += _c3 * _Delta * _common

    # Compute phir_dd term
    # Compute d(Delta)/d(delta) divided by (delta - 1.0) for numerical stability
    _dDelta_div = (A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54] * _c1
      + 2.0 * B_res55_56[i-54] * a_res55_56[i-54] * _c2)
    # Compute second derivative of Delta
    _c3 = A_res55_56[i-54] / beta_res55_56[i-54] * _c1
    _ddDelta = _dDelta_div + (
      4.0 * B_res55_56[i-54] * a_res55_56[i-54] * (a_res55_56[i-54] - 1.0) * _c2
      + 2.0 * _c3 * _c3 * d_quad
      + 4.0 * _theta * A_res55_56[i-54] / beta_res55_56[i-54] \
        * (_exp1_55_56[i-54] - 1.0) * _c1
    )
    # Finish d(Delta)/d(delta) computation in-place
    _dDelta_div *= d - 1.0
    # Compute coefficient to phir_dd
    _c1 = _Delta*_Delta * (-4.0 * C_res55_56[i-54] * (d-1.0) 
      + d * (2.0*C_res55_56[i-54]*d_quad - 1.0) * 2.0 * C_res55_56[i-54])
    _c1 += _Delta * 2.0 * b_res55_56[i-54] * _dDelta_div \
      * (1.0 - 2.0 * d * C_res55_56[i-54] * (d - 1.0))
    _c1 += b_res55_56[i-54] * (_Delta * _ddDelta
      + (b_res55_56[i-54] - 1.0) * _dDelta_div * _dDelta_div) * d
    out_phir_dd += _c1 * _common

  return Pair(out_phir_d, out_phir_dd)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phir_t(DTYPE_t d, DTYPE_t t):
  '''First tau-derivative of residual part of dimless Helmholtz function
      phi = f/(RT). '''
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  cdef DTYPE_t out = 0.0

  # Use strides as below
  # cdef n_coeff ndtc1_51[4*i]
  # cdef d_coeff ndtc1_51[4*i+1]
  # cdef t_coeff ndtc1_51[4*i+2]
  # cdef c_coeff ndtc1_51[4*i+3]

  # Compute uniform coefficients with mixed data (1-indices 1 to 51)
  cdef unsigned short i
  for i in range(7):
    out += ndtc1_51[4*i] * ndtc1_51[4*i+2] \
      * (d ** (ndtc1_51[4*i+1])) * (t ** (ndtc1_51[4*i+2]-1))
  for i in range(7, 51):
    out += ndtc1_51[4*i] * ndtc1_51[4*i+2] \
      * (d ** (ndtc1_51[4*i+1])) * (t ** (ndtc1_51[4*i+2]-1)) \
      * exp(-d ** ndtc1_51[4*i+3])

  # Declare temporary registers
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _d_shift
  cdef DTYPE_t _t_shift
  cdef DTYPE_t _coeff
  # Compute heterogeneous coefficients for 1-indices 52 to 54
  for i in range(51,54):
    _d_shift = d - eps_res52_54[i-51]
    _t_shift = t - gamma_res52_54[i-51]
    out += n_res[i] * (d ** (d_res[i])) * (t ** (t_res[i]-1.0)) \
      * (t_res[i] - 2.0 * beta_res52_54[i-51] * t * _t_shift) \
      * exp(-alpha_res52_54[i-51] * _d_shift * _d_shift \
      -beta_res52_54[i-51] * _t_shift * _t_shift)
  # Compute heterogeneous coefficients for 1-indices 55 to 56
  for i in range(54,56):
    _theta = (1.0 - t) + A_res55_56[i-54] * d_quad ** _exp1_55_56[i-54]
    _Delta = _theta*_theta + B_res55_56[i-54] * d_quad ** a_res55_56[i-54]
    _coeff = n_res[i] * d * 2.0 * (
      -_theta * b_res55_56[i-54] + _Delta * D_res55_56[i-54] * (1.0 - t)
    ) * exp(-C_res55_56[i-54] * d_quad - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))
    # Replace limiting value if Delta == 0
    if _Delta != 0.0:
      _coeff *= _Delta ** (b_res55_56[i-54] - 1.0) # 0.85 to 0.95
    else:
      _coeff = 0.0
    out += _coeff

  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phir_tt(DTYPE_t d, DTYPE_t t):
  '''Second tau-derivative of residual part of dimless Helmholtz function
      phi = f/(RT). '''
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  cdef DTYPE_t out = 0.0

  # Use strides as below
  # cdef n_coeff ndtc1_51[4*i]
  # cdef d_coeff ndtc1_51[4*i+1]
  # cdef t_coeff ndtc1_51[4*i+2]
  # cdef c_coeff ndtc1_51[4*i+3]

  # Stable summation pattern
  # # Get term
  # tent = out + term + comp
  # comp = (term + comp) - (tent - out)
  # out = tent

  # Compute uniform coefficients with mixed data (1-indices 1 to 51)
  cdef unsigned short i
  for i in range(7):
    out += ndtc1_51[4*i] * ndtc1_51[4*i+2] * (ndtc1_51[4*i+2] - 1.0) \
      * (d ** (ndtc1_51[4*i+1])) * (t ** (ndtc1_51[4*i+2] - 2.0))
  for i in range(7, 51):
    out += ndtc1_51[4*i] * ndtc1_51[4*i+2] * (ndtc1_51[4*i+2] - 1.0) \
      * (d ** (ndtc1_51[4*i+1])) * (t ** (ndtc1_51[4*i+2] - 2.0)) \
      * exp(-d ** ndtc1_51[4*i+3])

  # Declare temporary registers
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _d_shift
  cdef DTYPE_t _t_shift
  cdef DTYPE_t _coeff
  # Compute heterogeneous coefficients for 1-indices 52 to 54
  for i in range(51,54):
    _d_shift = d - eps_res52_54[i-51]
    _t_shift = t - gamma_res52_54[i-51]
    _coeff = t_res[i] - 2.0 * beta_res52_54[i-51] * t \
      * (t - gamma_res52_54[i-51])
    out += n_res[i] * (d ** (d_res[i])) * (t ** (t_res[i]-2.0)) \
      * exp(-alpha_res52_54[i-51] * _d_shift * _d_shift \
        -beta_res52_54[i-51] * _t_shift * _t_shift) \
      * (_coeff * _coeff - t_res[i] - 2.0 * beta_res52_54[i-51] * t * t)
  # Compute heterogeneous coefficients for 1-indices 55 to 56
  for i in range(54,56):
    _theta = (1.0 - t) + A_res55_56[i-54] * d_quad ** _exp1_55_56[i-54]
    _Delta = _theta*_theta + B_res55_56[i-54] * d_quad ** a_res55_56[i-54]
    # Replace limiting value if Delta == 0
    if _Delta != 0.0:
      _coeff = _Delta ** (b_res55_56[i-54] - 2.0) # 0.85 to 0.95
    else:
      return -float("inf")
    out += _coeff * n_res[i] * 2.0 * d * (
      b_res55_56[i-54] * (_Delta \
        + 2.0 * _theta*_theta * (b_res55_56[i-54] - 1.0)
        + 4.0 * _theta * _Delta * D_res55_56[i-54] * (t - 1.0))
      + _Delta * _Delta * D_res55_56[i-54] \
        * (2.0 * D_res55_56[i-54] * (t - 1.0) * (t - 1.0) - 1.0)
    ) * exp(-C_res55_56[i-54] * d_quad - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))

  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t phir_dt(DTYPE_t d, DTYPE_t t):
  '''Second mixed-derivative of residual part of dimless Helmholtz function
      phi = f/(RT). '''
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  cdef DTYPE_t out = 0.0
  cdef DTYPE_t _c1

  # Use strides as below
  # cdef n_coeff ndtc1_51[4*i]
  # cdef d_coeff ndtc1_51[4*i+1]
  # cdef t_coeff ndtc1_51[4*i+2]
  # cdef c_coeff ndtc1_51[4*i+3]

  # Compute uniform coefficients with mixed data (1-indices 1 to 51)
  cdef unsigned short i
  for i in range(7):
    out += ndtc1_51[4*i] * ndtc1_51[4*i+1] * ndtc1_51[4*i+2] \
      * (d ** (ndtc1_51[4*i+1] - 1.0)) * (t ** (ndtc1_51[4*i+2] - 1.0))
  for i in range(7, 51):
    _c1 = -d ** ndtc1_51[4*i+3]
    out += ndtc1_51[4*i] * ndtc1_51[4*i+2] \
      * (ndtc1_51[4*i+1] + c_res1_51[i] * _c1) \
      * (d ** (ndtc1_51[4*i+1] - 1.0)) * (t ** (ndtc1_51[4*i+2] - 1.0)) \
      * exp(_c1)

  # Declare temporary registers
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _dDelta
  cdef DTYPE_t _d_shift
  cdef DTYPE_t _t_shift
  cdef DTYPE_t _coeff
  cdef DTYPE_t _c2
  # Compute heterogeneous coefficients for 1-indices 52 to 54
  for i in range(51,54):
    _d_shift = d - eps_res52_54[i-51]
    _t_shift = t - gamma_res52_54[i-51]
    out += n_res[i] * (d ** (d_res[i] - 1.0)) * (t ** (t_res[i] - 1.0)) \
      * (d_res[i] - 2.0 * alpha_res52_54[i-51] * d * _d_shift) \
      * (t_res[i] - 2.0 * beta_res52_54[i-51] * t * _t_shift) \
      * exp(-alpha_res52_54[i-51] * _d_shift * _d_shift \
      -beta_res52_54[i-51] * _t_shift * _t_shift)
  # Compute heterogeneous coefficients for 1-indices 55 to 56
  for i in range(54,56):
    _c1 = d_quad**(_exp1_55_56[i-54] - 1.0)
    _c2 = d_quad**(a_res55_56[i-54] - 1.0)
    _theta = (1.0 - t) + A_res55_56[i-54] * d_quad * _c1
    _Delta = _theta * _theta + B_res55_56[i-54] * d_quad * _c2
    _dDelta = (d - 1.0) * (
      A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54] * _c1
      + 2.0 * B_res55_56[i-54] * a_res55_56[i-54] * _c2)
    _coeff = n_res[i] * (
      _Delta * _Delta * (-2.0 * D_res55_56[i-54] * (t - 1.0) \
      + d * 4.0 * C_res55_56[i-54] * D_res55_56[i-54] * (d - 1.0) * (t - 1.0))
      + d * _Delta * b_res55_56[i-54] * _dDelta \
        * (-2.0 * D_res55_56[i-54] * (t - 1.0))
      - 2.0 * _theta * b_res55_56[i-54] * _Delta \
        * (1.0 - 2.0 * d * C_res55_56[i-54] * (d - 1.0))
      + d * (
        -A_res55_56[i-54] * b_res55_56[i-54] * 2.0 / beta_res55_56[i-54] \
          * _Delta * (d - 1.0) * _c1
        - 2.0 * _theta * b_res55_56[i-54] * (b_res55_56[i-54] - 1.0) * _dDelta
      )
    ) * exp(-C_res55_56[i-54] * d_quad - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))
    # Replace limiting value if Delta == 0
    if _Delta != 0.0:
      _coeff *= _Delta ** (b_res55_56[i-54] - 2.0) # 0.85 to 0.95
    else:
      _coeff = 0.0
    out += _coeff
    _c1 = (
      _Delta * _Delta * (-2.0 * D_res55_56[i-54] * (t - 1.0) \
      + d * 4.0 * C_res55_56[i-54] * D_res55_56[i-54] * (d - 1.0) * (t - 1.0))
      + d * _Delta * b_res55_56[i-54] * _dDelta \
        * (-2.0 * D_res55_56[i-54] * (t - 1.0))
      - 2.0 * _theta * b_res55_56[i-54] * _Delta \
        * (1.0 - 2.0 * d * C_res55_56[i-54] * (d - 1.0))
      + d * (
        -A_res55_56[i-54] * b_res55_56[i-54] * 2.0 / beta_res55_56[i-54] \
          * _Delta * (d - 1.0) * _c1
        - 2.0 * _theta * b_res55_56[i-54] * (b_res55_56[i-54] - 1.0) * _dDelta
      )
    )

  return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef prho_sat_stepinfo(DTYPE_t T):
  ''' Returns isothermal saturation curve properties as tuple
  (psat, rho_satl, rho_satv). Solves the Maxwell construction (see e.g. P.
  Junglas).
  Calls def functions in float_phi_functions (can be optimized to cdef fns).
  '''
  # Compute reciprocal reduced temperature
  cdef DTYPE_t t = Tc / T
  if t < 1.0:
    return None, None, None, [0.0, 0.0], [0.0, 0.0]
  elif t == 1.0:
    # Special case: exactly critical
    return 22.06e6, None, None, [0.0, 0.0], [0.0, 0.0]

  cdef DTYPE_t _phir_d0
  cdef DTYPE_t _phir_dd0
  cdef DTYPE_t _phir_d1
  cdef DTYPE_t _phir_dd1
  cdef DTYPE_t _J00, _J01, _J10, _J11, _detJ
  cdef DTYPE_t f0, f1
  cdef DTYPE_t step0, step1
  cdef DTYPE_t d0 = 1.0
  cdef DTYPE_t d1 = 0
  cdef DTYPE_t _c0
  cdef Pair pair

  # Compute initial guess using independent sat curve correlations
  cdef unsigned short i
  for i in range(6):
    _c0 = 1.0-1.0/t
    d0 += satl_coeffsb[i] * _c0**satl_powsb[i]
    d1 += satv_coeffsc[i] * _c0**satv_powsc[i]
  d1 = exp(d1)
  # Fixed step Newton
  for i in range(3):
    # Compute phir_d, phir_dd values
    pair = fused_phir_d_phir_dd(d0, t)
    _phir_d0 = pair.first
    _phir_dd0 = pair.second
    pair = fused_phir_d_phir_dd(d1, t)
    _phir_d1 = pair.first
    _phir_dd1 = pair.second 
    # Assemble Jacobian for Maxwell residual equation
    _J00 = -2.0 * _phir_d0 - d0 * _phir_dd0 - phi0_d(d0, t)
    _J01 = 2.0 * _phir_d1 + d1 * _phir_dd1 + phi0_d(d1, t)
    _J10 = 1.0 + 2.0 * d0 * _phir_d0 + d0 * d0 * _phir_dd0
    _J11 = -1.0 - 2.0 * d1 * _phir_d1 - d1 * d1 * _phir_dd1
    _detJ = _J00 * _J11 - _J01 * _J10
    # Assemble vector of Maxwell residuals
    f0 = d1 * _phir_d1 - d0 * _phir_d0 \
        - phir(d0, t) - phi0(d0, t) + phir(d1, t) + phi0(d1, t)
    f1 = d0 + d0 * d0 * _phir_d0 - d1 - d1 * d1 * _phir_d1
    # Compute Newton step
    step0 = -( _J11 * f0 - _J01 * f1) / (_detJ)
    step1 = -(-_J10 * f0 + _J00 * f1) / (_detJ)
    d0 += step0
    d1 += step1
  # Compute latest function value
  pair = fused_phir_d_phir_dd(d0, t)
  _phir_d0 = pair.first
  _phir_dd0 = pair.second
  f0 = d1 * _phir_d1 - d0 * _phir_d0 \
       - phir(d0, t) - phi0(d0, t) + phir(d1, t) + phi0(d1, t)
  f1 = d0 + d0 * d0 * _phir_d0 - d1 - d1 * d1 * _phir_d1
  # Return psat, rho_satl, rho_satv, last_newton_step, residual
  return d0 * (1.0 + d0 * _phir_d0) * rhoc * R * T, \
    d0 * rhoc, d1 * rhoc, [step0, step1], [f0, f1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef SatTriple prho_sat(DTYPE_t T) noexcept:
  ''' Version of prho_sat that does not return stepping/convergence
  info. '''
  # Compute reciprocal reduced temperature
  cdef DTYPE_t t = Tc / T
  if t < 1.0:
    return SatTriple(-1.0, -1.0, -1.0)
  elif t == 1.0:
    # Special case: exactly critical
    return SatTriple(22.06e6, -1.0, -1.0)

  cdef DTYPE_t _phir_d0
  cdef DTYPE_t _phir_dd0
  cdef DTYPE_t _phir_d1
  cdef DTYPE_t _phir_dd1
  cdef DTYPE_t _J00, _J01, _J10, _J11, _detJ
  cdef DTYPE_t f0, f1
  cdef DTYPE_t step0, step1
  cdef DTYPE_t d0 = 1.0
  cdef DTYPE_t d1 = 0
  cdef DTYPE_t _c0

  # Compute initial guess using independent sat curve correlations
  cdef unsigned short i
  for i in range(6):
    _c0 = 1.0-1.0/t
    d0 += satl_coeffsb[i] * _c0**satl_powsb[i]
    d1 += satv_coeffsc[i] * _c0**satv_powsc[i]
  d1 = exp(d1)
  # Fixed step Newton
  for i in range(3):
    # Compute phir_d, phir_dd values
    pair = fused_phir_d_phir_dd(d0, t)
    _phir_d0 = pair.first
    _phir_dd0 = pair.second
    pair = fused_phir_d_phir_dd(d1, t)
    _phir_d1 = pair.first
    _phir_dd1 = pair.second 
    # Assemble Jacobian for Maxwell residual equation
    _J00 = -2.0 * _phir_d0 - d0 * _phir_dd0 - phi0_d(d0, t)
    _J01 = 2.0 * _phir_d1 + d1 * _phir_dd1 + phi0_d(d1, t)
    _J10 = 1.0 + 2.0 * d0 * _phir_d0 + d0 * d0 * _phir_dd0
    _J11 = -1.0 - 2.0 * d1 * _phir_d1 - d1 * d1 * _phir_dd1
    _detJ = _J00 * _J11 - _J01 * _J10
    # Assemble vector of Maxwell residuals
    f0 = d1 * _phir_d1 - d0 * _phir_d0 \
        - phir(d0, t) - phi0(d0, t) + phir(d1, t) + phi0(d1, t)
    f1 = d0 + d0 * d0 * _phir_d0 - d1 - d1 * d1 * _phir_d1
    # Compute Newton step
    step0 = -( _J11 * f0 - _J01 * f1) / (_detJ)
    step1 = -(-_J10 * f0 + _J00 * f1) / (_detJ)
    d0 += step0
    d1 += step1
  # Compute latest function value
  pair = fused_phir_d_phir_dd(d0, t)
  _phir_d0 = pair.first
  _phir_dd0 = pair.second
  # Return psat, rho_satl, rho_satv, last_newton_step, residual
  return SatTriple(d0 * (1.0 + d0 * _phir_d0) * rhoc * R * T, \
    d0 * rhoc, d1 * rhoc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef Derivatives_0_1_2 fused_phir_all(DTYPE_t d, DTYPE_t t) noexcept:
  ''' Optimized routine for simultaneously computing the 0th, 1st and 2nd
  derivatives of the residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details.
  Cython implementation for float input.
  '''

  cdef DTYPE_t out_phir    = 0.0
  cdef DTYPE_t out_phir_d  = 0.0
  cdef DTYPE_t out_phir_dd = 0.0
  cdef DTYPE_t out_phir_t  = 0.0
  cdef DTYPE_t out_phir_tt = 0.0
  cdef DTYPE_t out_phir_dt = 0.0
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  # Declare temporary registers
  cdef DTYPE_t _c1
  cdef DTYPE_t _c2
  cdef DTYPE_t _c3
  cdef DTYPE_t _c_coeff
  cdef DTYPE_t _common

  cdef unsigned short i
  
  # Compute terms with 1-indices 1 to 7 (0-indices 0 to 6)
  for i in range(7):
    # Compute common factors
    _common = typed_ndt_1_7[i].n * pow_fd(d, typed_ndt_1_7[i].d) \
      * (t ** typed_ndt_1_7[i].t)
    out_phir += _common
    # Cache intermediate result phir_d
    _c1 = typed_ndt_1_7[i].d * _common / d
    # Apply derivative operators
    out_phir_d += _c1
    out_phir_dd += (typed_ndt_1_7[i].d - 1) * _c1 / d
    out_phir_t += _common * typed_ndt_1_7[i].t / t
    out_phir_tt += _common * typed_ndt_1_7[i].t \
      * (typed_ndt_1_7[i].t - 1.0) / (t*t)
    out_phir_dt += _c1 * typed_ndt_1_7[i].t / t

  # Terms with 1-indices 8 to 51 Integer c_coeff optimization
  #   range(7,22) -> 1
  #   range(22,42) -> 2
  #   range(42,46) -> 3
  #   range(46,47) -> 4
  #   range(47,51) -> 6
  # and manually exponentiate d**c

  _c1 = -d
  _c2 = exp(_c1)
  _c_coeff = 1.0
  for i in range(0,15):
    # Compute common factors
    _common = typed_ndt_8_51[i].n \
      * pow_fd(d, typed_ndt_8_51[i].d) * pow_fd(t, typed_ndt_8_51[i].t) * _c2
    # Apply derivative operators
    out_phir += _common
    out_phir_d += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d
    out_phir_dd += _common * ((typed_ndt_8_51[i].d + _c_coeff * _c1) \
        * (typed_ndt_8_51[i].d + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / (d*d)
    out_phir_t += _common * typed_ndt_8_51[i].t / t
    out_phir_tt += _common * typed_ndt_8_51[i].t \
      * (typed_ndt_8_51[i].t - 1.0) / (t*t)
    out_phir_dt += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d \
      * typed_ndt_8_51[i].t / t

  _c1 = -d * d
  _c2 = exp(_c1)
  _c_coeff = 2.0
  for i in range(15,35):
    # Compute common factors
    _common = typed_ndt_8_51[i].n \
      * pow_fd(d, typed_ndt_8_51[i].d) * pow_fd(t, typed_ndt_8_51[i].t) * _c2
    # Apply derivative operators
    out_phir += _common
    out_phir_d += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d
    out_phir_dd += _common * ((typed_ndt_8_51[i].d + _c_coeff * _c1) \
        * (typed_ndt_8_51[i].d + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / (d*d)
    out_phir_t += _common * typed_ndt_8_51[i].t / t
    out_phir_tt += _common * typed_ndt_8_51[i].t \
      * (typed_ndt_8_51[i].t - 1.0) / (t*t)
    out_phir_dt += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d \
      * typed_ndt_8_51[i].t / t

  _c1 = -d * d * d
  _c2 = exp(_c1)
  _c_coeff = 3.0
  for i in range(35,39):
    # Compute common factors
    _common = typed_ndt_8_51[i].n \
      * pow_fd(d, typed_ndt_8_51[i].d) * pow_fd(t, typed_ndt_8_51[i].t) * _c2
    # Apply derivative operators
    out_phir += _common
    out_phir_d += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d
    out_phir_dd += _common * ((typed_ndt_8_51[i].d + _c_coeff * _c1) \
        * (typed_ndt_8_51[i].d + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / (d*d)
    out_phir_t += _common * typed_ndt_8_51[i].t / t
    out_phir_tt += _common * typed_ndt_8_51[i].t \
      * (typed_ndt_8_51[i].t - 1.0) / (t*t)
    out_phir_dt += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d \
      * typed_ndt_8_51[i].t / t

  _c1 = d * d
  _c1 *= -_c1
  _c2 = exp(_c1)
  _c_coeff = 4.0
  for i in range(39,40):
    # Compute common factors
    _common = typed_ndt_8_51[i].n \
      * pow_fd(d, typed_ndt_8_51[i].d) * pow_fd(t, typed_ndt_8_51[i].t) * _c2
    # Apply derivative operators
    out_phir += _common
    out_phir_d += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d
    out_phir_dd += _common * ((typed_ndt_8_51[i].d + _c_coeff * _c1) \
        * (typed_ndt_8_51[i].d + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / (d*d)
    out_phir_t += _common * typed_ndt_8_51[i].t / t
    out_phir_tt += _common * typed_ndt_8_51[i].t \
      * (typed_ndt_8_51[i].t - 1.0) / (t*t)
    out_phir_dt += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d \
      * typed_ndt_8_51[i].t / t

  _c1 = d * d * d
  _c1 *= -_c1
  _c2 = exp(_c1)
  _c_coeff = 6.0
  for i in range(40,44):
    # Compute common factors
    _common = typed_ndt_8_51[i].n \
      * pow_fd(d, typed_ndt_8_51[i].d) * pow_fd(t, typed_ndt_8_51[i].t) * _c2
    # Apply derivative operators
    out_phir += _common
    out_phir_d += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d
    out_phir_dd += _common * ((typed_ndt_8_51[i].d + _c_coeff * _c1) \
        * (typed_ndt_8_51[i].d + _c_coeff * _c1 - 1.0) \
        + _c_coeff * _c_coeff * _c1 ) / (d*d)
    out_phir_t += _common * typed_ndt_8_51[i].t / t
    out_phir_tt += _common * typed_ndt_8_51[i].t \
      * (typed_ndt_8_51[i].t - 1.0) / (t*t)
    out_phir_dt += _common * (typed_ndt_8_51[i].d + _c_coeff * _c1) / d \
      * typed_ndt_8_51[i].t / t
  
  # Compute Gaussian terms for 1-indices 52 to 54
  for i in range(51,54):
    # Compute commons
    _c1 = d - eps_res52_54[i-51]     # d_shift
    _c2 = t - gamma_res52_54[i-51]   # t_shift
    _common = n_res[i] * exp(-alpha_res52_54[i-51] * _c1 * _c1 \
      -beta_res52_54[i-51] * _c2 * _c2) \
      * (d * d # unrolled d ** (d_res[i]-1.0)
        ) * (t ** t_res[i])
    # Compute d derivative path
    out_phir += _common * d
    out_phir_d += _common * (d_res[i] \
      - 2.0 * alpha_res52_54[i-51] * d * _c1)
    out_phir_dd += _common * d * (-2.0 * alpha_res52_54[i-51]
      + 4.0 * alpha_res52_54[i-51] * alpha_res52_54[i-51]
        * _c1 * _c1
      - 4.0 * d_res[i] * alpha_res52_54[i-51] / d * _c1
      + d_res[i] * (d_res[i] - 1.0) / (d * d))
    # Compute t derivative path
    out_phir_t += _common * d / t \
      * (t_res[i] - 2.0 * beta_res52_54[i-51] * t * _c2)
    _c3 = t_res[i] - 2.0 * beta_res52_54[i-51] * t \
      * (t - gamma_res52_54[i-51])
    out_phir_tt += _common * d / (t * t) \
      * (_c3 * _c3 - t_res[i] - 2.0 * beta_res52_54[i-51] * t * t)
    # Compute mixed derivative
    out_phir_dt += _common / t \
      * (d_res[i] - 2.0 * alpha_res52_54[i-51] * d * _c1) \
      * (t_res[i] - 2.0 * beta_res52_54[i-51] * t * _c2)
  
  # Declare temporary registers
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _dDelta
  cdef DTYPE_t _ddDelta
  cdef bint phir_tt_isinf = False
    
  # Compute nonanalytical terms for 1-indices 55 to 56
  for i in range(54,56):
    # Compute commons
    _c1 = d_quad ** (_exp1_55_56[i-54] - 1.0) # _exp1_55_56 vals: 5/3
    _c2 = d_quad ** (a_res55_56[i-54] - 1.0)
    _theta = (1.0 - t) + A_res55_56[i-54] * _c1 * d_quad
    _Delta = _theta*_theta + B_res55_56[i-54] * _c2 * d_quad
    _common = n_res[i] * exp(-C_res55_56[i-54] * d_quad \
      - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))
    # Replace limiting value of Delta**(b-2) if Delta == 0
    if _Delta != 0.0:
      _common *= _Delta ** (b_res55_56[i-54] - 2.0) # 0.85 to 0.95
    else:
      _common = 0.0

    # Compute d derivative path
    out_phir += _common * d * _Delta * _Delta
    # Compute phir_d term
    out_phir_d += _common * _Delta * (
      _Delta * (1.0 - 2.0 * C_res55_56[i-54] * (d-1.0) * d)
      + b_res55_56[i-54] * d * (d-1.0) * (
        A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54] * _c1
        + 2.0 * B_res55_56[i-54] * a_res55_56[i-54] * _c2
      )
    )
    # Compute d(Delta)/d(delta) divided by (delta - 1.0) for numerical stability
    _dDelta = (A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54] * _c1
      + 2.0 * B_res55_56[i-54] * a_res55_56[i-54] * _c2)
    # Compute second derivative of Delta
    _c3 = A_res55_56[i-54] / beta_res55_56[i-54] * _c1
    _ddDelta = _dDelta + (
      4.0 * B_res55_56[i-54] * a_res55_56[i-54] * (a_res55_56[i-54] - 1.0) * _c2
      + 2.0 * _c3 * _c3 * d_quad
      + 4.0 * _theta * A_res55_56[i-54] / beta_res55_56[i-54] \
        * (_exp1_55_56[i-54] - 1.0) * _c1
    )
    # Finish d(Delta)/d(delta) computation in-place
    _dDelta *= d - 1.0
    # Compute phir_dd term
    _c2 = _Delta*_Delta * (-4.0 * C_res55_56[i-54] * (d-1.0) 
      + d * (2.0*C_res55_56[i-54]*d_quad - 1.0) * 2.0 * C_res55_56[i-54])
    _c2 += _Delta * 2.0 * b_res55_56[i-54] * _dDelta \
      * (1.0 - 2.0 * d * C_res55_56[i-54] * (d - 1.0))
    _c2 += b_res55_56[i-54] * (_Delta * _ddDelta
      + (b_res55_56[i-54] - 1.0) * _dDelta * _dDelta) * d
    out_phir_dd += _c2 * _common

    # Compute t derivative path
    out_phir_t += _common * 2.0 * d * (
      -_theta * b_res55_56[i-54] + _Delta * D_res55_56[i-54] * (1.0 - t)
      ) * _Delta
    # Compute phir_tt term
    # Replace limiting value if Delta == 0
    if _Delta == 0.0:
      phir_tt_isinf = True    
    out_phir_tt += _common * 2.0 * d * (
      b_res55_56[i-54] * (_Delta \
        + 2.0 * _theta*_theta * (b_res55_56[i-54] - 1.0)
        + 4.0 * _theta * _Delta * D_res55_56[i-54] * (t - 1.0))
      + _Delta * _Delta * D_res55_56[i-54] \
        * (2.0 * D_res55_56[i-54] * (t - 1.0) * (t - 1.0) - 1.0)
    )

    # Compute mixed derivative
    out_phir_dt += _common * (
      _Delta * _Delta * (-2.0 * D_res55_56[i-54] * (t - 1.0) \
      + d * 4.0 * C_res55_56[i-54] * D_res55_56[i-54] * (d - 1.0) * (t - 1.0))
      + d * _Delta * b_res55_56[i-54] * _dDelta \
        * (-2.0 * D_res55_56[i-54] * (t - 1.0))
      - 2.0 * _theta * b_res55_56[i-54] * _Delta \
        * (1.0 - 2.0 * d * C_res55_56[i-54] * (d - 1.0))
      + d * (
        -A_res55_56[i-54] * b_res55_56[i-54] * 2.0 / beta_res55_56[i-54] \
          * _Delta * (d - 1.0) * _c1
        - 2.0 * _theta * b_res55_56[i-54] * (b_res55_56[i-54] - 1.0) * _dDelta
      )
    )

  if phir_tt_isinf:
    out_phir_tt = -float("inf")
  return Derivatives_0_1_2(out_phir, out_phir_d, out_phir_dd,
    out_phir_t, out_phir_tt, out_phir_dt)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def p(DTYPE_t rho, DTYPE_t T) -> DTYPE_t:
  cdef DTYPE_t d = rho / rhoc
  cdef DTYPE_t t = Tc / T
  cdef DTYPE_t _phir_d, _phir_dd
  cdef DTYPE_t _c0
  cdef DTYPE_t dsatl = 1.0
  cdef DTYPE_t dsatv = 0
  cdef DTYPE_t sat_atol = 0.5e-2
  cdef Pair pair
  cdef SatTriple sat_triple

  # Compute approximate saturation curve
  cdef unsigned short i
  if t > 1.0:
    for i in range(6):
      _c0 = 1.0-1.0/t
      dsatl += satl_coeffsb[i] * _c0**satl_powsb[i]
      dsatv += satv_coeffsc[i] * _c0**satv_powsc[i]
    dsatv = exp(dsatv)

    # Check if in or near phase equilibrium region
    if d < dsatl + sat_atol and d > dsatv - sat_atol:
      # Compute precise saturation curve and saturation pressure
      sat_triple = prho_sat(T)
      if d <= sat_triple.rho_satl / rhoc and d >= sat_triple.rho_satv / rhoc:
        return sat_triple.psat

    # TODO: Near-critical-point treatment
    pass

  # Pure phase pressure computation
  pair = fused_phir_d_phir_dd(d, t)
  _phir_d = pair.first
  _phir_dd = pair.second
  return rho * R * T * (1.0 + d * _phir_d)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def u(DTYPE_t rho, DTYPE_t T) -> DTYPE_t:
  ''' Energy per unit mass (SI -- J / kg). '''
  cdef DTYPE_t d = rho / rhoc
  cdef DTYPE_t t = Tc / T
  cdef DTYPE_t _c0
  cdef DTYPE_t dsatl = 1.0
  cdef DTYPE_t dsatv = 0
  cdef DTYPE_t sat_atol = 0.5e-2
  cdef SatTriple sat_triple
  cdef DTYPE_t x


  # Compute approximate saturation curve
  cdef unsigned short i
  if t > 1.0:
    for i in range(6):
      _c0 = 1.0-1.0/t
      dsatl += satl_coeffsb[i] * _c0**satl_powsb[i]
      dsatv += satv_coeffsc[i] * _c0**satv_powsc[i]
    dsatv = exp(dsatv)

    # Check if in or near phase equilibrium region
    if d < dsatl + sat_atol and d > dsatv - sat_atol:
      # Compute precise saturation curve and saturation pressure
      sat_triple = prho_sat(T)
      dsatl = sat_triple.rho_satl / rhoc
      dsatv = sat_triple.rho_satv / rhoc
      if d <= dsatl and d >= dsatv:
        # Compute vapour mass fraction
        x = (1.0 / rho - 1.0 / sat_triple.rho_satl) \
            / (1.0 / sat_triple.rho_satv - 1.0 / sat_triple.rho_satl)
        # Return mass-weighted sum saturation energies
        return t * R * T * (
          x * (phir_t(dsatv, t) + phi0_t(dsatv, t)) \
          + (1.0-x) * (phir_t(dsatl, t) + phi0_t(dsatl, t)))

    # TODO: Near-critical-point treatment
    pass

  # Pure phase pressure computation
  return t * R * T * (phir_t(d, t) + phi0_t(d, t))

def rho_pT(p, T):
  pass
