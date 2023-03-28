

''' # For cython timing: place at top
# cython: binding=True
# cython: linetrace=True
# cython: profile=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1'''

ctypedef double DTYPE_t
cimport cython

cdef extern from "math.h":
    double exp(double x)

cdef DTYPE_t[2] a_res55_56 = [3.5, 3.5]
cdef DTYPE_t[2] A_res55_56 = [0.32, 0.32]
cdef DTYPE_t[3] alpha_res52_54 = [20., 20., 20.]
cdef DTYPE_t[2] b_res55_56 = [0.85, 0.95]
cdef DTYPE_t[2] B_res55_56 = [0.2, 0.2]
cdef DTYPE_t[3] beta_res52_54 = [150., 150., 250.]
cdef DTYPE_t[2] beta_res55_56 = [0.3, 0.3]
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
cdef DTYPE_t R = 461.51805
cdef DTYPE_t[56] n_res = [
  1.25335479e-02,  7.89576347e+00, -8.78032033e+00,  3.18025093e-01,
  -2.61455339e-01, -7.81997517e-03,  8.80894931e-03, -6.68565723e-01,
  2.04338110e-01, -6.62126050e-05, -1.92327212e-01, -2.57090430e-01,
  1.60748685e-01, -4.00928289e-02,  3.93434226e-07, -7.59413771e-06,
  5.62509794e-04, -1.56086523e-05,  1.15379964e-09,  3.65821651e-07,
  -1.32511801e-12, -6.26395869e-10, -1.07936009e-01,  1.76114910e-02,
  2.21322952e-01, -4.02476698e-01,  5.80834000e-01,  4.99691470e-03,
  -3.13587007e-02, -7.43159297e-01,  4.78073299e-01,  2.05279409e-02,
  -1.36364351e-01,  1.41806344e-02,  8.33265049e-03, -2.90523360e-02,
  3.86150856e-02, -2.03934865e-02, -1.65540501e-03,  1.99555720e-03,
  1.58703083e-04, -1.63885683e-05,  4.36136157e-02,  3.49940055e-02,
  -7.67881978e-02,  2.24462773e-02, -6.26897104e-05, -5.57111186e-10,
  -1.99057184e-01,  3.17774973e-01, -1.18411824e-01, -3.13062603e+01,
  3.15461402e+01, -2.52131543e+03, -1.48746409e-01,  3.18061109e-01]
cdef DTYPE_t[56] t_res = [
  -0.5  ,  0.875,  1.   ,  0.5  ,  0.75 ,  0.375,  1.   ,  4.   ,
  6.   , 12.   ,  1.   ,  5.   ,  4.   ,  2.   , 13.   ,  9.   ,
  3.   ,  4.   , 11.   ,  4.   , 13.   ,  1.   ,  7.   ,  1.   ,
  9.   , 10.   , 10.   ,  3.   ,  7.   , 10.   , 10.   ,  6.   ,
  10.   , 10.   ,  1.   ,  2.   ,  3.   ,  4.   ,  8.   ,  6.   ,
  9.   ,  8.   , 16.   , 22.   , 23.   , 23.   , 10.   , 50.   ,
  44.   , 46.   , 50.   ,  0.   ,  1.   ,  4.   ,  0.   ,  0.   ]
cdef DTYPE_t[2] _exp1_55_56 = [1.66666667, 1.66666667]

# Fused coefficient array for 1 to 51.
#   Coefficients (n, d, t, c) are contiguous in memory
cdef DTYPE_t[204] ndtc1_51 = [
        1.25335479e-02,  1.00000000e+00, -5.00000000e-01,  0.00000000e+00,
        7.89576347e+00,  1.00000000e+00,  8.75000000e-01,  0.00000000e+00,
       -8.78032033e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
        3.18025093e-01,  2.00000000e+00,  5.00000000e-01,  0.00000000e+00,
       -2.61455339e-01,  2.00000000e+00,  7.50000000e-01,  0.00000000e+00,
       -7.81997517e-03,  3.00000000e+00,  3.75000000e-01,  0.00000000e+00,
        8.80894931e-03,  4.00000000e+00,  1.00000000e+00,  0.00000000e+00,
       -6.68565723e-01,  1.00000000e+00,  4.00000000e+00,  1.00000000e+00,
        2.04338110e-01,  1.00000000e+00,  6.00000000e+00,  1.00000000e+00,
       -6.62126050e-05,  1.00000000e+00,  1.20000000e+01,  1.00000000e+00,
       -1.92327212e-01,  2.00000000e+00,  1.00000000e+00,  1.00000000e+00,
       -2.57090430e-01,  2.00000000e+00,  5.00000000e+00,  1.00000000e+00,
        1.60748685e-01,  3.00000000e+00,  4.00000000e+00,  1.00000000e+00,
       -4.00928289e-02,  4.00000000e+00,  2.00000000e+00,  1.00000000e+00,
        3.93434226e-07,  4.00000000e+00,  1.30000000e+01,  1.00000000e+00,
       -7.59413771e-06,  5.00000000e+00,  9.00000000e+00,  1.00000000e+00,
        5.62509794e-04,  7.00000000e+00,  3.00000000e+00,  1.00000000e+00,
       -1.56086523e-05,  9.00000000e+00,  4.00000000e+00,  1.00000000e+00,
        1.15379964e-09,  1.00000000e+01,  1.10000000e+01,  1.00000000e+00,
        3.65821651e-07,  1.10000000e+01,  4.00000000e+00,  1.00000000e+00,
       -1.32511801e-12,  1.30000000e+01,  1.30000000e+01,  1.00000000e+00,
       -6.26395869e-10,  1.50000000e+01,  1.00000000e+00,  1.00000000e+00,
       -1.07936009e-01,  1.00000000e+00,  7.00000000e+00,  2.00000000e+00,
        1.76114910e-02,  2.00000000e+00,  1.00000000e+00,  2.00000000e+00,
        2.21322952e-01,  2.00000000e+00,  9.00000000e+00,  2.00000000e+00,
       -4.02476698e-01,  2.00000000e+00,  1.00000000e+01,  2.00000000e+00,
        5.80834000e-01,  3.00000000e+00,  1.00000000e+01,  2.00000000e+00,
        4.99691470e-03,  4.00000000e+00,  3.00000000e+00,  2.00000000e+00,
       -3.13587007e-02,  4.00000000e+00,  7.00000000e+00,  2.00000000e+00,
       -7.43159297e-01,  4.00000000e+00,  1.00000000e+01,  2.00000000e+00,
        4.78073299e-01,  5.00000000e+00,  1.00000000e+01,  2.00000000e+00,
        2.05279409e-02,  6.00000000e+00,  6.00000000e+00,  2.00000000e+00,
       -1.36364351e-01,  6.00000000e+00,  1.00000000e+01,  2.00000000e+00,
        1.41806344e-02,  7.00000000e+00,  1.00000000e+01,  2.00000000e+00,
        8.33265049e-03,  9.00000000e+00,  1.00000000e+00,  2.00000000e+00,
       -2.90523360e-02,  9.00000000e+00,  2.00000000e+00,  2.00000000e+00,
        3.86150856e-02,  9.00000000e+00,  3.00000000e+00,  2.00000000e+00,
       -2.03934865e-02,  9.00000000e+00,  4.00000000e+00,  2.00000000e+00,
       -1.65540501e-03,  9.00000000e+00,  8.00000000e+00,  2.00000000e+00,
        1.99555720e-03,  1.00000000e+01,  6.00000000e+00,  2.00000000e+00,
        1.58703083e-04,  1.00000000e+01,  9.00000000e+00,  2.00000000e+00,
       -1.63885683e-05,  1.20000000e+01,  8.00000000e+00,  2.00000000e+00,
        4.36136157e-02,  3.00000000e+00,  1.60000000e+01,  3.00000000e+00,
        3.49940055e-02,  4.00000000e+00,  2.20000000e+01,  3.00000000e+00,
       -7.67881978e-02,  4.00000000e+00,  2.30000000e+01,  3.00000000e+00,
        2.24462773e-02,  5.00000000e+00,  2.30000000e+01,  3.00000000e+00,
       -6.26897104e-05,  1.40000000e+01,  1.00000000e+01,  4.00000000e+00,
       -5.57111186e-10,  3.00000000e+00,  5.00000000e+01,  6.00000000e+00,
       -1.99057184e-01,  6.00000000e+00,  4.40000000e+01,  6.00000000e+00,
        3.17774973e-01,  6.00000000e+00,  4.60000000e+01,  6.00000000e+00,
       -1.18411824e-01,  6.00000000e+00,  5.00000000e+01,  6.00000000e+00]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _dummy(DTYPE_t d, DTYPE_t t):
  return d

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def phir_d(DTYPE_t d, DTYPE_t t):
  ''' First delta-derivative of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details.
  Cython implementation for float input.
  '''

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
  cdef unsigned int i
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
def phir_dd(DTYPE_t d, DTYPE_t t):
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
  cdef unsigned int i
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
def fused_phir_d_phir_dd(DTYPE_t d, DTYPE_t t):
  ''' First and second delta-derivative of residual part of dimless
    Helmholtz function
      phi = f/(RT).
  See also phir for more details.
  Cython implementation for float input.
  '''

  cdef DTYPE_t out_phir_d = 0.0
  cdef DTYPE_t out_phir_dd = 0.0
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  # Declare temporary registers
  cdef DTYPE_t _neg_dpowc
  cdef DTYPE_t _temp

  # Use strides as below
  # cdef n_coeff ndtc1_51[4*i]
  # cdef d_coeff ndtc1_51[4*i+1]
  # cdef t_coeff ndtc1_51[4*i+2]
  # cdef c_coeff ndtc1_51[4*i+3]

  # Compute uniform coefficients with mixed data (1-indices 1 to 51)
  cdef unsigned short i
  for i in range(7):
    _temp = ndtc1_51[4*i] * ndtc1_51[4*i+1] \
      * (d ** (ndtc1_51[4*i+1]-1.0)) * (t ** ndtc1_51[4*i+2])
    out_phir_d += _temp
    out_phir_dd += _temp * (ndtc1_51[4*i+1] - 1.0) / d
  for i in range(7, 51):
    _neg_dpowc = -d ** ndtc1_51[4*i+3]
    _temp = ndtc1_51[4*i] \
      * (d ** (ndtc1_51[4*i+1]-1.0)) * (t ** ndtc1_51[4*i+2]) \
      * exp(_neg_dpowc)
    out_phir_d += _temp * (ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _neg_dpowc)
    out_phir_dd += _temp * ((ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _neg_dpowc) \
        * (ndtc1_51[4*i+1] + ndtc1_51[4*i+3] * _neg_dpowc - 1.0) \
        + ndtc1_51[4*i+3] * ndtc1_51[4*i+3] * _neg_dpowc ) / d

  # Declare temporary registers
  cdef DTYPE_t _c1
  cdef DTYPE_t _c2
  cdef DTYPE_t _theta
  cdef DTYPE_t _Delta
  cdef DTYPE_t _dDelta_div
  cdef DTYPE_t _ddDelta
  # Compute heterogeneous coefficients for 1-indices 52 to 54
  for i in range(51,54):
    # Compute commons
    _c1 = d - eps_res52_54[i-51]
    _c2 = t - gamma_res52_54[i-51]
    _temp = n_res[i] * exp(-alpha_res52_54[i-51] * _c1 * _c1 \
      -beta_res52_54[i-51] * _c2 * _c2) \
      * (d ** (d_res[i]-1.0)) * (t ** t_res[i])
    # Compute phir_d term
    out_phir_d += _temp * (d_res[i] \
      - 2.0 * alpha_res52_54[i-51] * d * (d - eps_res52_54[i-51]))
    # Compute phir_dd term
    out_phir_dd += _temp * d * (-2.0 * alpha_res52_54[i-51] \
      + 4.0 * alpha_res52_54[i-51] * alpha_res52_54[i-51] \
        * (d - eps_res52_54[i-51]) * (d - eps_res52_54[i-51]) \
      - 4.0 * d_res[i] * alpha_res52_54[i-51] / d * (d - eps_res52_54[i-51]) \
      + d_res[i] * (d_res[i] - 1.0) / (d * d))
  # Compute heterogeneous coefficients for 1-indices 55 to 56
  for i in range(54,56):
    # Compute commons
    _theta = (1.0 - t) + A_res55_56[i-54] * d_quad ** _exp1_55_56[i-54]
    _Delta = _theta*_theta + B_res55_56[i-54] * d_quad ** a_res55_56[i-54]
    _temp = exp(-C_res55_56[i-54] * d_quad \
      - D_res55_56[i-54]*(t - 1.0)*(t - 1.0))

    # Compute phir_d
    _c1 = n_res[i] * (d ** (d_res[i]-1.0)) * (t ** t_res[i])
    _c1 *= (
      _Delta * (1.0 - 2.0 * C_res55_56[i-54] * (d-1.0) * d)
      + b_res55_56[i-54] * d * (d-1.0) * (
        A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54]
        * d_quad**(_exp1_55_56[i-54] - 1.0)
        + 2.0 * B_res55_56[i-54] * a_res55_56[i-54]
        * d_quad**(a_res55_56[i-54] - 1.0)
      )
    )
    if _Delta != 0.0:
      _c1 *= _Delta ** (b_res55_56[i-54]-1.0)
    else:
      _c1 = 0.0
    out_phir_d += _c1 * _temp

    # Compute phir_dd
    # Compute d(Delta)/d(delta) divided by (delta - 1.0) for numerical stability
    _dDelta_div = A_res55_56[i-54] * _theta * 2.0 / beta_res55_56[i-54] \
        * d_quad**(_exp1_55_56[i-54] - 1.0) \
      + 2.0 * B_res55_56[i-54] * a_res55_56[i-54] \
        * d_quad**(a_res55_56[i-54] - 1.0)
    if d_quad != 0.0:
      _c1 = d_quad ** (_exp1_55_56[i-54] - 2.0)
    else:
      _c1 = 0.0
    _c2 = A_res55_56[i-54] / beta_res55_56[i-54] \
      * d_quad**(_exp1_55_56[i-54] - 1.0)
    _ddDelta = _dDelta_div + ((d-1.0)*(d-1.0)) * (
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
    _c1 = _Delta*_Delta * (-4.0 * C_res55_56[i-54] * (d-1.0) 
      + d * (2.0*C_res55_56[i-54]*d_quad - 1.0) * 2.0 * C_res55_56[i-54])
    _c1 += _Delta * 2.0 * b_res55_56[i-54] * _dDelta_div \
      * (1.0 - 2.0 * d * C_res55_56[i-54] * (d - 1.0))
    _c1 += b_res55_56[i-54] * (_Delta * _ddDelta
      + (b_res55_56[i-54] - 1.0) * _dDelta_div * _dDelta_div) * d
    # Reuse register
    if _Delta != 0.0:
      _c2 = _Delta ** (b_res55_56[i-54] - 2.0)
    else:
      _c2 = 0.0
    _c1 *= n_res[i] * _c2
    out_phir_dd += _c1 * _temp

  return out_phir_d, out_phir_dd