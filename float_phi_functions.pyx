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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def phir_d(DTYPE_t d, DTYPE_t t):
  ''' First delta-derivative of residual part of dimless Helmholtz function
      phi = f/(RT).
  See also phir for more details.
  Cython implementation for float input.
  '''

  # Precompute quantities
  cdef DTYPE_t d_quad = (d - 1.0) * (d - 1.0)
  # Allocate and partially evaluate coeffs
  cdef DTYPE_t coeffs[56]
  cdef DTYPE_t theta
  cdef DTYPE_t Delta
  cdef unsigned int i
  # Compute pre-exponential coefficients
  for i in range(56):
    coeffs[i] = n_res[i] * (d ** (d_res[i]-1.0)) * (t ** t_res[i])
  for i in range(51):
    coeffs[i] *= (d_res[i] - c_res1_51[i] * d ** c_res1_51[i])
  for i in range(51,54):
    coeffs[i] *= d_res[i] - 2.0 * alpha_res52_54[i-51] * d * (d - eps_res52_54[i-51])
  for i in range(54,56):
    theta = (1.0 - t) + A_res55_56[i-54] * d_quad ** _exp1_55_56[i-54]
    Delta = theta*theta + B_res55_56[i-54] * d_quad ** a_res55_56[i-54]
    coeffs[i] *= (
      Delta * (1.0 - 2.0 * C_res55_56[i-54] * (d-1.0) * d)
      + b_res55_56[i-54] * d * (d-1.0) * (
        A_res55_56[i-54] * theta * 2.0 / beta_res55_56[i-54]
        * d_quad**(_exp1_55_56[i-54] - 1.0)
        + 2.0 * B_res55_56[i-54] * a_res55_56[i-54]
        * d_quad**(a_res55_56[i-54] - 1.0)
      )
    )
    if Delta != 0:
      Delta = Delta ** (b_res55_56[i-54]-1.0)
    coeffs[i] *= Delta

  # Allocate exponent cache
  cdef DTYPE_t exponents[56]
  cdef DTYPE_t _d_shift
  cdef DTYPE_t _t_shift
  # Compute exponents
  for i in range(7):
    exponents[i] = 0.0
  for i in range(7,51):
    exponents[i] = -d ** c_res1_51[i]
  for i in range(51,54):
    _d_shift = d - eps_res52_54[i-51]
    _t_shift = t - gamma_res52_54[i-51]
    exponents[i] = -alpha_res52_54[i-51] * _d_shift * _d_shift \
      -beta_res52_54[i-51] * _t_shift * _t_shift
  for i in range(54,56):
    exponents[i] = -C_res55_56[i-54] * d_quad \
      - D_res55_56[i-54]*(t - 1.0)*(t - 1.0)

  cdef DTYPE_t out_dot = 0.0
  for i in range(56):
    out_dot += coeffs[i] * exp(exponents[i])
  return out_dot