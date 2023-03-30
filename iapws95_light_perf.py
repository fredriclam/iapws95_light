''' High-performance version of iapws95 routines via Cython.
'''

import iapws95_light
import timeit

try:
  import float_phi_functions as cfuncs
except ModuleNotFoundError as e:
  raise ModuleNotFoundError("Could not find float_phi_functions in local " +
    "directory. The module must be compiled locally using Cython. See " +
    "the repository readme for more info. For a portable version, use " +
    "iapws95_light.") from e

''' Verification utilities. '''

def print_verification_values():
  ''' Prints verification values (p. 436 of Wagner and Pruss)'''
  names0 = ["phi0   ", "phi0_d ", "phi0_dd", "phi0_t ", "phi0_tt", "phi0_dt"]
  namesr = ["phir   ", "phir_d ", "phir_dd", "phir_t ", "phir_tt", "phir_dt"]

  def print_table(results0, resultsr):
    ''' Prints results0, resultsr (each length 6) into a fixed format table.'''
    print("===============================================")
    for tup in zip(names0, results0, namesr, resultsr):
      print(f"{tup[0]} | " +
        f"{tup[1]:{'.9f' if tup[1] < 0 else ' .9f'}} | " +
        f"{tup[2]} | " +
        f"{tup[3]:{'.9f' if tup[3] < 0 else ' .9f'}}")
  
  print("Test case 1: rho = 838.025 kg m^{-3}, T = 500 K")
  rho = 838.025
  T = 500
  # Evaluate results using this implementation
  d = rho / iapws95_light.rhoc
  t = iapws95_light.Tc / T
  results0 = [cfuncs.phi0(d,t),
    cfuncs.phi0_d(d,t),
    cfuncs.phi0_dd(d,t),
    cfuncs.phi0_t(d,t),
    cfuncs.phi0_tt(d,t),
    cfuncs.phi0_dt(d,t)]
  resultsr = [cfuncs.phir(d,t),
    cfuncs.phir_d(d,t),
    cfuncs.phir_dd(d,t),
    cfuncs.phir_t(d,t),
    cfuncs.phir_tt(d,t),
    cfuncs.phir_dt(d,t)]
  # Print results
  print("Computed: ")
  print_table(results0, resultsr)
  print("Reference (9 significant figures): ")
  ref0 = [0.204_797_734e1, 0.384_236_747, -0.147_637_878,
          0.904_611_106e1, -0.193_249_185e1, 0]
  refr = [-0.342_693_206e1, -0.364_366_650, 0.856_063_701,
        -0.581_403_435e1, -0.223_440_737e1, -0.112_176_915e1]
  print_table(ref0, refr)

  print("")
  print("Test case 2: rho = 358 kg m^{-3}, T = 647 K")
  rho = 358
  T = 647
  # Evaluate results using this implementation
  d = rho / iapws95_light.rhoc
  t = iapws95_light.Tc / T
  results0 = [cfuncs.phi0(d,t),
    cfuncs.phi0_d(d,t),
    cfuncs.phi0_dd(d,t),
    cfuncs.phi0_t(d,t),
    cfuncs.phi0_tt(d,t),
    cfuncs.phi0_dt(d,t)]
  resultsr = [cfuncs.phir(d,t),
    cfuncs.phir_d(d,t),
    cfuncs.phir_dd(d,t),
    cfuncs.phir_t(d,t),
    cfuncs.phir_tt(d,t),
    cfuncs.phir_dt(d,t)]
  # Print results
  print("Computed: ")
  print_table(results0, resultsr)
  print("Reference (9 significant figures): ")
  ref0 = [-0.156_319_605e1, 0.899_441_341, -0.808_994_726,
    0.980_343_918e1, -0.343_316_334e1, 0]
  refr = [-0.121_202_657e1, -0.714_012_024, 0.475_730_696,
    -0.321_722_501e1, -0.996_029_507e1, -0.133_214_720e1]
  print_table(ref0, refr)

def print_timing():
  ''' Prints timing values as compared to ideal gas computations. '''
  print(f"Timing iapws95_light_perf calculations for scalar input.")
  rho = 358
  T = 647
  d = rho / iapws95_light.rhoc
  t = iapws95_light.Tc / T

  f0_list = [lambda: cfuncs.phi0(d,t),
    lambda: cfuncs.phi0_d(d,t),
    lambda: cfuncs.phi0_dd(d,t),
    lambda: cfuncs.phi0_t(d,t),
    lambda: cfuncs.phi0_tt(d,t),
    lambda: cfuncs.phi0_dt(d,t)]
  fr_list = [lambda: cfuncs.phir(d,t),
    lambda: cfuncs.phir_d(d,t),
    lambda: cfuncs.phir_dd(d,t),
    lambda: cfuncs.phir_t(d,t),
    lambda: cfuncs.phir_tt(d,t),
    lambda: cfuncs.phir_dt(d,t)]

  # Main routines
  names0 = ["phi0   ", "phi0_d ", "phi0_dd", "phi0_t ", "phi0_tt", "phi0_dt"]
  namesr = ["phir   ", "phir_d ", "phir_dd", "phir_t ", "phir_tt", "phir_dt"]
  N_runs = 10000
  t_f0 = [timeit.timeit(f, number=N_runs)/N_runs for f in f0_list]
  t_fr = [timeit.timeit(f, number=N_runs)/N_runs for f in fr_list]
  # Ideal gas for comparison
  t_ig = timeit.timeit(lambda: rho * iapws95_light.R * T, number=N_runs)/N_runs
  # Optimized fused routines
  t_phir_d_phir_dd = timeit.timeit(
    lambda: cfuncs.fused_phir_d_phir_dd(d,t), number=N_runs)/N_runs

  for name, t in zip(names0, t_f0):
    print(f"{name}      : {t*1e6:.2f} us")
  for name, t in zip(namesr, t_fr):
    print(f"{name}      : {t*1e6:.2f} us")
  print(f"Ideal gas    : {t_ig * 1e6:.2f} us")
  print(f"phi_d+_dd    : {t_phir_d_phir_dd * 1e6:.2f} us")