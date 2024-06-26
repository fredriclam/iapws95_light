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
      _str_results0 = (round(tup[1], 9) if abs(tup[1]) < 1.0
        else round(tup[1], 8))
      _str_resultsr = (round(tup[3], 9) if abs(tup[3]) < 1.0
        else round(tup[3], 8))
      print(f"{tup[0]} | " +
        f"{_str_results0:{'.9f' if tup[1] < 0 else ' .9f'}} | " +
        f"{tup[2]} | " +
        f"{_str_resultsr:{'.9f' if tup[3] < 0 else ' .9f'}}")
  
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
  print("Computed (rounded to 9 significant figures): ")
  print_table(results0, resultsr)
  print("Reference (9 significant figures): ")
  ref0 = [0.204_797_734e1, 0.384_236_747, -0.147_637_878,
          0.904_611_106e1, -0.193_249_185e1, 0]
  refr = [-0.342_693_206e1, -0.364_366_650, 0.856_063_701,
        -0.581_403_435e1, -0.223_440_737e1, -0.112_176_915e1]
  print_table(ref0, refr)

  # Compute internal consistency of phir_fused_all vs. individual funcs
  output_phir_fused = cfuncs.fused_phir_all(d,t)
  resultsr_fused = [output_phir_fused[key] for key in
    ["phir", "phir_d", "phir_dd", "phir_t", "phir_tt", "phir_dt"]]
  func_differences = [abs(resultsr_fused[i] - resultsr[i])
    for i in range(len(resultsr))]
  func_reldiff = [abs(resultsr_fused[i]/resultsr[i] - 1.0)
    for i in range(len(resultsr))]
  print("Max abs difference of phir_fused_all and individual functions: " +
    f"{max(func_differences):.5e}.")
  print("Max rel difference of phir_fused_all and individual functions: " +
    f"{max(func_reldiff):.5e}.")  

  # Compute internal consistency of phi0_fused_all vs. individual funcs
  output_phi0_fused = cfuncs.fused_phi0_all(d,t)
  results0_fused = [output_phi0_fused[key] for key in
    ["phi0", "phi0_d", "phi0_dd", "phi0_t", "phi0_tt", "phi0_dt"]]
  func_differences = [abs(results0_fused[i] - results0[i])
    for i in range(len(results0))]
  print("Max abs difference of phi0_fused_all and individual functions: " +
    f"{max(func_differences):.5e}.")


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
  print("Computed (rounded to 9 significant figures): ")
  print_table(results0, resultsr)
  print("Reference (9 significant figures): ")
  ref0 = [-0.156_319_605e1, 0.899_441_341, -0.808_994_726,
    0.980_343_918e1, -0.343_316_334e1, 0]
  refr = [-0.121_202_657e1, -0.714_012_024, 0.475_730_696,
    -0.321_722_501e1, -0.996_029_507e1, -0.133_214_720e1]
  print_table(ref0, refr)

  # Compute internal consistency of phir_fused_all vs. individual funcs
  output_phir_fused = cfuncs.fused_phir_all(d,t)
  resultsr_fused = [output_phir_fused[key] for key in
    ["phir", "phir_d", "phir_dd", "phir_t", "phir_tt", "phir_dt"]]
  func_differences = [abs(resultsr_fused[i] - resultsr[i])
    for i in range(len(resultsr))]
  func_reldiff = [abs(resultsr_fused[i]/resultsr[i] - 1.0)
    for i in range(len(resultsr))]
  print("Max abs difference of phir_fused_all and individual functions: " +
    f"{max(func_differences):.5e}.")
  print("Max rel difference of phir_fused_all and individual functions: " +
    f"{max(func_reldiff):.5e}.")

  # Compute internal consistency of phi0_fused_all vs. individual funcs
  output_phi0_fused = cfuncs.fused_phi0_all(d,t)
  results0_fused = [output_phi0_fused[key] for key in
    ["phi0", "phi0_d", "phi0_dd", "phi0_t", "phi0_tt", "phi0_dt"]]
  func_differences = [abs(results0_fused[i] - results0[i])
    for i in range(len(results0))]
  print("Max abs difference of phi0_fused_all and individual functions: " +
    f"{max(func_differences):.5e}.")

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
  t_ig_sqrt = timeit.timeit(lambda: (rho * T)**0.5, number=N_runs)/N_runs
  t_ig_noop = timeit.timeit(lambda: None, number=N_runs)/N_runs

  print(f"=== Individual routines ===")
  for name, t in zip(names0, t_f0):
    print(f"{name}      : {t*1e6:.2f} us")
  for name, t in zip(namesr, t_fr):
    print(f"{name}      : {t*1e6:.2f} us")
  print(f"=== Reference ops (pure python) ===")
  print(f"rho * R * T  : {t_ig * 1e6:.2f} us")
  print(f"(rho*T)**.5  : {t_ig_sqrt * 1e6:.2f} us")
  print(f"lambda no-op : {t_ig_noop * 1e6:.2f} us")

  # Optimized fused routines
  print(f"=== Optimized routines ===")
  # t_phir_d_phir_dd = timeit.timeit(
  #   lambda: cfuncs.fused_phir_d_phir_dd(d,t), number=N_runs)/N_runs
  # print(f"phir_d+_dd   : {t_phir_d_phir_dd * 1e6:.2f} us")
  t_phir_all = timeit.timeit(
    lambda: cfuncs.fused_phir_all(d,t), number=N_runs)/N_runs
  print(f"phir_*       : {t_phir_all * 1e6:.2f} us")
  t_phi0_all = timeit.timeit(
    lambda: cfuncs.fused_phi0_all(d,t), number=N_runs)/N_runs
  print(f"phi0_*       : {t_phi0_all * 1e6:.2f} us")
  t_phir_d_dd = timeit.timeit(
    lambda: cfuncs.fused_phir_d_phir_dd(d,t), number=N_runs)/N_runs
  print(f"phir_d_dd    : {t_phir_d_dd * 1e6:.2f} us")
