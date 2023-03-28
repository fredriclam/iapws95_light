''' High-performance version of iapws95 routines via Cython.
'''

import iapws95_light
try:
  import float_phi_functions as cfuncs
except ModuleNotFoundError as e:
  raise ModuleNotFoundError("Could not find float_phi_functions in local " +
    "directory. The module must be compiled locally using Cython. See " +
    "the repository readme for more info.") from e

''' Verification utilities. '''

def print_verification_values():
  ''' Prints verification values (p. 436 of Wagner and Pruss)'''
  namesr = ["phir_d ", "phir_dd"]

  def print_table(resultsr):
    print("===============================================")
    for tup in zip(namesr, resultsr):
      print(f"{tup[0]} | " +
        f"{tup[1]:{'.9f' if tup[1] < 0 else ' .9f'}} | ")
  
  print("Test case 1: rho = 838.025 kg m^{-3}, T = 500 K")
  rho = 838.025
  T = 500
  # Evaluate results using this implementation
  resultsr = cfuncs.fused_phir_d_phir_dd(
    rho/iapws95_light.rhoc, iapws95_light.Tc/T)
  # Print results
  print("Computed: ")
  print_table(resultsr)
  print("Reference (9 significant figures): ")
  refr = [-0.364_366_650, 0.856_063_701]
  print_table(refr)

  print("")
  print("Test case 2: rho = 358 kg m^{-3}, T = 647 K")
  rho = 358
  T = 647
  # Evaluate results using this implementation
  resultsr = cfuncs.fused_phir_d_phir_dd(
    rho/iapws95_light.rhoc, iapws95_light.Tc/T)
  # Print results
  print("Computed: ")
  print_table(resultsr)
  print("Reference (9 significant figures): ")
  refr = [-0.714_012_024, 0.475_730_696]
  print_table(refr)