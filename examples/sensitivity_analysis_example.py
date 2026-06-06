"""
Interactive example for FCM sensitivity analysis.
Run this script from the command line: python sensitivity_analysis_example.py
"""

from pyfcm import run_sensitivity

print("\nThe file location is the path to your single-FCM adjacency matrix .xls workbook.")
print("Example: C:/MyProject/Adjacency_matrix.xls\n")

file_location = input("Paste your project file path here:  ")

print("""
Noise threshold: edges with |weight| <= threshold are removed.
Example: 0.15 removes all edges with weight in [-0.15, 0.15].
""")
noise_threshold = float(input("Noise threshold (e.g. 0.1):  "))

print("\nInference rules:  k (Kosko)  |  mk (Modified Kosko)  |  r (Rescaled)")
infer_rule = input("Inference rule:  ")

print("\nSquashing functions:  sig  |  tanh  |  bivalent  |  trivalent")
function_type = input("Squashing function:  ")

lambda_ = float(input("Lambda parameter (0 to 10):  "))

n_princ = int(input("\nHow many principle (output) concepts?  "))
principles = []
for i in range(n_princ):
    principles.append(input("  Name of principle {}:  ".format(i + 1)))

n_var = int(input("\nHow many scenario variables to sweep?  "))
scenario_variables = []
for i in range(n_var):
    scenario_variables.append(input("  Name of variable {}:  ".format(i + 1)))

run_sensitivity(file_location, noise_threshold, infer_rule, function_type,
                lambda_, principles, scenario_variables)
