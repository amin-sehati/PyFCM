"""
Interactive example for FCM scenario analysis.
Run this script from the command line: python scenario_analysis_example.py
"""

from pyfcm import run_scenario

print("\nThe file location is the path to your single-FCM adjacency matrix Excel file.")
print("Example: C:/MyProject/Adjacency_matrix.xlsx\n")

file_location = input("Paste your project file path here:  ")

print("""
Noise threshold: edges with |weight| <= threshold are removed.
Example: 0.15 removes all edges with weight in [-0.15, 0.15].
""")
noise_threshold = float(input("Noise threshold (e.g. 0.1):  "))

print("\nInference rules:  k (Kosko)  |  mk (Modified Kosko)  |  r (Rescaled)")
infer_rule = input("Inference rule:  ")

print("\nSquashing functions:  sig  |  tanh  |  biv  |  triv")
function_type = input("Squashing function:  ")

lambda_ = float(input("Lambda parameter (0 to 10):  "))

n_princ = int(input("\nHow many principle (output) concepts?  "))
principles = []
for i in range(n_princ):
    principles.append(input("  Name of principle {}:  ".format(i + 1)))

n_var = int(input("\nHow many scenario variables to activate?  "))
change_levels = {}
for i in range(n_var):
    name = input("  Name of variable {}:  ".format(i + 1))
    level = float(input("  Activation level for {} (between -1 and 1):  ".format(name)))
    change_levels[name] = level

show = input("\nShow results for All concepts (A) or Principles only (P)?  ").strip().upper()

changes = run_scenario(file_location, noise_threshold, infer_rule, function_type,
                       lambda_, principles, change_levels, show=show)

print("\nChanges in all concepts:")
for concept, change in changes.items():
    print("  {}: {:.4f}".format(concept, change))
