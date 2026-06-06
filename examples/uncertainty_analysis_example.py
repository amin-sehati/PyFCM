"""
Interactive example for FCM uncertainty analysis.
Run this script from the command line: python uncertainty_analysis_example.py
"""

from pyfcm import run_uncertainty

print("\nThe file location is the path to your single-FCM adjacency matrix .xls workbook.")
print("Example: C:/MyProject/Adjacency_matrix.xls\n")

file_location = input("Paste your project file path here:  ")

print("""
Noise threshold: edges with |weight| <= threshold are removed.
Example: 0.15 removes all edges with weight in [-0.15, 0.15].
""")
noise_threshold = float(input("Noise threshold (e.g. 0.0):  "))

print("\nInference rules:  k (Kosko)  |  mk (Modified Kosko)  |  r (Rescaled)")
infer_rule = input("Inference rule:  ")

print("\nSquashing functions:  sig  |  tanh  |  bivalent  |  trivalent")
function_type = input("Squashing function:  ")

lambda_ = float(input("Lambda parameter (0 to 10):  "))

n_princ = int(input("\nHow many principle (output) concepts?  "))
principles = []
for i in range(n_princ):
    principles.append(input("  Name of principle {}:  ".format(i + 1)))

thresh = int(input("\nMaximum in-degree for a concept to be eligible as an input node:  "))
n_iteration = int(input("Number of Monte Carlo iterations:  "))

df = run_uncertainty(file_location, noise_threshold, infer_rule, function_type,
                     lambda_, principles, thresh, n_iteration)

print("\nResults saved. DataFrame shape:", df.shape)
