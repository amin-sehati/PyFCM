"""
Interactive example for FCM clustering.
Run this script from the command line: python clustering_example.py
"""

from pyfcm import cluster

print("\nThe file location is the path to your .xls workbook containing all participants' adjacency matrices.")
print("Example: C:/MyProject/AllParticipants_Adjacency_Matrix.xls\n")

file_location = input("Paste your project file path here:  ")

print("""
Reference FCM generation methods:
  AMI  –  Arithmetic mean including zeros
  AMX  –  Arithmetic mean excluding zeros
  Z   –  Zero matrix
  O   –  Ones matrix
""")
aggregation_technique = input("How to generate the reference FCM? (AMI / AMX / Z / O):  ")

print("""
Clustering criteria:
  S  –  Structural similarity (spectral)
  D  –  Dynamic similarity
""")
clustering_method = input("Which clustering criterion? (S / D):  ")

n_clusters = int(input("\nHow many clusters?  "))

function_type = "tanh"
infer_rule = "k"
if clustering_method == "D":
    print("\nSquashing functions: sig, tanh, bivalent, trivalent")
    function_type = input("Squashing function type:  ")
    print("Inference rules: k (Kosko), mk (Modified Kosko), r (Rescaled)")
    infer_rule = input("Inference rule:  ")

clusters, assignments = cluster(file_location, aggregation_technique, clustering_method,
                                n_clusters, function_type=function_type, infer_rule=infer_rule)
