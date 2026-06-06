"""
Interactive example for FCM clustering.
Run this script from the command line: python clustering_example.py
"""

from pyfcm import cluster

print("\nThe file location is the path to your Excel file containing all participants' adjacency matrices.")
print("Example: C:/MyProject/AllParticipants_Adjacency_Matrix.xlsx\n")

file_location = input("Paste your project file path here:  ")

print("""
Reference FCM generation methods:
  AI  –  Average of all FCMs (including zeros)
  AX  –  Average of all FCMs (excluding zeros)
  Z   –  Zero matrix
  O   –  Ones matrix
""")
aggregation_technique = input("How to generate the reference FCM? (AI / AX / Z / O):  ")

print("""
Clustering criteria:
  S  –  Structural similarity (spectral)
  D  –  Dynamic similarity
""")
clustering_method = input("Which clustering criterion? (S / D):  ")

n_clusters = int(input("\nHow many clusters?  "))

f_type = "tanh"
infer_rule = "k"
if clustering_method == "D":
    print("\nSquashing functions: sig, tanh, bivalent, trivalent")
    f_type = input("Squashing function type:  ")
    print("Inference rules: k (Kosko), mk (Modified Kosko), r (Rescaled)")
    infer_rule = input("Inference rule:  ")

clusters, assignments = cluster(file_location, aggregation_technique, clustering_method,
                                n_clusters, f_type=f_type, infer_rule=infer_rule)
