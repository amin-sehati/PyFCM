"""
Interactive example for FCM aggregation.
Run this script from the command line: python aggregation_example.py
"""

from pyfcm import aggregate, visualize_aggregated

print("\nThe file location is the path to your .xls workbook containing all participants' adjacency matrices.")
print("Example: C:/MyProject/AllParticipants_Adjacency_Matrix.xls")
print("Each sheet in the file should contain one participant's matrix.\n")

file_location = input("Paste your project file path here:  ")

print("""
Aggregation techniques:
  AMX  –  Arithmetic mean excluding zero edges
  AMI  –  Arithmetic mean including zero edges
  MED  –  Median
  GM   –  Geometric mean (non-zero edges only)
""")
aggregation_technique = input("Which aggregation method? (AMX / AMI / MED / GM):  ")

Adj_aggregated_FCM, sheet, n_concepts = aggregate(file_location, aggregation_technique)

print("\nAggregated adjacency matrix:")
print(Adj_aggregated_FCM)

visualize_aggregated(Adj_aggregated_FCM, sheet)
