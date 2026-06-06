# PyFCM

PyFCM is a Python package for analyzing Fuzzy Cognitive Maps (FCMs). It supports
group aggregation, participant clustering, scenario analysis, sensitivity
analysis, uncertainty analysis, and FCM visualization.

An FCM is represented as an adjacency matrix:

```text
row concept -> column concept = causal edge weight
```

Positive weights represent positive causal effects. Negative weights represent
negative causal effects. Zero means no edge or no reported relationship.

## Workbook Format

PyFCM reads legacy Excel `.xls` workbooks with `xlrd`.

The expected worksheet format is:

```text
          Concept A   Concept B   Concept C
Concept A   0.0         0.5        -0.2
Concept B  -0.1         0.0         0.7
Concept C   0.3         0.0         0.0
```

Rules:

- The first row contains concept names.
- The first column contains concept names.
- The numeric matrix starts at row 2, column 2 in Excel.
- For multi-participant workflows, each worksheet contains one participant's
  adjacency matrix.
- For clustering, the participant ID is read from cell `A1`.
- All sheets in a multi-participant workbook should use the same concepts in the
  same order.

## Install

```bash
pip install -e .
```

Or with `uv`:

```bash
uv pip install -e .
```

## Public API

The main functions are available directly from `pyfcm`:

```python
from pyfcm import (
    aggregate,
    visualize_aggregated,
    cluster,
    run_scenario,
    run_sensitivity,
    run_uncertainty,
)
```

## File Overview

### `aggregation.py`

Combines multiple participant FCMs into one group/reference FCM.

Public functions:

```python
aggregate(file_location, aggregation_technique)
```

Inputs:

- `file_location`: path to a multi-sheet `.xls` workbook.
- `aggregation_technique`:
  - `"AMI"`: arithmetic mean including zeros.
  - `"AMX"`: arithmetic mean excluding zeros.
  - `"MED"`: median.
  - `"GM"`: geometric mean of nonzero values.
  - `"AI"` and `"AX"` are accepted as backward-compatible aliases for `"AMI"`
    and `"AMX"`.

Outputs:

```python
Adj_aggregated_FCM, sheet, n_concepts
```

- `Adj_aggregated_FCM`: NumPy adjacency matrix for the group FCM.
- `sheet`: last worksheet read, used for concept labels.
- `n_concepts`: number of concepts.

Example:

```python
from pyfcm import aggregate

adj, sheet, n_concepts = aggregate("AllParticipants.xls", "AMI")
```

```python
visualize_aggregated(Adj_aggregated_FCM, sheet, save_edgelist=True)
```

Inputs:

- `Adj_aggregated_FCM`: aggregated adjacency matrix.
- `sheet`: worksheet returned by `aggregate()`.
- `save_edgelist`: if `True`, writes `aggregated_edg.csv`.

Output:

- A `networkx.DiGraph`.

Example:

```python
from pyfcm import aggregate, visualize_aggregated

adj, sheet, _ = aggregate("AllParticipants.xls", "AMX")
graph = visualize_aggregated(adj, sheet)
```

Lower-level reusable function:

```python
aggregate_matrices(matrices, aggregation_technique)
```

This aggregates an existing list of NumPy matrices without reading Excel.

### `clustering.py`

Groups participants by similarity to a reference FCM.

Public function:

```python
cluster(file_location, aggregation_technique, clustering_method, n_clusters,
        f_type="tanh", infer_rule="k", function_type=None)
```

Inputs:

- `file_location`: path to a multi-sheet `.xls` workbook.
- `aggregation_technique`:
  - `"AMI"`: average including zeros.
  - `"AMX"`: average excluding zeros.
  - `"O"`: ones matrix reference.
  - `"Z"`: zero matrix reference.
- `clustering_method`:
  - `"S"`: structural similarity using graph spectra.
  - `"D"`: dynamic similarity using repeated FCM simulations.
- `n_clusters`: number of KMeans clusters.
- `function_type`: squashing function for dynamic clustering:
  - `"sig"`
  - `"tanh"`
  - `"bivalent"`
  - `"trivalent"`
- `f_type`: backward-compatible alias for `function_type`.
- `infer_rule`:
  - `"k"`: Kosko rule.
  - `"mk"`: modified Kosko rule.
  - `"r"`: rescaled rule.

Outputs:

```python
clusters, assignments
```

- `clusters`: dictionary of cluster ID to similarity/distance values.
- `assignments`: list of `(participant_id, cluster_label)` tuples.

Example:

```python
from pyfcm import cluster

clusters, assignments = cluster(
    "AllParticipants.xls",
    aggregation_technique="AMI",
    clustering_method="S",
    n_clusters=2,
)
```

Dynamic clustering example:

```python
clusters, assignments = cluster(
    "AllParticipants.xls",
    aggregation_technique="AMX",
    clustering_method="D",
    n_clusters=3,
    function_type="tanh",
    infer_rule="k",
)
```

Notes:

- Structural clustering compares graph shape.
- Dynamic clustering applies many random scenarios to each participant FCM and
  the reference FCM. Lower distance means more similar behavior.
- The function saves `Clusters.pdf` and prints participant assignments.

### `scenario_analysis.py`

Runs one user-defined "what if" scenario on a single FCM.

Public function:

```python
run_scenario(file_location, noise_threshold, infer_rule, function_type, lambda_,
             principles, change_levels, show="A")
```

Inputs:

- `file_location`: path to a single-sheet `.xls` workbook.
- `noise_threshold`: edges with `abs(weight) <= noise_threshold` are set to 0.
- `infer_rule`: `"k"`, `"mk"`, or `"r"`.
- `function_type`: `"sig"`, `"tanh"`, `"bivalent"`, or `"trivalent"`.
- `lambda_`: squashing-function steepness.
- `principles`: concept names to track as important output concepts.
- `change_levels`: dictionary of forced scenario values:

```python
{"Concept name": activation_level}
```

- `show`:
  - `"A"`: plot all concepts.
  - `"P"`: plot only principles.

Output:

```python
changes_dic
```

Dictionary mapping concept name to change from the baseline steady state.

Example:

```python
from pyfcm import run_scenario

changes = run_scenario(
    "SingleFCM.xls",
    noise_threshold=0.1,
    infer_rule="k",
    function_type="tanh",
    lambda_=1,
    principles=["Sustainability", "Cost"],
    change_levels={"Policy Support": 0.8, "Funding": 0.5},
    show="P",
)
```

The function saves:

- `Scenario_Results.pdf`
- `Changes_In_All_Concepts.csv`

### `sensitivity_analysis.py`

Tests how important output concepts respond as selected input concepts move from
0 to 1.

Public function:

```python
run_sensitivity(file_location, noise_threshold, infer_rule, function_type,
                lambda_, principles, scenario_variables)
```

Inputs:

- `file_location`: path to a single-sheet `.xls` workbook.
- `noise_threshold`: edges with `abs(weight) <= noise_threshold` are set to 0.
- `infer_rule`: `"k"`, `"mk"`, or `"r"`.
- `function_type`: `"sig"`, `"tanh"`, `"bivalent"`, or `"trivalent"`.
- `lambda_`: squashing-function steepness.
- `principles`: output concept names to track.
- `scenario_variables`: input concept names to sweep from 0 to 1.

Output:

- No Python return value.
- Saves one PDF per scenario variable, named after that variable.

Example:

```python
from pyfcm import run_sensitivity

run_sensitivity(
    "SingleFCM.xls",
    noise_threshold=0.1,
    infer_rule="k",
    function_type="tanh",
    lambda_=1,
    principles=["Sustainability", "Cost"],
    scenario_variables=["Policy Support", "Funding"],
)
```

Plain meaning:

- Scenario analysis asks: "What happens if I force these values?"
- Sensitivity analysis asks: "How do the important outputs change as this input
  gradually increases?"

### `uncertainty_analysis.py`

Runs Monte Carlo-style random scenarios and records how selected principle
concepts respond.

Public function:

```python
run_uncertainty(file_location, noise_threshold, infer_rule, function_type,
                lambda_, principles, thresh, n_iteration)
```

Inputs:

- `file_location`: path to a single-sheet `.xls` workbook.
- `noise_threshold`: edges with `abs(weight) <= noise_threshold` are set to 0.
- `infer_rule`: `"k"`, `"mk"`, or `"r"`.
- `function_type`: `"sig"`, `"tanh"`, `"bivalent"`, or `"trivalent"`.
- `lambda_`: squashing-function steepness.
- `principles`: output concept names to track.
- `thresh`: maximum in-degree for a concept to be eligible as a randomly
  activated input node.
- `n_iteration`: number of random scenario runs.

Output:

```python
df
```

A `pandas.DataFrame` with:

- `IDS`: iteration number.
- one column per principle concept.

Example:

```python
from pyfcm import run_uncertainty

df = run_uncertainty(
    "SingleFCM.xls",
    noise_threshold=0.1,
    infer_rule="k",
    function_type="tanh",
    lambda_=1,
    principles=["Sustainability", "Cost"],
    thresh=2,
    n_iteration=100,
)
```

The function saves:

- `Uncertainty_Analysis_Results.pdf`

### `simulation.py`

Reusable lower-level FCM simulation functions.

These are useful if you already have an adjacency matrix and want to run the FCM
logic directly without using the higher-level Excel workflows.

```python
transform_func(x, n, function_type, lambda_=1)
```

Transforms raw concept values using one of the squashing functions.

```python
infer_steady(adj_matrix, n, init_vec=None, function_type="tanh",
             infer_rule="k", lambda_=1, tolerance=0.00001)
```

Runs an FCM until it reaches a steady state.

```python
infer_scenario(scenario_concepts, levels, adj_matrix, n, init_vec=None,
               function_type="tanh", infer_rule="k", lambda_=1,
               tolerance=0.00001, zero_concepts=None)
```

Runs an FCM while forcing selected concepts to fixed activation levels.

Example:

```python
import numpy as np
from pyfcm.simulation import infer_steady, infer_scenario

adj = np.array([
    [0.0, 0.5],
    [-0.2, 0.0],
])

steady = infer_steady(adj, n=2, function_type="tanh")
scenario = infer_scenario(
    scenario_concepts=[0],
    levels={0: 0.8},
    adj_matrix=adj,
    n=2,
    function_type="tanh",
)
```

### `workbooks.py`

Reusable workbook-loading utilities.

```python
read_single_fcm(file_location, noise_threshold=None)
```

Reads the first worksheet from a workbook.

Returns:

```python
adj_matrix, sheet, n_concepts
```

```python
read_participant_fcms(file_location)
```

Reads every worksheet from a multi-participant workbook.

Returns:

```python
participants, participant_ids, matrices, sheet, n_concepts
```

```python
concept_metadata(sheet, n_concepts, principles)
```

Builds concept names, node labels, and principle indexes from a worksheet.

### `__init__.py`

Exports the main public API:

```python
aggregate
visualize_aggregated
cluster
run_scenario
run_sensitivity
run_uncertainty
```

## Common Options

Squashing functions:

- `"sig"`: sigmoid.
- `"tanh"`: hyperbolic tangent.
- `"bivalent"`: positive values become 1, all others become 0.
- `"trivalent"`: positive values become 1, zero stays 0, negative values become
  -1.

Inference rules:

- `"k"`: Kosko.
- `"mk"`: modified Kosko.
- `"r"`: rescaled.

Aggregation methods:

- `"AMI"`: arithmetic mean including zeros.
- `"AMX"`: arithmetic mean excluding zeros.
- `"MED"`: median.
- `"GM"`: geometric mean of nonzero values.

## Output Files

Some workflows save files in the current working directory:

- `visualize_aggregated()`: `aggregated_edg.csv`
- `cluster()`: `Clusters.pdf`
- `run_scenario()`: `Scenario_Results.pdf`,
  `Changes_In_All_Concepts.csv`
- `run_sensitivity()`: one PDF per scenario variable.
- `run_uncertainty()`: `Uncertainty_Analysis_Results.pdf`

## Notes

- Dynamic clustering and uncertainty analysis use random sampling, so results
  may vary between runs.
- `GM` aggregation may not be appropriate for negative edge weights.
- The current Excel dependency is `xlrd>=2.0`, which supports `.xls` files.
