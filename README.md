<div style="text-align:center">
<img src="./somos/config/svg/pyPCBanner.svg" alt="SOMOs" width="1000"/>
</div>

> **Version [0.9.1] & [0.9.2] - 2024-04-26**
>
> **Changed**
>
> - gSOMOs-v3.pdf scientific document now downloadable in `gsomos.readthedocs.io`
>
> **Added**
>
> - Short examples in the documentation
> - docstring for `projection_heatmap_from_df`
> - docstring for `show_alpha_to_homo` translated in English
>
> **Version [0.9.0] - 2024-04-26**
> 
> **Changed**
> 
> - logo is now gSOMOs instead of SOMOs
> - in the projection scheme (`proj.py`), there are now two criteria to identify a SOMO, namely "SOMO P2v?" (formerly SOMO?) and "SOMO dom. β MO?" (see scientific documentation)
>     - SOMOs identified according to the P^2_virtual criterion are highlighted in green
>     - SOMOs identified only on the basis of a dominant virtual beta MO are highlighted in orange (weaker criterion)
> - Scientific documentation renamed `gSOMOs.pdf`. And content updated
> 
> **Added**
> 
> - new analyzis scheme in `proj.py`: bases on the diagonalization of projection matrices
> - new `clean_logfile_name()` function in `io.py` (made to solve a prefix issue for X.log.gz files)

# SOMOs

🔗 Available on [PyPI](https://pypi.org/project/gSOMOs/)

A Python library to identify and analyze Single Occupied Molecular Orbitals (SOMOs) from Gaussian 09 or Gaussian 16 `.log` files.

[![PyPI version](https://img.shields.io/pypi/v/gSOMOs.svg?color=blue)](https://pypi.org/project/gSOMOs/)
[![Documentation Status](https://readthedocs.org/projects/gsomos/badge/?version=latest)](https://gsomos.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/rpoteau/gSOMOs)](https://github.com/rpoteau/gSOMOs/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://www.python.org/downloads/)
[![Build](https://img.shields.io/badge/build-manual-lightgrey)](https://github.com/rpoteau/gSOMOs)


<hr style="height:3px; background-color:#00aaaa; border:none;" />

## Installation

```bash
pip install SOMOs
```

<hr style="height:3px; background-color:#00aaaa; border:none;" />

## Capabilities Overview

`SOMOs` is a Python toolkit for analyzing **molecular orbitals** (MOs) from Gaussian log files, with a focus on identifying **SOMOs** (Singly Occupied Molecular Orbitals) in open-shell systems.

---

### 🚀 Main Features

#### Load Gaussian Log Files
- Parses `.log` and `.log.gz` Gaussian output files
- Extracts orbital energies, coefficients, overlap matrices, and spin
```python
from somos import io
alpha_df, beta_df, alpha_mat, beta_mat, nbasis, S, info = io.load_mos_from_cclib(logfolder, logfile)
```

#### Cosine Similarity & SOMO Detection
- Computes cosine similarities between alpha and beta orbitals
- Identifies SOMO candidates from orbital projections
```python
from somos import cosim
listMOs, coeffMOs, nBasis, dfSOMOs, S = cosim.analyzeSimilarity(logfolder, logfile)
```


#### Projection-Based Analysis
- Projects occupied and virtual alpha MOs onto virtual beta MOs
- Decomposes projection matrix to extract leading contributions
```python
from somos import proj
df_proj, info = proj.project_occupied_alpha_onto_beta(logfolder, logfile)
display(proj.show_alpha_to_homo(df_proj, logfolder, logfile))
```

---

### 📊 Visualization Tools

#### Heatmaps
- Interactive or static heatmaps of MO similarities
- Highlights SOMO-related regions and orbital clustering

#### t-SNE (Dimensionality Reduction)
- Projects high-dimensional orbital space to 2D for visual exploration
- Enables inspection of orbital families and similarity patterns

```python
cosim.heatmap_MOs(listMOs, coeffMOs, nBasis, S, logfolder, logfile)          # Generates heatmap from cosine similarities
cosim.tsne(listMOs, coeffMOs, S, logfolder,logfile)                          # Generates 2D layout from cosine similarities
proj.projection_heatmap_from_df(df_proj, info["nbasis"], logfolder, logfile) # Generates heatmap from the projection scheme
```

---

### 📁 Output
- `.xlsx` tables of SOMO similarity and projections
- `.png` images of heatmaps and projections
- All results saved in the `logs/` folder

---

### ✅ Example Used in Notebook
- `H2CO_T1_g09_wOverlaps.log.gz` (compressed Gaussian file)

<hr style="height:3px; background-color:#00aaaa; border:none;" />

## Examples

see the `SOMOs-examples.ipynb` Jupyter notebook

<hr style="height:3px; background-color:#00aaaa; border:none;" />

## Technical and scsientific documentation

This document describes two complementary methods to identify singly occupied molecular orbitals (SOMOs) in open-shell systems:
- **Orbital projection analysis**, where occupied α orbitals are projected onto the β orbital basis using the AO overlap matrix;
- **Cosine similarity mapping**, which computes the angular similarity between α and β orbitals and matches them using the Kuhn–Munkres (Hungarian) algorithm.

An example based on the triplet state (T₁) of formaldehyde (H₂CO) is included in the doc/ folder

<hr style="height:3px; background-color:#00aaaa; border:none;" />

