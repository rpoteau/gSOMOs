# 🧪 How to Build the HTML Documentation for gSOMOs (with Sphinx)

This guide explains how to build a full HTML documentation for the gSOMOs package using **Sphinx**.

---

## ✅ 1. Install Required Packages

```bash
pip install sphinx sphinx_rtd_theme myst-parser numpydoc
```

Optional:
```bash
pip install sphinx-autodoc-typehints
```

---

## 🛠️ 2. Initialize the Sphinx project (only needed once)

```bash
sphinx-quickstart docs
```

Answer:
- Separate source and build dirs → yes
- Project name → gSOMOs
- Author name → Romuald Poteau
- Use Makefile → yes

---

## ✏️ 3. Edit `docs/source/conf.py`

Make sure it includes:

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'myst_parser',
    'numpydoc',
]

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
html_extra_path = ['_static']
```

---

## 🛠️ Explanation: `html_static_path` vs `html_extra_path`

- `html_static_path = ['_static']`
  - ➔ Tells Sphinx where to find **static assets** like CSS files, JS files, and local images used for theming or layout.
- `html_extra_path = ['_static']`
  - ➔ Tells Sphinx to **copy raw files (PDFs, datasets, etc.)** into the final built HTML folder, so they are downloadable.

✅ You can safely use both for `_static/` if you want to include custom styles **and** downloadable files like `.pdf`.

---

## 🧱 4. Edit `docs/source/index.rst`

```rst
Welcome to gSOMOs's documentation!
===================================

.. automodule:: somos.io
   :members:
   :undoc-members:

.. automodule:: somos.cosim
   :members:
   :undoc-members:

.. automodule:: somos.proj
   :members:
   :undoc-members:
```

---

## 🚀 5. Build the documentation

From the `docs/` directory:

```bash
make html
```

Open the generated file:

```
docs/build/html/index.html
```

That's your full HTML documentation!

---
