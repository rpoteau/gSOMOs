# 🚀 Publishing gSOMOs to PyPI

This guide explains how to properly publish the gSOMOs package to [PyPI](https://pypi.org/).

---

## ✅ 1. Prepare the Environment

Install required tools:

```bash
pip install build twine
```

---

## 🛠️ 2. Check and Update Metadata

Ensure `pyproject.toml` includes:

- Correct `name`, `version`, `description`
- Proper `authors` and `maintainers`
- Accurate `[project.urls]` (especially the "Documentation" link)

Update the version number before each release.

---

## 📦 3. Build the Distribution

```bash
rm -rf dist/ build/ *.egg-info
python -m build
```

This will generate:

- `dist/gSOMOs-x.y.z.tar.gz`
- `dist/gSOMOs-x.y.z-py3-none-any.whl`

---

## 📤 4. Upload to TestPyPI (optional)

To verify everything is correct:

```bash
twine upload --repository testpypi dist/*
```

---

## 🚀 5. Upload to PyPI

When ready for a real release:

```bash
twine upload dist/*
```

You will need your PyPI API token.

---

## 📋 6. Check the Uploaded Package

After upload:

- Check the PyPI project page
- Check the README rendering
- Check the "Project links" (documentation, repository, changelog)

---

# 📄 Managing `.gitignore` and `MANIFEST.in`

When preparing your package for PyPI, you must control which files are:

- Included inside your distribution archive (`.tar.gz`, `.whl`)
- Ignored from Git versioning

---

## 🧹 `.gitignore`

The `.gitignore` file tells Git **what NOT to track** in your repository.

Typical entries:

```bash
# Ignore temporary files
__pycache__/
*.pyc
*.pyo
*.swp
*.swo
*.~ 

# Ignore builds
build/
dist/
*.egg-info/

# Ignore documentation build
docs/_build/

# Ignore datasets, logs, notebooks cache
*.log
.ipynb_checkpoints/
```

✅ It keeps your GitHub repo **clean and lightweight**.

---

## 📦 `MANIFEST.in`

The `MANIFEST.in` file tells setuptools **what TO INCLUDE** in the distribution archive.

Example:

```bash
recursive-include somos *.py
recursive-include docs/source/_static *
include README.md
include LICENSE
include pyproject.toml
include docs/source/_static/*.pdf
```

✅ It ensures that:
- Your code
- Your documentation files (PDFs, banners, etc.)
- Your `README.md`, `LICENSE`, and `pyproject.toml`
  
are all packed into your `.tar.gz` or `.whl` files.

---

## 🧠 Important

- **Git and PyPI are independent**:  
  What you push to GitHub (controlled by `.gitignore`) is not exactly what you upload to PyPI (controlled by `MANIFEST.in`).
  
- **You can ignore a file in Git but still include it in your PyPI package** (and vice-versa).

- **Missing MANIFEST entries** will cause your PyPI package to lack essential files (like images, PDFs, etc.)

---

## ✅ Best practices for gSOMOs

- Have a clean `.gitignore` excluding build artifacts
- Explicitly include your important docs/assets in `MANIFEST.in`
- Always rebuild (`python -m build`) and check the content of `dist/*.tar.gz` before uploading

---
