# 📜 Changelog

All notable changes to this gSOMOs project will be documented here.

## [0.2.1] - 2024-04-24
### Fixed
- added again `*.pdf` in `.gitignore`
- added `doc-latex/*.pdf` in `.gitignore`
- added MANIFEST.in

## [0.2.0] - 2024-04-24
### Fixed
- removed `*.pdf` in `.gitignore`

## [0.1.9] - 2024-04-24
### Added
- Added `CHANGELOG.md`

---

## [0.1.8] - 2024-04-24
### Added
- Official release of `gSOMOs` on [PyPI](https://pypi.org/project/gSOMOs/)
- Link toward [PyPI](https://pypi.org/project/gSOMOs/) in `README.md`

### Improved
- Finalized project name after conflict with `somos`
- Confirmed compatibility of `.log.gz` support with Gaussian files
- PyPI upload workflow and authentication instructions

### Documentation
- Added detailed publishing guide
- Added SVG banner handling in Sphinx documentation
- Structured examples and notebook summary

---

## [0.1.5] - 2024-04-24
### Added
- Automatic support for `.log.gz` compressed Gaussian files
- Added a PyPI badge to the GitHub `README.md`
- Generated Sphinx documentation scaffold with custom theming
- Markdown summaries of features and usage

### Fixed
- `<S**2>` spin parsing now runs before deleting temp files
- Improved PyPI packaging logic for `twine upload`
- Clarified error messages during `pip install -e .` behavior

---

## [0.1.4] - 2024-04-23
### Changed
- Project renamed from `SOMOs` to `gSOMOs` to avoid conflict on PyPI
- Updated all references in `README.md` and `pyproject.toml`
- Switched badge links to lowercase for compatibility with Shields.io

---

## [0.1.3] - 2024-04-22
### Added
- Initial modularization of the project into `io`, `cosim`, and `proj`
- Split functionality cleanly between loading, similarity, and projection
- Editable install and internal testing

---

## [0.1.0] - 2024-04-19
### Added
- First public version of the SOMOs utility
- Support for UDFT/DFT Gaussian logs
- Cosine similarity detection of SOMOs
- Projection-based analysis of alpha → beta orbitals
- Interactive heatmaps and 2D t-SNE projection
