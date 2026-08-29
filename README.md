# Coal Crossover reproducible research code

This repository contains the Python pipeline, validation tests, and figure code for *A global dataset for coal-to-renewable cost crossover analysis* by Lilly Loghmani, Haoran Fu, Connor Butterfield, and Gernot Wagner (Mimeo).

The pipeline links operating coal-fired generating units to nearby solar and onshore-wind resources, applies documented cost and financing assumptions, and produces unit-, plant-, sensitivity-, and validation-level records. Original third-party files are not redistributed. Their releases, retrieval dates, checksums, and roles are recorded in `sources/RECEIVED_FILE_MANIFEST.csv` and `metadata/THIRD_PARTY_SOURCES.md`.

## Environment

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The same environment can be created with Conda:

```bash
conda env create -f environment.yml
conda activate coal-crossover
```

## Source files

Place authorized source files under `sources/incoming/` using the repository-relative paths in `sources/RECEIVED_FILE_MANIFEST.csv`. The pipeline checks the registered files and records their SHA-256 hashes. Large source files and generated outputs are intentionally excluded from Git.

The source, work, and output roots can be overridden when needed:

```bash
export COAL_CROSSOVER_SOURCE_DIR=/path/to/sources/incoming
export COAL_CROSSOVER_WORK_DIR=/path/to/work
export COAL_CROSSOVER_OUTPUT_DIR=/path/to/outputs
```

## Build the dataset

The first full build extracts the GEM workbook, refreshes local solar and wind matches, and generates all downstream records:

```bash
python -m pipeline.build_dataset --refresh-resources
python -m pipeline.validate_phase2
python -m pipeline.phase3_validation
```

Subsequent runs can reuse the resource-match cache:

```bash
python -m pipeline.build_dataset
```

## Tests

Tests that do not require downloaded source files or generated results can run immediately:

```bash
python -m pytest -m "not source_data and not generated_data"
```

After the source files are registered and the dataset is built, run the full suite:

```bash
python -m pytest
```

## Figures

Generate all six manuscript figures and their source tables from the completed outputs:

```bash
python scripts/build_figures.py
```

Figures are written to `figures/generated/`; source tables are written to `figures/source_data/`. Each figure also receives a JSON provenance record containing input and output hashes.

## Browser data exports

Create the compact records used by the interactive explorer after completing the dataset build:

```bash
python -m pipeline.build_website_data
```

The files are written to `outputs/website_export/` and remain outside version control. Set `COAL_CROSSOVER_WEBSITE_EXPORT_DIR` to write them elsewhere.

The browser export includes a deterministic gzip copy of the scenario inputs and one
resource-matrix file per search radius. The explorer can therefore load the central
scenario first and retrieve other radii only when requested, without changing the
underlying values.

## Reporting samples

The release distinguishes two operating-fleet samples. `grid_power_comparable` contains 2,836 units at 1,056 plants and excludes units identified as captive or combined heat and power. `expanded_operating_fleet` contains all 6,580 operating units at 2,384 plants.

The interactive explorer opens with the expanded sample so that users can inspect the full operating fleet. Unless stated otherwise, manuscript summary statistics and prespecified validation results use `grid_power_comparable`.

## Release checks

```bash
python scripts/check_repository.py
python scripts/build_data_release.py --help
python scripts/verify_data_release.py --help
```

Method specifications are versioned in `methodology/`. The development repository may continue to change, while each tagged release and archived DOI identifies a fixed version of the code.
