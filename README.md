# Coal Crossover reproducible research code

This repository contains the Python pipeline, validation tests, and figure code for *A global dataset for coal-to-renewables cost crossover analysis* by Lilly Loghmani, Haoran Fu, Connor Butterfield, and Gernot Wagner (Mimeo).

The pipeline links operating coal-fired generating units to nearby solar and onshore-wind resources, applies documented cost and financing assumptions, and produces unit-, plant-, sensitivity-, and validation-level records. Original third-party files are not redistributed. Their releases, retrieval dates, checksums, and roles are recorded in `sources/RECEIVED_FILE_MANIFEST.csv` and `metadata/THIRD_PARTY_SOURCES.md`.

## Environment

Use Python 3.12. The reproduction check on 6 September 2026 used Python 3.12.14 and all 32 versions in `requirements-lock.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
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

The current four-hour battery capital-cost input is 164.44 USD/kWh. Its source and assumptions are recorded in `methodology/sensitivity_and_storage_spec_v1.json`.

The first full build extracts the GEM workbook, refreshes local solar and wind matches, and generates all downstream records:

```bash
python -m pipeline.build_dataset --refresh-resources
python -m pipeline.validate_phase2
python -m pipeline.phase3_validation
```

The historical comparison needs two additional processed Parquet files from the earlier study. Set `COAL_CROSSOVER_LEGACY_AWS_PARQUET` and `COAL_CROSSOVER_LEGACY_LILLY_PARQUET` to authorized copies. These are not needed to calculate current central and sensitivity records, but are needed for the historical comparison and its tests. Their checksums and sample counts are recorded in the build metadata.

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

All 51 tests passed in the 6 September 2026 reproduction check. GitHub Actions runs the source-independent subset; the full suite needs the registered inputs and generated outputs.

## Figures

Figures can use either the completed pipeline outputs or the separately supplied data package. To use the data package without repeating raster extraction, restore its tables to a new directory:

```bash
python scripts/restore_outputs.py --data-package /path/to/Data --output /path/to/new/outputs
export COAL_CROSSOVER_OUTPUT_DIR=/path/to/new/outputs
```

The data package is not included in a GitHub clone. Figures 1 and 2 also need the Natural Earth 5.1.1 Admin 0 Countries shapefile and its companion files. These are included under `Figures/Map_Data/` in the submission folder, or can be obtained from Natural Earth using the source manifest.

```bash
export COAL_CROSSOVER_NATURAL_EARTH_SHP=/path/to/ne_10m_admin_0_countries.shp
```

Generate all seven manuscript figures and their source tables from the completed outputs:

```bash
python scripts/build_figures.py
```

Figures are written to `figures/generated/`; source tables are written to `figures/source_data/`. Each figure also receives a JSON provenance record containing input and output hashes.

The manuscript figures use Arial. The scripts fall back to Helvetica or DejaVu Sans if Arial is unavailable; numerical results remain the same, but text placement can differ slightly.

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
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
python scripts/verify_data_release.py /path/to/Data
```

To package a new review snapshot from completed outputs, supply its date explicitly:

```bash
python scripts/build_data_release.py --snapshot-date YYYY-MM-DD --target /path/to/new-data-package
```

The builder refuses to replace an existing package. Method specifications are versioned in `methodology/`. This repository contains research code and documentation, not manuscripts, data archives, credentials, AWS configuration, or the website frontend. Updating the development branch does not publish or replace a DOI archive.
