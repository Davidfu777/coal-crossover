# Reproduce the revision checks

Use the pinned environment and the restored 1.0.0-rc1 dataset (see the main README).
These checks do not change the global dataset. Run the commands from the repository root.

1. Obtain the final 2025 [EIA-860](https://www.eia.gov/electricity/data/eia860/xls/eia8602025.zip)
   and [EIA-923](https://www.eia.gov/electricity/data/eia923/xls/f923_2025.zip) workbooks.
   Put the two original ZIPs in one directory and set `COAL_CROSSOVER_EIA_DIR` to its path.
2. Run `python -m scripts.revision_checks.checks`. The script reads the released unit
   results and exports the US site matches, delivered-price comparisons, and global
   single-cost sensitivity counts. Set `COAL_CROSSOVER_CHECK_DIR` to an output folder;
   the default is `revision_checks/` under the model output directory.
3. For the search-radius check, run `python -m scripts.revision_checks.wider_radius`
   with the registered wind and solar rasters, country boundaries, and prepared solar
   raster cache (see `pipeline/paths.py`). It prints aggregate counts.

Source ZIP SHA-256:

```text
2b27929d26cc1da9ad6a530da1cac6718cd8498ee560c4d47cfffd96d08ba  eia8602025.zip
1bff7092a86f6678c069c986829da5a23104048fc1aafb65b0776e3c904284de  f923_2025.zip
```
