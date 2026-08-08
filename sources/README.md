# Source-file placement

Original source files are not stored in this repository. Obtain them from the providers listed in `../metadata/THIRD_PARTY_SOURCES.md`, retain them under their original terms, and place authorized copies in `sources/incoming/` using the relative paths recorded in `RECEIVED_FILE_MANIFEST.csv`.

Before a full build, verify each file against the registered SHA-256 value. A different source release should be added as a new manifest record rather than silently replacing the registered file.
