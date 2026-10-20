# Data directory (Docker volume mount point)

Inside Docker this directory is the named volume **`heart_dataset_data`**
mounted at `/app/data`. The trainer downloads the official UCI dataset here
automatically:

- Source page: https://archive.ics.uci.edu/dataset/45/heart+disease
- Direct archive: https://archive.ics.uci.edu/static/public/45/heart+disease.zip
- Stored file: `processed.cleveland.data` (Cleveland subset, 303 records)
- Version identifier: SHA-256 of the file, recorded in every model's metadata

## Manual installation (only needed when the container has no internet access)

1. Download `heart+disease.zip` on any machine with internet access.
2. Extract the member `processed.cleveland.data`.
3. Copy it into the volume:

   ```bash
   docker run --rm -v heart_dataset_data:/data -v "$PWD":/host \
       alpine cp /host/processed.cleveland.data /data/processed.cleveland.data
   ```

4. Re-run the trainer: `docker compose run --rm trainer`

No dataset files are committed to git.
