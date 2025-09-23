from __future__ import annotations

import os
import urllib


def default_output_name(url):
  url_parts = urllib.parse.urlparse(url, scheme="file")
  # Create zarr dataset in current working directory (strip off directories)
  dataset_filename = os.path.basename(url_parts.path)
  # Get rid of the ".full" bit on RDB files (it's the same dataset)
  full_rdb_ext = ".full.rdb"
  if dataset_filename.endswith(full_rdb_ext):
    dataset_basename = dataset_filename[: -len(full_rdb_ext)]
  else:
    dataset_basename = os.path.splitext(dataset_filename)[0]
  return f"{dataset_basename}.zarr"
