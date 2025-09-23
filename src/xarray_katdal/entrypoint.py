from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable
from urllib.parse import urlsplit

import katdal
from xarray import DataTree
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore

from xarray_katdal.msv2_facade import XArrayMSv2Facade

if TYPE_CHECKING:
  from io import BufferedIOBase


class KatdalStore(AbstractDataStore):
  """Store for reading from a katdal data source"""


class KatdalEntryPoint(BackendEntrypoint):
  open_dataset_parameters = ["filename_or_obj"]

  description = "Opens a katdal data source"

  def guess_can_open(
    self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore
  ) -> bool:
    if not isinstance(filename_or_obj, (str, os.PathLike)):
      return False

    urlbits = urlsplit(str(filename_or_obj))

    return (
      "kat.ac.za" in urlbits.netloc
      and urlbits.path.endswith("rdb")
      and (
        urlbits.scheme == "http"
        or (urlbits.scheme == "https" and "token" in urlbits.query)
      )
    )

  def open_datatree(self, filename_or_obj, *, drop_variables=None):
    group_dicts = self.open_groups_as_dict(
      filename_or_obj, drop_variables=drop_variables
    )
    return DataTree.from_dict(group_dicts)

  def open_groups_as_dict(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
  ):
    katds = katdal.open(filename_or_obj)
    facade = XArrayMSv2Facade(katds, no_auto=True, view_type="msv4")
    datasets, _ = facade.xarray_datasets()

    groups = {}

    for i, ds in enumerate(datasets):
      # TODO(sjperkins): Replace with capture block id
      groups[f"katdal-{i}"] = ds

    return groups
