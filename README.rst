xarray views over the MeerKAT archive
=====================================


.. code-block:: bash

  $ pip install xarray-katdal zarr

.. code-block:: python

  import xarray_katdal
  import xarray

  cbid = 1234567890
  token = "eYabcdefgh"
  url = f"https://archive-gw-1.kat.ac.za/{cb_id}/{cb_id}_sdp_l0.full.rdb?token={token}"
  dt = xarray.open_datatree(url)
  dt.to_zarr("/path/to/dataset.zarr", compute=True)
  dt2 = xarray.open_datatree("/path/to/dataset.zarr")
  assert dt.identical(dt2)
