# Much of the subtable generation code is derived from
# https://github.com/ska-sa/katdal/blob/v0.22/katdal/ms_extra.py
# under the following license
#
# ################################################################################
# Copyright (c) 2011-2023, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version as importlib_version
from operator import getitem
from typing import Dict, List, Literal

import dask.array as da
import numpy as np
import xarray
from katdal.dataset import DataSet
from katdal.lazy_indexer import DaskLazyIndexer
from katpoint import Timestamp

from xarray_katdal.corr_products import corrprod_index
from xarray_katdal.transpose import transpose
from xarray_katdal.uvw import uvw_coords

TAG_TO_INTENT = {
  "gaincal": "CALIBRATE_PHASE,CALIBRATE_AMPLI",
  "bpcal": "CALIBRATE_BANDPASS,CALIBRATE_FLUX",
  "target": "TARGET",
}


# Partitioning columns
GROUP_COLS = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]

# katdal datasets only have one spectral window
# and one polarisation. Thus, there
# is only one DATA_DESC_ID and it is zero
DATA_DESC_ID = 0


def to_mjds(timestamp: Timestamp):
  """Converts a katpoint Timestamp to Modified Julian Date Seconds"""
  return timestamp.to_mjd() * 24 * 60 * 60


DEFAULT_TIME_CHUNKS = 100


class XArrayMSv2Facade:
  """Provides a simplified xarray Dataset view over a katdal dataset"""

  def __init__(
    self,
    dataset: DataSet,
    no_auto: bool = True,
    view_type: Literal["msv2", "msv4"] = "msv4",
    chunks: dict | list[dict] | None = None,
  ):
    self._dataset = dataset
    self._no_auto = no_auto
    self._view_type = view_type
    self._pols_to_use = ["HH", "HV", "VH", "VV"]
    # Reset the dataset selection
    self._dataset.select(reset="")
    self._cp_info = corrprod_index(dataset, self._pols_to_use, not no_auto)

    assert view_type in {"msv2", "msv4"}

    chunks_list: List[Dict[str, int]]

    if chunks is None:
      chunks_list = [{"time": DEFAULT_TIME_CHUNKS}]
    elif isinstance(chunks, dict):
      chunks_list = [chunks]
    elif not isinstance(chunks, list) and not all(isinstance(c, dict) for c in chunks):
      raise TypeError(f"{chunks} must a dictionary or list of dictionaries")
    elif len(chunks) == 0:
      chunks_list.append({"time": DEFAULT_TIME_CHUNKS})

    xformed_chunks = []

    for ds_chunks in chunks_list:
      if self.is_msv4:
        xformed_chunks.append(ds_chunks)
      else:
        # katdal's internal data shape is (time, chan, baseline*pol)
        # If chunking reasoning is row-based it's necessary to
        # derive a time based chunking from the row dimension
        # We cannot always exactly supply the requested number of rows,
        # as we always have to supply a multiple of the number of baselines
        row = ds_chunks.pop("row", DEFAULT_TIME_CHUNKS * self.nbl)
        # We need at least one timestamps worth of rows
        row = max(row, self.nbl)
        time = row // self.nbl
        ds_chunks["time"] = min(time, len(self._dataset.timestamps))
        xformed_chunks.append(ds_chunks)

    self._chunks = xformed_chunks

  @property
  def is_msv2(self):
    return self._view_type == "msv2"

  @property
  def is_msv4(self):
    return self._view_type == "msv4"

  @property
  def cp_info(self):
    return self._cp_info

  @property
  def ntime(self):
    return len(self._dataset.timestamps)

  @property
  def na(self):
    return len(self._dataset.ants)

  @property
  def nbl(self):
    return self._cp_info.cp_index.shape[0]

  @property
  def npol(self):
    return self._cp_info.cp_index.shape[1]

  def _main_xarray_factory(
    self, field_id, state_id, scan_index, state_tag, scan_state, target, chunks
  ):
    # Extract numpy and dask products
    dataset = self._dataset
    cp_info = self._cp_info
    time_utc = dataset.timestamps
    t_chunks, chan_chunks, cp_chunks = dataset.vis.dataset.chunks

    # Override time and channel chunking
    t_chunks = chunks.get("time", t_chunks)
    chan_chunks = chunks.get("chan", chan_chunks)

    # Modified Julian Date in Seconds
    time_mjds = np.asarray([to_mjds(t) for t in map(Timestamp, time_utc)])

    # Create a dask chunking transform
    rechunk = partial(da.rechunk, chunks=(t_chunks, chan_chunks, cp_chunks))

    # Transpose from (time, chan, corrprod) to (time, bl, chan, corr)
    cpi = cp_info.cp_index
    flag_transpose = partial(
      transpose,
      cp_index=cpi,
      data_type="flags",
      row=self.is_msv2,
    )
    weight_transpose = partial(
      transpose,
      cp_index=cpi,
      data_type="weights",
      row=self.is_msv2,
    )
    vis_transpose = partial(
      transpose,
      cp_index=cpi,
      data_type="vis",
      row=self.is_msv2,
    )

    flags = DaskLazyIndexer(dataset.flags, (), (rechunk, flag_transpose))
    weights = DaskLazyIndexer(dataset.weights, (), (rechunk, weight_transpose))
    vis = DaskLazyIndexer(dataset.vis, (), (rechunk, vis_transpose))

    time = da.from_array(time_mjds[:, None], chunks=(t_chunks, 1))
    ant1 = da.from_array(cp_info.ant1_index[None, :], chunks=(1, cpi.shape[0]))
    ant2 = da.from_array(cp_info.ant2_index[None, :], chunks=(1, cpi.shape[0]))

    uvw = uvw_coords(
      target,
      da.from_array(time_utc, chunks=t_chunks),
      dataset.ants,
      cp_info,
      row=self.is_msv2,
    )

    # Better graph than da.broadcast_arrays
    bcast = da.blockwise(
      np.broadcast_arrays,
      ("time", "bl"),
      time,
      ("time", "bl"),
      ant1,
      ("time", "bl"),
      ant2,
      ("time", "bl"),
      align_arrays=False,
      adjust_chunks={"time": time.chunks[0], "bl": ant1.chunks[1]},
      meta=np.empty((0,) * 2, dtype=np.int32),
    )

    btime = da.blockwise(
      getitem, ("time", "bl"), bcast, ("time", "bl"), 0, None, dtype=time.dtype
    )

    bant1 = da.blockwise(
      getitem, ("time", "bl"), bcast, ("time", "bl"), 1, None, dtype=ant1.dtype
    )

    bant2 = da.blockwise(
      getitem, ("time", "bl"), bcast, ("time", "bl"), 2, None, dtype=ant2.dtype
    )

    data_vars = {}
    coords = {}
    attrs = {}

    if self.is_msv2:
      primary_dims = ("row",)
      full_dims = primary_dims + ("chan", "corr")
      btime = btime.ravel()
      bant1 = bant1.ravel()
      bant2 = bant2.ravel()
      data_vars.update(
        {
          # Primary indexing columns
          "TIME": (primary_dims, btime),
          "ANTENNA1": (primary_dims, bant1),
          "ANTENNA2": (primary_dims, bant2),
          "FEED1": (primary_dims, da.zeros_like(bant1)),
          "FEED2": (primary_dims, da.zeros_like(bant1)),
          "DATA_DESC_ID": (primary_dims, da.full_like(bant1, DATA_DESC_ID)),
          "FIELD_ID": (primary_dims, da.full_like(bant1, field_id)),
          "STATE_ID": (primary_dims, da.full_like(bant1, state_id)),
          "ARRAY_ID": (primary_dims, da.zeros_like(bant1)),
          "OBSERVATION_ID": (primary_dims, da.zeros_like(bant1)),
          "PROCESSOR_ID": (primary_dims, da.ones_like(bant1)),
          "SCAN_NUMBER": (primary_dims, da.full_like(bant1, scan_index)),
          "TIME_CENTROID": (primary_dims, btime),
          "INTERVAL": (
            primary_dims,
            da.full_like(btime, dataset.dump_period),
          ),
          "EXPOSURE": (
            primary_dims,
            da.full_like(btime, dataset.dump_period),
          ),
          "UVW": (primary_dims + ("uvw",), uvw),
          "DATA": (full_dims, vis.dataset),
          "FLAG": (full_dims, flags.dataset),
          "WEIGHT_SPECTRUM": (full_dims, weights.dataset),
          # Estimated RMS noise per frequency channel
          # note this column is used when computing calibration weights
          # in CASA - WEIGHT_SPECTRUM may be modified based on the
          # values in this column. See
          # https://casadocs.readthedocs.io/en/stable/notebooks/data_weights.html
          # for further details
          "SIGMA_SPECTRUM": (full_dims, weights.dataset**-0.5),
        }
      )
    else:
      (spw,) = self._dataset.spectral_windows
      primary_dims = ("time", "baseline_id")
      full_dims = primary_dims + ("frequency", "polarization")
      data_vars = {
        "UVW": (primary_dims + ("uvw_label",), uvw),
        "VISIBILITY": (full_dims, vis.dataset),
        "FLAG": (full_dims, flags.dataset),
        "WEIGHT": (full_dims, weights.dataset),
      }

      time_attrs = {
        "type": "quantity",
        "units": "s",
        "integration_time": {
          "data": float(dataset.dump_period),
          "attrs": {
            "type": "quantity",
            "units": "s",
          },
        },
      }

      frequency_attrs = {
        "spectral_window_name": f"{spw.band}-band",
        "channel_width": {
          "data": float(spw.channel_width),
          "attrs": {"type": "quantity", "units": "Hz"},
        },
        "reference_frequency": {
          "data": float(spw.centre_freq),
          "attrs": {
            "type": "spectral_coord",
            "observer": "REST",
            "units": "Hz",
          },
        },
        "units": "Hz",
        "observer": "REST",
        "type": "spectral_coord",
      }

      pol_map = {"HH": "XX", "HV": "XY", "VH": "YX", "VV": "YY"}

      antennas = self._dataset.ants
      ant1_names = [antennas[a].name for a in cp_info.ant1_index]
      ant2_names = [antennas[a].name for a in cp_info.ant2_index]

      coords.update(
        {
          "time": ("time", time_mjds, time_attrs),
          "baseline_id": ("baseline_id", np.arange(self.nbl)),
          "baseline_antenna1_name": ("baseline_id", ant1_names),
          "baseline_antenna2_name": ("baseline_id", ant2_names),
          "frequency": ("frequency", spw.channel_freqs, frequency_attrs),
          "polarization": (
            "polarization",
            [pol_map[p] for p in self._pols_to_use],
          ),
          "uvw_label": ("uvw_label", ["u", "v", "w"]),
        }
      )

      attrs.update(
        {
          "creator": {
            "software": "xarray-katdal",
            "version": importlib_version("xarray-katdal"),
          },
          "creation_date": datetime.now(timezone.utc).isoformat(),
          "processor_info": {"sub_type": "MEERKAT", "type": "CORRELATOR"},
          "type": "visibility",
          "observation_info": {
            "intents": state_tag.split(","),
            "observer": ["observed"],
            "project": "project",
            "release_date": datetime.now(timezone.utc).isoformat(),
          },
        }
      )

    return xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

  def _antenna_xarray_factory(self):
    antennas = self._dataset.ants
    nant = len(antennas)
    return xarray.Dataset(
      {
        "NAME": ("row", np.asarray([a.name for a in antennas], dtype=object)),
        "STATION": (
          "row",
          np.asarray([a.name for a in antennas], dtype=object),
        ),
        "POSITION": (
          ("row", "xyz"),
          np.asarray([a.position_ecef for a in antennas]),
        ),
        "OFFSET": (("row", "xyz"), np.zeros((nant, 3))),
        "DISH_DIAMETER": ("row", np.asarray([a.diameter for a in antennas])),
        "MOUNT": ("row", np.array(["ALT-AZ"] * nant, dtype=object)),
        "TYPE": ("row", np.array(["GROUND-BASED"] * nant, dtype=object)),
        "FLAG_ROW": ("row", np.zeros(nant, dtype=np.int32)),
      }
    )

  def _spw_xarray_factory(self):
    def ref_freq(chan_freqs):
      return chan_freqs[len(chan_freqs) // 2].astype(np.float64)

    return [
      xarray.Dataset(
        {
          "NUM_CHAN": (("row",), np.array([spw.num_chans], dtype=np.int32)),
          "CHAN_FREQ": (("row", "chan"), spw.channel_freqs[np.newaxis, :]),
          "RESOLUTION": (("row", "chan"), spw.channel_freqs[np.newaxis, :]),
          "CHAN_WIDTH": (
            ("row", "chan"),
            np.full_like(spw.channel_freqs[np.newaxis, :], spw.channel_width),
          ),
          "EFFECTIVE_BW": (
            ("row", "chan"),
            np.full_like(spw.channel_freqs[np.newaxis, :], spw.channel_width),
          ),
          "MEAS_FREQ_REF": ("row", np.array([5], dtype=np.int32)),
          "REF_FREQUENCY": ("row", [ref_freq(spw.channel_freqs)]),
          "NAME": ("row", np.asarray([f"{spw.band}-band"], dtype=object)),
          "FREQ_GROUP_NAME": (
            "row",
            np.asarray([f"{spw.band}-band"], dtype=object),
          ),
          "FREQ_GROUP": ("row", np.zeros(1, dtype=np.int32)),
          "IF_CONV_CHAN": ("row", np.zeros(1, dtype=np.int32)),
          "NET_SIDEBAND": ("row", np.ones(1, dtype=np.int32)),
          "TOTAL_BANDWIDTH": ("row", np.asarray([spw.channel_freqs.sum()])),
          "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
        }
      )
      for spw in self._dataset.spectral_windows
    ]

  def _pol_xarray_factory(self):
    pol_num = {"H": 0, "V": 1}
    # MeerKAT only has linear feeds, these map to
    # CASA ["XX", "XY", "YX", "YY"]
    pol_types = {"HH": 9, "HV": 10, "VH": 11, "VV": 12}
    return xarray.Dataset(
      {
        "NUM_CORR": ("row", np.array([len(self._pols_to_use)], dtype=np.int32)),
        "CORR_PRODUCT": (
          ("row", "corr", "corrprod_idx"),
          np.array(
            [[[pol_num[p[0]], pol_num[p[1]]] for p in self._pols_to_use]],
            dtype=np.int32,
          ),
        ),
        "CORR_TYPE": (
          ("row", "corr"),
          np.asarray([[pol_types[p] for p in self._pols_to_use]], dtype=np.int32),
        ),
        "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
      }
    )

  def _ddid_xarray_factory(self):
    return xarray.Dataset(
      {
        "SPECTRAL_WINDOW_ID": ("row", np.zeros(1, dtype=np.int32)),
        "POLARIZATION_ID": ("row", np.zeros(1, dtype=np.int32)),
        "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
      }
    )

  def _feed_xarray_factory(self):
    nfeeds = len(self._dataset.ants)
    NRECEPTORS = 2

    return xarray.Dataset(
      {
        # ID of antenna in this array (integer)
        "ANTENNA_ID": ("row", np.arange(nfeeds, dtype=np.int32)),
        # Id for BEAM model (integer)
        "BEAM_ID": ("row", np.ones(nfeeds, dtype=np.int32)),
        # Beam position offset (on sky but in antenna reference frame): (double, 2-dim)
        "BEAM_OFFSET": (
          ("row", "receptors", "radec"),
          np.zeros((nfeeds, 2, 2), dtype=np.float64),
        ),
        # Feed id (integer)
        "FEED_ID": ("row", np.zeros(nfeeds, dtype=np.int32)),
        # Interval for which this set of parameters is accurate (double)
        "INTERVAL": ("row", np.zeros(nfeeds, dtype=np.float64)),
        # Number of receptors on this feed (probably 1 or 2) (integer)
        "NUM_RECEPTORS": ("row", np.full(nfeeds, NRECEPTORS, dtype=np.int32)),
        # Type of polarisation to which a given RECEPTOR responds (string, 1-dim)
        "POLARIZATION_TYPE": (
          ("row", "receptors"),
          np.array([["X", "Y"]] * nfeeds, dtype=object),
        ),
        # D-matrix i.e. leakage between two receptors (complex, 2-dim)
        "POL_RESPONSE": (
          ("row", "receptors", "receptors-2"),
          np.array([np.eye(2, dtype=np.complex64) for _ in range(nfeeds)]),
        ),
        # Position of feed relative to feed reference position
        # (double, 1-dim, shape=(3,))
        "POSITION": (("row", "xyz"), np.zeros((nfeeds, 3), np.float64)),
        # The reference angle for polarisation (double, 1-dim). A parallactic angle of
        # 0 means that V is aligned to x (celestial North), but we are mapping H to x
        # so we have to correct with a -90 degree rotation.
        "RECEPTOR_ANGLE": (
          ("row", "receptors"),
          np.full((nfeeds, NRECEPTORS), -np.pi / 2, dtype=np.float64),
        ),
        # ID for this spectral window setup (integer)
        "SPECTRAL_WINDOW_ID": ("row", np.full(nfeeds, -1, dtype=np.int32)),
        # Midpoint of time for which this set of parameters is accurate (double)
        "TIME": ("row", np.zeros(nfeeds, dtype=np.float64)),
      }
    )

  def _field_xarray_factory(self, field_data):
    fields = [
      xarray.Dataset(
        {
          "NAME": ("row", np.array([target.name], object)),
          "CODE": ("row", np.array(["T"], object)),
          "SOURCE_ID": ("row", np.array([field_id], dtype=np.int32)),
          "NUM_POLY": ("row", np.zeros(1, dtype=np.int32)),
          "TIME": ("row", np.array([time])),
          "DELAY_DIR": (
            ("row", "field-poly", "field-dir"),
            np.array([[radec]], dtype=np.float64),
          ),
          "PHASE_DIR": (
            ("row", "field-poly", "field-dir"),
            np.array([[radec]], dtype=np.float64),
          ),
          "REFERENCE_DIR": (
            ("row", "field-poly", "field-dir"),
            np.array([[radec]], dtype=np.float64),
          ),
          "FLAG_ROW": ("row", np.zeros(1, dtype=np.int32)),
        }
      )
      for field_id, time, target, radec in field_data.values()
    ]

    return xarray.concat(fields, dim="row")

  def _source_xarray_factory(self, field_data):
    field_ids, times, targets, radecs = zip(*(field_data.values()))
    times = np.array(times, dtype=np.float64)
    nfields = len(field_ids)
    return xarray.Dataset(
      {
        "NAME": ("row", np.array([t.name for t in targets], dtype=object)),
        "SOURCE_ID": ("row", np.array(field_ids, dtype=np.int32)),
        "PROPER_MOTION": (
          ("row", "radec-per-sec"),
          np.zeros((nfields, 2), dtype=np.float32),
        ),
        "CALIBRATION_GROUP": ("row", np.full(nfields, -1, dtype=np.int32)),
        "DIRECTION": (("row", "radec"), np.array(radecs)),
        "TIME": ("row", times),
        "NUM_LINES": ("row", np.ones(nfields, dtype=np.int32)),
        "REST_FREQUENCY": (
          ("row", "lines"),
          np.zeros((nfields, 1), dtype=np.float64),
        ),
      }
    )

  def _state_xarray_factory(self, state_modes):
    state_ids, modes = zip(*sorted((i, m) for m, i in state_modes.items()))
    nstates = len(state_ids)
    return xarray.Dataset(
      {
        "SIG": np.ones(nstates, dtype=np.uint8),
        "REF": np.zeros(nstates, dtype=np.uint8),
        "CAL": np.zeros(nstates, dtype=np.float64),
        "LOAD": np.zeros(nstates, dtype=np.float64),
        "SUB_SCAN": np.zeros(nstates, dtype=np.int32),
        "OBS_MODE": np.array(modes, dtype=object),
        "FLAG_ROW": np.zeros(nstates, dtype=np.int32),
      }
    )

  def _observation_xarray_factory(self):
    ds = self._dataset
    start, end = [to_mjds(t) for t in [ds.start_time, ds.end_time]]
    return xarray.Dataset(
      {
        "OBSERVER": ("row", np.array([ds.observer], dtype=object)),
        "PROJECT": ("row", np.array([ds.experiment_id], dtype=object)),
        "LOG": (("row", "extra"), np.array([["unavailable"]], dtype=object)),
        "SCHEDULE": (
          ("row", "extra"),
          np.array([["unavailable"]], dtype=object),
        ),
        "SCHEDULE_TYPE": ("row", np.array(["unknown"], dtype=object)),
        "TELESCOPE": ("row", np.array(["MeerKAT"], dtype=object)),
        "TIME_RANGE": (("row", "extent"), np.array([[start, end]])),
        "FLAG_ROW": ("row", np.zeros(1, np.uint8)),
      }
    )

  def xarray_datasets(self):
    """Generates partitions of the main MSv2 table, as well as the subtables.

    Returns
    -------
    main_xds: list of :code:`xarray.Dataset`
        A list of xarray datasets corresponding to Measurement Set 2
        partitions
    subtable_xds: dict of :code:`xarray.Dataset`
        A dictionary of datasets keyed on subtable names
    """
    main_xds = []
    field_data = []
    field_data = {}
    UNKNOWN_STATE_ID = 0
    state_modes = {"UNKNOWN": UNKNOWN_STATE_ID}

    # Generate MAIN table xarray partition datasets
    for i, (scan_index, scan_state, target) in enumerate(self._dataset.scans()):
      if scan_state == "slew":
        continue

      # Retrieve existing field data, or create
      try:
        field_id, _, _, _ = field_data[target.name]
      except KeyError:
        field_id = len(field_data)
        time_origin = Timestamp(self._dataset.timestamps[0])
        field_data[target.name] = (
          field_id,
          to_mjds(time_origin),
          target,
          target.radec(time_origin),
        )

      # Create or retrieve the state_id associated
      # with the tags of the current source
      state_tag = ",".join(
        TAG_TO_INTENT[tag] for tag in target.tags if tag in TAG_TO_INTENT
      )
      if state_tag and state_tag not in state_modes:
        state_modes[state_tag] = len(state_modes)
      state_id = state_modes.get(state_tag, UNKNOWN_STATE_ID)

      try:
        chunks = self._chunks[i]
      except IndexError:
        chunks = self._chunks[-1]

      main_xds.append(
        self._main_xarray_factory(
          field_id, state_id, scan_index, state_tag, scan_state, target, chunks
        )
      )

    # Generate subtable xarray datasets
    subtables = {
      "ANTENNA": self._antenna_xarray_factory(),
      "DATA_DESCRIPTION": self._ddid_xarray_factory(),
      "SPECTRAL_WINDOW": self._spw_xarray_factory(),
      "POLARIZATION": self._pol_xarray_factory(),
      "FEED": self._feed_xarray_factory(),
      "FIELD": self._field_xarray_factory(field_data),
      "SOURCE": self._source_xarray_factory(field_data),
      "OBSERVATION": self._observation_xarray_factory(),
      "STATE": self._state_xarray_factory(state_modes),
    }

    return main_xds, subtables
