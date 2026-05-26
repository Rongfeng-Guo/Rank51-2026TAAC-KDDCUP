"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

# NOTE: PERIODIC_TIME_NS - sample-level timestamps are converted into small
# cyclical categorical IDs and appended as synthetic user_int features. These
# reserved fids intentionally live outside schema.json so the raw parquet schema
# stays unchanged while train.py/infer.py can still group them into User NS.
PERIODIC_TIME_NS_FIDS = (1_200_001, 1_200_002, 1_200_003, 1_200_004)
PERIODIC_TIME_NS_VOCAB_SIZES = (24, 7, 2, 4)
PERIODIC_TIME_NS_LOCAL_OFFSET_SECONDS = 8 * 60 * 60

# NOTE: SEQ_PERIODIC_HOUR_DAY_SIDEINFO - optional sequence event local hour and
# weekday exposed as ordinary synthetic categorical side-info slots.
SEQ_PERIODIC_HOUR_FID = 1_400_001
SEQ_PERIODIC_DAY_FID = 1_400_002
SEQ_PERIODIC_HOUR_VOCAB_SIZE = 24
SEQ_PERIODIC_DAY_VOCAB_SIZE = 7

# NOTE: ITEM_ID_FEATURE - raw parquet item_id can be appended as a synthetic
# hashed item-side categorical feature at runtime, without modifying
# schema.json.
ITEM_ID_FEATURE_FID = 1_300_001
ITEM_ID_HASH_BUCKETS_DEFAULT = 50_000


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        timestamp_min: Optional[int] = None,
        timestamp_max: Optional[int] = None,
        num_rows_override: Optional[int] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        use_periodic_time_ns: bool = False,
        use_seq_periodic_hour_day_sideinfo: bool = False,
        use_item_id_feature: bool = False,
        item_id_hash_buckets: int = ITEM_ID_HASH_BUCKETS_DEFAULT,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            timestamp_min/timestamp_max: optional row-level timestamp filter.
                Used to delete the oldest rows by primary timestamp while preserving
                the original batch conversion/model interface.
            num_rows_override: estimated rows after timestamp filtering; only used
                for DataLoader length/progress estimates.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            use_periodic_time_ns: if True, append synthetic categorical
                hour/day/weekend/time-of-day fields derived from timestamp to
                user_int_feats for User NS consumption.
            use_seq_periodic_hour_day_sideinfo: if True, append per-event
                hour-of-day and day-of-week as synthetic sequence side-info
                features.
            use_item_id_feature: if True, append a synthetic hashed item_id
                scalar slot to item_int_feats for Item NS consumption.
            item_id_hash_buckets: number of positive hash buckets used by the
                synthetic item_id feature when enabled.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.timestamp_min = timestamp_min
        self.timestamp_max = timestamp_max
        self._num_rows_override = num_rows_override
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        # NOTE: PERIODIC_TIME_NS - default off preserves historical schema
        # dimensions; enabling appends four timestamp-derived categorical slots
        # to user_int_feats without reading any extra parquet columns.
        self.use_periodic_time_ns = use_periodic_time_ns
        self.use_seq_periodic_hour_day_sideinfo = bool(
            use_seq_periodic_hour_day_sideinfo)
        # NOTE: ITEM_ID_FEATURE - default off preserves historical item_int
        # dimensions; enabling appends one hashed raw-item_id slot to the item
        # side without changing schema.json.
        self.use_item_id_feature = use_item_id_feature
        self.item_id_hash_buckets = int(item_id_hash_buckets)
        if self.use_item_id_feature and self.item_id_hash_buckets <= 0:
            raise ValueError('item_id_hash_buckets must be > 0 when item_id feature is enabled')
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        raw_num_rows = sum(r[2] for r in self._rg_list)
        self.num_rows = int(num_rows_override) if num_rows_override is not None else raw_num_rows

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        # NOTE: PERIODIC_TIME_NS - the synthetic time features are appended to
        # user_int_schema during _load_schema, so the slice can be filled
        # directly after timestamp is read for each batch.
        if self.use_periodic_time_ns:
            self._periodic_time_ns_offset, _ = self.user_int_schema.get_offset_length(
                PERIODIC_TIME_NS_FIDS[0])
        else:
            self._periodic_time_ns_offset = None

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        # NOTE: ITEM_ID_FEATURE - the synthetic hashed item_id slot is
        # appended to item_int_schema during _load_schema, then filled from the
        # raw parquet item_id column at batch-conversion time.
        if self.use_item_id_feature:
            self._item_id_feature_col_idx = self._col_idx.get('item_id')
            if self._item_id_feature_col_idx is None:
                raise KeyError(
                    'item_id column is missing from parquet schema; '
                    '--use_item_id_feature requires raw parquet item_id')
            self._item_id_feature_offset, _ = self.item_int_schema.get_offset_length(
                ITEM_ID_FEATURE_FID)
            self._item_id_feature_vocab_size = self.item_id_hash_buckets + 1
        else:
            self._item_id_feature_col_idx = None
            self._item_id_feature_offset = None
            self._item_id_feature_vocab_size = 0

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                if fid in {SEQ_PERIODIC_HOUR_FID, SEQ_PERIODIC_DAY_FID}:
                    ci = None
                else:
                    ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs, fid))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}, "
            f"timestamp_min={timestamp_min}, timestamp_max={timestamp_max}")
        if self.use_periodic_time_ns:
            logging.info(
                "Periodic time NS enabled: appended synthetic user_int fids=%s",
                PERIODIC_TIME_NS_FIDS,
            )
        if self.use_seq_periodic_hour_day_sideinfo:
            logging.info(
                "Sequence hour/day side-info enabled: hour_fid=%s, day_fid=%s",
                SEQ_PERIODIC_HOUR_FID,
                SEQ_PERIODIC_DAY_FID,
            )
            for domain in self.seq_domains:
                logging.info(
                    "Sequence domain %s sideinfo_fids=%s",
                    domain,
                    self.sideinfo_fids[domain],
                )
        if self.use_item_id_feature:
            logging.info(
                "Item id feature enabled: appended synthetic item_int fid=%s, hash_buckets=%s",
                ITEM_ID_FEATURE_FID,
                self.item_id_hash_buckets,
            )

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # NOTE: PERIODIC_TIME_NS - append four timestamp-derived categorical
        # slots after all real user_int features. Values emitted at runtime
        # are 1-based so 0 remains padding.
        if self.use_periodic_time_ns:
            for fid, vocab_size in zip(PERIODIC_TIME_NS_FIDS, PERIODIC_TIME_NS_VOCAB_SIZES):
                self.user_int_schema.add(fid, 1)
                self.user_int_vocab_sizes.append(vocab_size)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # NOTE: ITEM_ID_FEATURE - append one hashed raw-item_id categorical
        # slot after all real item_int features. The hash emits positive ids
        # so 0 remains padding.
        if self.use_item_id_feature:
            self.item_int_schema.add(ITEM_ID_FEATURE_FID, 1)
            self.item_int_vocab_sizes.append(self.item_id_hash_buckets + 1)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            domain_vocab_sizes = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]
            if self.use_seq_periodic_hour_day_sideinfo:
                self.seq_vocab_sizes[domain][SEQ_PERIODIC_HOUR_FID] = (
                    SEQ_PERIODIC_HOUR_VOCAB_SIZE
                )
                self.seq_vocab_sizes[domain][SEQ_PERIODIC_DAY_FID] = (
                    SEQ_PERIODIC_DAY_VOCAB_SIZE
                )
                sideinfo.append(SEQ_PERIODIC_HOUR_FID)
                sideinfo.append(SEQ_PERIODIC_DAY_FID)
                domain_vocab_sizes.append(SEQ_PERIODIC_HOUR_VOCAB_SIZE)
                domain_vocab_sizes.append(SEQ_PERIODIC_DAY_VOCAB_SIZE)
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = domain_vocab_sizes

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                batch_dict = self._filter_batch_by_timestamp(batch_dict)
                if batch_dict is None:
                    continue
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _filter_batch_by_timestamp(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter a converted batch by primary sample timestamp.

        This keeps the dataset/model interface unchanged. Metadata keys such as
        ``_seq_domains`` are batch-level and must not be row-filtered even when
        their list length happens to equal the batch size.
        """
        if self.timestamp_min is None and self.timestamp_max is None:
            return batch
        ts = batch['timestamp']
        mask = torch.ones(ts.shape[0], dtype=torch.bool)
        if self.timestamp_min is not None:
            mask &= ts >= int(self.timestamp_min)
        if self.timestamp_max is not None:
            mask &= ts <= int(self.timestamp_max)
        if bool(mask.all()):
            return batch
        keep = int(mask.sum().item())
        if keep == 0:
            return None
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        out: Dict[str, Any] = {}
        B = ts.shape[0]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.shape[:1] == (B,):
                out[k] = v.index_select(0, idx)
            elif k == 'user_id' and isinstance(v, list) and len(v) == B:
                out[k] = [v[int(i)] for i in idx.tolist()]
            else:
                out[k] = v
        return out

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _build_periodic_time_ns_feats(
        self,
        timestamps: "npt.NDArray[np.int64]",
    ) -> "npt.NDArray[np.int64]":
        """Build 1-based periodic timestamp categorical IDs for User NS."""
        # NOTE: PERIODIC_TIME_NS - timestamp is Unix seconds; add the fixed
        # Asia/Shanghai offset before deriving local hour and weekday.
        local_seconds = timestamps + PERIODIC_TIME_NS_LOCAL_OFFSET_SECONDS
        local_days = np.floor_divide(local_seconds, 86400)
        seconds_of_day = np.mod(local_seconds, 86400)

        hour_of_day = np.floor_divide(seconds_of_day, 3600)
        day_of_week = np.mod(local_days + 3, 7)
        is_weekend = (day_of_week >= 5).astype(np.int64)
        time_of_day_bucket = np.floor_divide(hour_of_day, 6)

        return np.stack([
            hour_of_day + 1,
            day_of_week + 1,
            is_weekend + 1,
            time_of_day_bucket + 1,
        ], axis=1).astype(np.int64, copy=False)

    def _build_seq_hour_day_feats(
        self,
        timestamps: "npt.NDArray[np.int64]",
    ) -> "Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]":
        """Build per-event local hour and weekday IDs with shape [B, L]."""
        timestamps = np.asarray(timestamps, dtype=np.int64)
        valid = timestamps > 0
        local_seconds = timestamps + PERIODIC_TIME_NS_LOCAL_OFFSET_SECONDS
        local_days = np.floor_divide(local_seconds, 86400)
        seconds_of_day = np.mod(local_seconds, 86400)

        hour0 = np.floor_divide(seconds_of_day, 3600).astype(np.int64, copy=False)
        day0 = np.mod(local_days + 3, 7).astype(np.int64, copy=False)

        hour_ids = (hour0 + 1).astype(np.int64, copy=False)
        day_ids = (day0 + 1).astype(np.int64, copy=False)
        hour_ids[~valid] = 0
        day_ids[~valid] = 0
        return hour_ids, day_ids

    def _hash_item_id_column(
        self,
        arrow_col: "pa.Array",
    ) -> "npt.NDArray[np.int64]":
        """Hash raw item_id values into positive synthetic categorical ids."""
        null_mask = arrow_col.is_null().to_numpy(zero_copy_only=False).astype(bool, copy=False)
        arr = arrow_col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        valid_mask = (~null_mask) & (arr > 0)
        if valid_mask.any():
            # NOTE: ITEM_ID_FEATURE - use 1-based hash buckets so the result
            # can flow through the ordinary categorical encoder with 0 kept as
            # padding and <=0 still treated as missing/unknown.
            arr[valid_mask] = (
                np.remainder(arr[valid_mask] - 1, self.item_id_hash_buckets) + 1)
        arr[~valid_mask] = 0
        return arr

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        if self.use_periodic_time_ns:
            # NOTE: PERIODIC_TIME_NS - append cyclic sample-time categories to
            # the user side so NS tokenizers see them as ordinary categorical
            # embeddings, never as raw continuous timestamps.
            offset = self._periodic_time_ns_offset
            user_int[:, offset:offset + len(PERIODIC_TIME_NS_FIDS)] = (
                self._build_periodic_time_ns_feats(timestamps))

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        if self.use_item_id_feature:
            # NOTE: ITEM_ID_FEATURE - materialize raw parquet item_id as a
            # hashed synthetic item_int scalar so the model reuses the existing
            # Item NS tokenizer path without any dedicated model input field.
            item_id_arr = self._hash_item_id_column(
                batch.column(self._item_id_feature_col_idx))
            self._record_oob(
                'item_int',
                self._item_id_feature_col_idx,
                item_id_arr,
                self._item_id_feature_vocab_size,
            )
            item_int[:, self._item_id_feature_offset] = item_id_arr

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            seq_periodic_hour_slots = []
            seq_periodic_day_slots = []
            for ci, slot, vs, fid in side_plan:
                if fid == SEQ_PERIODIC_HOUR_FID:
                    seq_periodic_hour_slots.append(slot)
                    continue
                if fid == SEQ_PERIODIC_DAY_FID:
                    seq_periodic_day_slots.append(slot)
                    continue
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci, slot))

            for offs, vals, _vs, _ci, slot in col_data:
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, slot, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for _offs, _vals, vs, ci, slot in col_data:
                slice_c = out[:, slot, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets
                if seq_periodic_hour_slots or seq_periodic_day_slots:
                    hour_ids, day_ids = self._build_seq_hour_day_feats(ts_padded)
                    seq_valid_mask = (
                        np.arange(max_len, dtype=np.int64)[None, :] < lengths[:, None]
                    )
                    valid_mask_int = seq_valid_mask.astype(np.int64, copy=False)
                    hour_ids *= valid_mask_int
                    day_ids *= valid_mask_int
                    for slot in seq_periodic_hour_slots:
                        out[:, slot, :] = hour_ids
                    for slot in seq_periodic_day_slots:
                        out[:, slot, :] = day_ids

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        return result



def _collect_primary_timestamps(
    rg_info: List[Tuple[str, int, int]],
    max_sample_size: int = 0,
) -> np.ndarray:
    """Collect/sample primary timestamps from row groups for recent-window split.

    max_sample_size <= 0 means exact scan. For the competition scale (~1M rows)
    exact scan of one int64 column is cheap and avoids quantile jitter.
    """
    chunks: List[np.ndarray] = []
    for file_path, rg_idx, _ in rg_info:
        pf = pq.ParquetFile(file_path)
        if 'timestamp' not in pf.schema_arrow.names:
            raise KeyError("timestamp column not found in parquet data")
        arr = pf.read_row_group(rg_idx, columns=['timestamp']).column('timestamp')
        chunks.append(arr.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64))
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    ts = np.concatenate(chunks)
    if max_sample_size and max_sample_size > 0 and ts.size > max_sample_size:
        rng = np.random.default_rng(2026)
        idx = rng.choice(ts.size, size=max_sample_size, replace=False)
        ts = ts[idx]
    return ts


def _count_rows_ge_timestamp(
    rg_info: List[Tuple[str, int, int]],
    threshold: int,
) -> int:
    total = 0
    for file_path, rg_idx, _ in rg_info:
        pf = pq.ParquetFile(file_path)
        arr = pf.read_row_group(rg_idx, columns=['timestamp']).column('timestamp')
        ts = arr.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        total += int((ts >= threshold).sum())
    return total

def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    split_mode: str = 'row_group_tail',
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    train_recent_ratio: float = 1.0,
    split_sample_size: int = 0,
    use_periodic_time_ns: bool = False,
    use_seq_periodic_hour_day_sideinfo: bool = False,
    use_item_id_feature: bool = False,
    item_id_hash_buckets: int = ITEM_ID_HASH_BUCKETS_DEFAULT,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``). Optionally, training rows
    are further filtered to keep only the latest ``train_recent_ratio`` by the
    primary timestamp. Validation remains the original tail split.

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    split_mode = 'row_group_tail' if split_mode == 'row_group' else split_mode
    if split_mode not in ('row_group_tail', 'overlap_tail'):
        raise ValueError(f"unknown split_mode={split_mode!r}")

    if split_mode == 'overlap_tail':
        # True overlap-tail mode for final-fit style training:
        #   train = all row groups, row-level filtered to latest train_recent_ratio by primary timestamp
        #   valid = all row groups, row-level filtered to latest valid_ratio by primary timestamp
        # so the validation monitor is included in the training stream.  This is intentionally
        # a monitor, not a disjoint validation set.
        if not (0.0 < train_recent_ratio <= 1.0):
            raise ValueError(f"train_recent_ratio must be in (0,1], got {train_recent_ratio}")
        if not (0.0 < valid_ratio <= 1.0):
            raise ValueError(f"valid_ratio must be in (0,1], got {valid_ratio}")

        ts_sample = _collect_primary_timestamps(rg_info, max_sample_size=split_sample_size)
        if ts_sample.size == 0:
            raise ValueError('no timestamps available for overlap_tail split')

        def _lower_quantile_threshold(keep_ratio: float) -> int:
            q = max(0.0, min(1.0, 1.0 - float(keep_ratio)))
            kth = int(np.floor(q * (ts_sample.size - 1)))
            return int(np.partition(ts_sample, kth)[kth])

        train_timestamp_min = _lower_quantile_threshold(train_recent_ratio)
        valid_timestamp_min = _lower_quantile_threshold(valid_ratio)
        raw_rows = sum(r[2] for r in rg_info)
        train_rows_after_recent = _count_rows_ge_timestamp(rg_info, train_timestamp_min)
        valid_rows_after_recent = _count_rows_ge_timestamp(rg_info, valid_timestamp_min)

        logging.info(
            f"Overlap-tail timestamp split: raw_rows={raw_rows}; "
            f"train keeps latest {train_recent_ratio:.4f} (delete_oldest={1.0-train_recent_ratio:.4f}), "
            f"train_timestamp_min={train_timestamp_min}, train_rows={train_rows_after_recent}; "
            f"valid monitor keeps latest {valid_ratio:.4f}, valid_timestamp_min={valid_timestamp_min}, "
            f"valid_rows={valid_rows_after_recent}; valid is included in train")

        train_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=shuffle_train,
            buffer_batches=buffer_batches,
            row_group_range=None,
            timestamp_min=train_timestamp_min,
            num_rows_override=train_rows_after_recent,
            clip_vocab=clip_vocab,
            # NOTE: PERIODIC_TIME_NS - train/valid must expose the same
            # synthetic timestamp categorical slots so model specs stay aligned.
            use_periodic_time_ns=use_periodic_time_ns,
            use_seq_periodic_hour_day_sideinfo=use_seq_periodic_hour_day_sideinfo,
            # NOTE: ITEM_ID_FEATURE - train/valid must expose the same
            # synthetic hashed item_id slot so item_int specs stay aligned.
            use_item_id_feature=use_item_id_feature,
            item_id_hash_buckets=item_id_hash_buckets,
        )

        valid_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            row_group_range=None,
            timestamp_min=valid_timestamp_min,
            num_rows_override=valid_rows_after_recent,
            clip_vocab=clip_vocab,
            # NOTE: PERIODIC_TIME_NS - validation mirrors training-side
            # synthetic timestamp categorical features.
            use_periodic_time_ns=use_periodic_time_ns,
            use_seq_periodic_hour_day_sideinfo=use_seq_periodic_hour_day_sideinfo,
            # NOTE: ITEM_ID_FEATURE - validation mirrors the training-side
            # hashed raw item_id feature.
            use_item_id_feature=use_item_id_feature,
            item_id_hash_buckets=item_id_hash_buckets,
        )

        train_rows = train_rows_after_recent
        valid_rows = valid_rows_after_recent
    else:
        n_valid_rgs = max(1, int(total_rgs * valid_ratio))
        n_train_rgs = total_rgs - n_valid_rgs

        # train_ratio: use only the first N% of the training Row Groups.
        if train_ratio < 1.0:
            n_train_rgs = max(1, int(n_train_rgs * train_ratio))
            logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

        train_rows_raw = sum(r[2] for r in rg_info[:n_train_rgs])
        valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

        timestamp_min = None
        train_rows_after_recent = train_rows_raw
        if train_recent_ratio < 1.0:
            if not (0.0 < train_recent_ratio <= 1.0):
                raise ValueError(f"train_recent_ratio must be in (0,1], got {train_recent_ratio}")
            ts_sample = _collect_primary_timestamps(rg_info, max_sample_size=split_sample_size)
            q = max(0.0, min(1.0, 1.0 - float(train_recent_ratio)))
            if ts_sample.size == 0:
                raise ValueError('no timestamps available for train_recent_ratio split')
            kth = int(np.floor(q * (ts_sample.size - 1)))
            timestamp_min = int(np.partition(ts_sample, kth)[kth])
            train_rows_after_recent = _count_rows_ge_timestamp(rg_info[:n_train_rgs], timestamp_min)
            logging.info(
                f"Recent timestamp filter enabled: keep latest {train_recent_ratio:.4f} rows; "
                f"delete_oldest_ratio={1.0 - train_recent_ratio:.4f}; timestamp_min={timestamp_min}; "
                f"estimated train rows after filter={train_rows_after_recent}/{train_rows_raw}")

        logging.info(f"Row Group split: {n_train_rgs} train ({train_rows_raw} raw rows, {train_rows_after_recent} after recent filter), "
                     f"{n_valid_rgs} valid ({valid_rows} rows)")

        train_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=shuffle_train,
            buffer_batches=buffer_batches,
            row_group_range=(0, n_train_rgs),
            timestamp_min=timestamp_min,
            num_rows_override=train_rows_after_recent if timestamp_min is not None else None,
            clip_vocab=clip_vocab,
            # NOTE: PERIODIC_TIME_NS - train/valid must expose the same
            # synthetic timestamp categorical slots so model specs stay aligned.
            use_periodic_time_ns=use_periodic_time_ns,
            use_seq_periodic_hour_day_sideinfo=use_seq_periodic_hour_day_sideinfo,
            # NOTE: ITEM_ID_FEATURE - train/valid must expose the same
            # synthetic hashed item_id slot so item_int specs stay aligned.
            use_item_id_feature=use_item_id_feature,
            item_id_hash_buckets=item_id_hash_buckets,
        )

        valid_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            row_group_range=(n_train_rgs, total_rgs),
            clip_vocab=clip_vocab,
            # NOTE: PERIODIC_TIME_NS - validation mirrors training-side
            # synthetic timestamp categorical features.
            use_periodic_time_ns=use_periodic_time_ns,
            use_seq_periodic_hour_day_sideinfo=use_seq_periodic_hour_day_sideinfo,
            # NOTE: ITEM_ID_FEATURE - validation mirrors the training-side
            # hashed raw item_id feature.
            use_item_id_feature=use_item_id_feature,
            item_id_hash_buckets=item_id_hash_buckets,
        )
        train_rows = train_rows_after_recent
    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
