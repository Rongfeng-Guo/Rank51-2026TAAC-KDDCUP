"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    FeatureSchema,
    get_pcvr_data,
    NUM_TIME_BUCKETS,
    PERIODIC_TIME_NS_FIDS,
    ITEM_ID_FEATURE_FID,
    ITEM_ID_HASH_BUCKETS_DEFAULT,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def append_periodic_time_ns_to_user_groups(
    user_ns_groups: List[List[int]],
    user_fid_to_idx: Dict[int, int],
) -> List[List[int]]:
    """Append synthetic timestamp feature indices to the last User NS group."""
    # NOTE: PERIODIC_TIME_NS - keep the User NS token count unchanged by
    # merging timestamp-derived synthetic categorical features into the last
    # existing user group instead of creating additional NS groups.
    time_indices = []
    for fid in PERIODIC_TIME_NS_FIDS:
        if fid not in user_fid_to_idx:
            raise KeyError(
                f"Periodic time NS fid={fid} is missing from user_int_schema; "
                "make sure --use_periodic_time_ns was passed to the dataset")
        time_indices.append(user_fid_to_idx[fid])

    time_idx_set = set(time_indices)
    groups = [
        [idx for idx in group if idx not in time_idx_set]
        for group in user_ns_groups
    ]
    groups = [group for group in groups if group]
    if groups:
        groups[-1].extend(time_indices)
    else:
        groups = [time_indices]
    logging.info(
        "Periodic time NS synthetic fids appended to last user group: %s",
        PERIODIC_TIME_NS_FIDS,
    )
    return groups


def append_item_id_feature_to_item_groups(
    item_ns_groups: List[List[int]],
    item_fid_to_idx: Dict[int, int],
) -> List[List[int]]:
    """Append the synthetic item_id feature index to the last Item NS group."""
    # NOTE: ITEM_ID_FEATURE - keep the Item NS token count unchanged by
    # merging the hashed raw-item_id synthetic feature into the last existing
    # item group instead of creating an additional Item NS token.
    if ITEM_ID_FEATURE_FID not in item_fid_to_idx:
        raise KeyError(
            f"Synthetic item_id fid={ITEM_ID_FEATURE_FID} is missing from item_int_schema; "
            "make sure --use_item_id_feature was passed to the dataset")

    item_idx = item_fid_to_idx[ITEM_ID_FEATURE_FID]
    groups = [
        [idx for idx in group if idx != item_idx]
        for group in item_ns_groups
    ]
    groups = [group for group in groups if group]
    if groups:
        groups[-1].append(item_idx)
    else:
        groups = [[item_idx]]
    logging.info(
        "Synthetic item_id fid appended to last item group: %s",
        ITEM_ID_FEATURE_FID,
    )
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--publish_epochs', type=str, default='',
                        help='Comma-separated epoch indices to export as '
                             'additional self-contained checkpoints for '
                             'manual publish, e.g. "4,5,6,7"')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Validation ratio. In overlap_tail mode this is the latest-row monitor ratio.')
    parser.add_argument('--split_mode', type=str, default='row_group_tail',
                        choices=['row_group_tail', 'row_group', 'overlap_tail'],
                        help='row_group_tail/row_group = original baseline row-group tail validation; '
                             'overlap_tail = train on latest train_recent_ratio rows by primary timestamp and '
                             'monitor on latest valid_ratio rows, with valid included in train.')
    parser.add_argument('--train_recent_ratio', type=float, default=1.0,
                        help='If <1, keep only the latest fraction of training rows by primary timestamp. '
                             'Example: 0.85 deletes the oldest 15%% while keeping the original row-group valid split.')
    parser.add_argument('--split_sample_size', type=int, default=0,
                        help='Max timestamps sampled to estimate the recent split threshold; 0 = exact scan')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')
    # NOTE: PERIODIC_TIME_NS - optional timestamp-derived categorical user_int
    # features. This flag is one-way/default-off so old experiments and
    # checkpoints keep their historical input shapes.
    parser.add_argument('--use_periodic_time_ns', action='store_true', default=False,
                        help='Append timestamp-derived categorical features '
                             '(hour/day/weekend/time-of-day bucket) to User NS')
    parser.add_argument(
        '--use_seq_periodic_hour_day_sideinfo',
        action='store_true',
        default=False,
        help='Append sequence event hour-of-day and day-of-week as synthetic side-info features.',
    )
    # NOTE: ITEM_ID_FEATURE - optional hashed raw-item_id categorical feature
    # appended to item_int_feats at runtime without changing schema.json.
    parser.add_argument('--use_item_id_feature', action='store_true', default=False,
                        help='Append raw parquet item_id as a hashed synthetic '
                             'item-side categorical feature')
    parser.add_argument('--item_id_hash_buckets', type=int,
                        default=ITEM_ID_HASH_BUCKETS_DEFAULT,
                        help='Number of positive hash buckets used by the '
                             'synthetic item_id feature '
                             '(effective only when --use_item_id_feature is enabled)')
    # NOTE: USER_DENSE_SMOOTH_CLIP - default off so baseline behavior and
    # parameter count stay unchanged unless the flag is explicitly passed.
    parser.add_argument('--use_user_dense_smooth_clip',
                        action='store_true',
                        default=False,
                        help='Enable smooth clipping + learnable affine transform '
                             'for user_dense_feats before projection.')
    # NOTE: USER_DENSE_GROUP_PROJECTOR - Optional replacement for the original
    # single user_dense token; splits dense fids into emb/stat/quantile tokens.
    parser.add_argument(
        '--use_user_dense_group_projector',
        action='store_true',
        help=(
            'Replace the original single user_dense token with 3 grouped '
            'dense tokens: emb group, stat group, and quantile trend group.'
        ),
    )
    parser.add_argument(
        '--dense_stat_log_clamp_max',
        type=float,
        default=20.0,
        help=(
            'Maximum value after log1p transform for stat/count user dense '
            'group. Only effective when --use_user_dense_group_projector is '
            'enabled.'
        ),
    )
    parser.add_argument(
        '--dense_quantile_mid_dim',
        type=int,
        default=64,
        help=(
            'Hidden channel size for QuantileTrendEncoder Conv1d. Only '
            'effective when --use_user_dense_group_projector is enabled.'
        ),
    )
    parser.add_argument(
        '--dense_group_dropout',
        type=float,
        default=0.05,
        help=(
            'Dropout used inside UserDenseGroupProjector. Only effective when '
            '--use_user_dense_group_projector is enabled.'
        ),
    )

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--query_pooling', type=str, default='mean',
                        choices=['mean', 'mean_din'],
                        help='Sequence summary used by QueryGenerator: '
                             'mean = masked mean pooling; '
                             'mean_din = masked mean plus item-conditioned DIN pooling')
    parser.add_argument('--din_dropout', type=float, default=0.0,
                        help='Dropout inside the mean_din scorer MLP '
                             '(effective only when --query_pooling=mean_din)')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')
    parser.add_argument('--seq_hash_bucket_size', type=int, default=0,
                        help='If >0, allowlisted sequence fields that would be skipped by '
                             '--emb_skip_threshold are mapped to a fixed-size hash embedding table.')
    parser.add_argument('--seq_hash_gate_init', type=float, default=-1.0,
                        help='Initial logit for learnable hash gates. '
                             'Examples: -2.0=>0.119, -1.5=>0.182, -1.0=>0.269.')
    parser.add_argument('--seq_hash_allowlist', type=str, default='',
                        help='Comma-separated fids to hash, e.g. "seq_b:69,seq_c:29,34,47". '
                             'Use "all" to hash every sequence field skipped by emb_skip_threshold.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    args = parser.parse_args()
    if args.item_id_hash_buckets <= 0:
        parser.error('--item_id_hash_buckets must be > 0')
    if not 0.0 <= args.din_dropout <= 1.0:
        parser.error('--din_dropout must be in [0, 1]')
    # NOTE: USER_DENSE_GROUP_PROJECTOR - Validate grouped dense projector knobs
    # before persisting train_config.json so train/infer shapes agree.
    if args.dense_stat_log_clamp_max <= 0:
        parser.error('--dense_stat_log_clamp_max must be > 0')
    if args.dense_quantile_mid_dim <= 0:
        parser.error('--dense_quantile_mid_dim must be > 0')
    if not 0.0 <= args.dense_group_dropout <= 1.0:
        parser.error('--dense_group_dropout must be in [0, 1]')

    # NOTE: PUBLISH_EPOCHS - parse the UI-facing extra checkpoint export list
    # once at startup so trainer.py receives a validated integer list and the
    # same values are persisted into train_config.json.
    publish_epochs = []
    if args.publish_epochs:
        try:
            publish_epochs = [
                int(token.strip()) for token in args.publish_epochs.split(',')
                if token.strip()
            ]
        except ValueError as exc:
            parser.error(f'--publish_epochs must be a comma-separated list of positive integers: {exc}')
        if any(epoch <= 0 for epoch in publish_epochs):
            parser.error('--publish_epochs must contain only positive integers')
    args.publish_epochs = sorted(set(publish_epochs))

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")
    logging.info(f"use_user_dense_group_projector={args.use_user_dense_group_projector}")
    logging.info(f"dense_stat_log_clamp_max={args.dense_stat_log_clamp_max}")
    logging.info(f"dense_quantile_mid_dim={args.dense_quantile_mid_dim}")
    logging.info(f"dense_group_dropout={args.dense_group_dropout}")
    logging.info(
        "use_seq_periodic_hour_day_sideinfo=%s",
        args.use_seq_periodic_hour_day_sideinfo,
    )

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        split_mode=args.split_mode,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        train_recent_ratio=args.train_recent_ratio,
        split_sample_size=args.split_sample_size,
        # NOTE: PERIODIC_TIME_NS - train/valid expose the same synthetic
        # timestamp categorical features and vars(args) persists this switch
        # for infer.py to rebuild the same model shape.
        use_periodic_time_ns=args.use_periodic_time_ns,
        use_seq_periodic_hour_day_sideinfo=args.use_seq_periodic_hour_day_sideinfo,
        # NOTE: ITEM_ID_FEATURE - train/valid expose the same synthetic hashed
        # raw item_id slot and vars(args) persists both the switch and bucket
        # count for infer.py to rebuild the same item_int shape.
        use_item_id_feature=args.use_item_id_feature,
        item_id_hash_buckets=args.item_id_hash_buckets,
    )

    # ---- NS groups ----
    user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
    item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]
    if args.use_periodic_time_ns:
        user_ns_groups = append_periodic_time_ns_to_user_groups(
            user_ns_groups, user_fid_to_idx)
    if args.use_item_id_feature:
        item_ns_groups = append_item_id_feature_to_item_groups(
            item_ns_groups, item_fid_to_idx)

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "seq_feature_ids": pcvr_dataset.sideinfo_fids,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        # NOTE: USER_DENSE_GROUP_PROJECTOR - dense fid spans are looked up
        # from schema entries so grouped projection follows schema layout.
        "user_dense_schema_entries": pcvr_dataset.user_dense_schema.entries,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "query_pooling": args.query_pooling,
        "din_dropout": args.din_dropout,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "seq_hash_bucket_size": args.seq_hash_bucket_size,
        "seq_hash_gate_init": args.seq_hash_gate_init,
        "seq_hash_allowlist": args.seq_hash_allowlist,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        # NOTE: USER_DENSE_SMOOTH_CLIP - passed through to model construction
        # and persisted via vars(args) in train_config.json.
        "use_user_dense_smooth_clip": args.use_user_dense_smooth_clip,
        # NOTE: USER_DENSE_GROUP_PROJECTOR - Structural grouped dense-token
        # replacement; vars(args) persists the exact train-side knobs.
        "use_user_dense_group_projector": args.use_user_dense_group_projector,
        "dense_stat_log_clamp_max": args.dense_stat_log_clamp_max,
        "dense_quantile_mid_dim": args.dense_quantile_mid_dim,
        "dense_group_dropout": args.dense_group_dropout,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(
        f"PCVRHyFormer model created: num_user_dense_tokens={model.num_user_dense_tokens}, "
        f"num_ns={num_ns}, T={T}, d_model={args.d_model}, "
        f"rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        publish_epochs=args.publish_epochs,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
