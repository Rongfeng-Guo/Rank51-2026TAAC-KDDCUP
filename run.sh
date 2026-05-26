#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${SCRIPT_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${TRAIN_CKPT_PATH:-${SCRIPT_DIR}/ckpt}" \
         "${TRAIN_LOG_PATH:-${SCRIPT_DIR}/log}" \
         "${TRAIN_TF_EVENTS_PATH:-${SCRIPT_DIR}/events}"

NUM_EPOCHS="${NUM_EPOCHS:-7}"
PUBLISH_EPOCHS="${PUBLISH_EPOCHS:-1,2,4,5,6,7}"
TRAIN_RECENT_RATIO="${TRAIN_RECENT_RATIO:-0.90}"
VALID_RATIO="${VALID_RATIO:-0.016}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BUFFER_BATCHES="${BUFFER_BATCHES:-20}"
SPLIT_SAMPLE_SIZE="${SPLIT_SAMPLE_SIZE:-0}"
SEQ_HASH_ALLOWLIST="${SEQ_HASH_ALLOWLIST:-seq_b:69,seq_c:29,34,47}"
SEQ_HASH_BUCKET_SIZE="${SEQ_HASH_BUCKET_SIZE:-500000}"
SEQ_HASH_GATE_INIT="${SEQ_HASH_GATE_INIT:--0.75}"

echo "========== Time features + DIN pooling + recent90 overlap-tail =========="
echo "train_recent_ratio=${TRAIN_RECENT_RATIO}"
echo "valid_ratio=${VALID_RATIO}"
echo "num_epochs=${NUM_EPOCHS}; publish_epochs=${PUBLISH_EPOCHS}"
echo "Large-id hash supplement allowlist=${SEQ_HASH_ALLOWLIST}; bucket=${SEQ_HASH_BUCKET_SIZE}; gate_init=${SEQ_HASH_GATE_INIT}"
echo "======================================================================="

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train.py" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --buffer_batches "${BUFFER_BATCHES}" \
    --split_mode overlap_tail \
    --valid_ratio "${VALID_RATIO}" \
    --train_ratio 1.0 \
    --train_recent_ratio "${TRAIN_RECENT_RATIO}" \
    --split_sample_size "${SPLIT_SAMPLE_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --publish_epochs "${PUBLISH_EPOCHS}" \
    --patience 5 \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --query_pooling mean_din \
    --din_dropout 0.05 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --reinit_sparse_after_epoch 1 \
    --reinit_cardinality_threshold 0 \
    --seq_hash_bucket_size "${SEQ_HASH_BUCKET_SIZE}" \
    --seq_hash_gate_init "${SEQ_HASH_GATE_INIT}" \
    --seq_hash_allowlist "${SEQ_HASH_ALLOWLIST}" \
    --use_periodic_time_ns \
    --use_seq_periodic_hour_day_sideinfo \
    --use_user_dense_group_projector \
    --d_model 72 \
    "$@"
