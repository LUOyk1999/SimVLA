#!/bin/bash
# SimVLA Training Script for LeRobot LIBERO (Large Model)

set -e

BATCH_SIZE=${1:-256}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_libero_large_lerobot}
RESUME_CKPT=${4:-""}
TASK_SUITE_NAME=${5:-""}
CAMERA_MODE=${6:-dual}

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"
echo "   task_suite_name: ${TASK_SUITE_NAME:-'all'}"
echo "   camera_mode: ${CAMERA_MODE}"

GPU_DEVICES=${SIMVLA_CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${SIMVLA_NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${SIMVLA_MAIN_PROCESS_PORT:-29505}
MIXED_PRECISION=${SIMVLA_MIXED_PRECISION:-bf16}

export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}
export TF_CPP_MIN_LOG_LEVEL=2

DATASET_ROOT=${SIMVLA_LEROBOT_ROOT:-""}
DATASET_REPO_ID=${SIMVLA_LEROBOT_REPO_ID:-HuggingFaceVLA/libero}
SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

LEARNING_RATE=2e-4
NUM_ACTIONS=10
ITERS=200000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=10000
LOG_INTERVAL=20
NUM_WORKERS=4
MAX_GRAD_NORM=1.0

HIDDEN_SIZE=1024
DEPTH=24
NUM_HEADS=16
USE_ADALN=false

ARGS="--output_dir ${OUTPUT_DIR} \
    --dataset_backend lerobot_hf \
    --dataset_repo_id ${DATASET_REPO_ID} \
    --camera_mode ${CAMERA_MODE} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --max_grad_norm ${MAX_GRAD_NORM}"

if [ -n "${DATASET_ROOT}" ]; then
    ARGS="${ARGS} --dataset_root ${DATASET_ROOT}"
fi

if [ -n "${TASK_SUITE_NAME}" ]; then
    ARGS="${ARGS} --task_suite_name ${TASK_SUITE_NAME}"
fi

if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

echo "============================================================"
echo "Starting SimVLA Training on LeRobot LIBERO (Large)"
echo "============================================================"
echo "Dataset repo: ${DATASET_REPO_ID}"
echo "Dataset root: ${DATASET_ROOT:-'auto'}"
echo "Task suite: ${TASK_SUITE_NAME:-'all'}"
echo "Camera mode: ${CAMERA_MODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "num_processes: ${NUM_PROCESSES}"
echo "mixed_precision: ${MIXED_PRECISION}"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --mixed_precision ${MIXED_PRECISION} \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
