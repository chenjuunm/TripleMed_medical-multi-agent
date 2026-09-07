#!/usr/bin/env bash
set -euo pipefail

LMS_EXECUTABLE="${LMS_EXECUTABLE:-${HOME}/.lmstudio/bin/lms}"
LM_STUDIO_PORT="${LM_STUDIO_PORT:-1234}"

# 9B 模型统一强制使用 4bit，避免依赖 LM Studio 当前选中的 variant。
# 27B 主模型仍允许在 Q6_K 与 Q4_K_M 之间按机器内存切换。
MAIN_MODEL_KEY="${LM_STUDIO_MAIN_MODEL_KEY:-qwen/qwen3.8-27b}"
ROUTER_MODEL_KEY="${LM_STUDIO_ROUTER_MODEL_KEY:-qwen/qwen3.5-9b}"


MAIN_IDENTIFIER="${AGENT_MODEL:-medical-main-qwen38-27b}"
ROUTER_IDENTIFIER="${ROUTER_MODEL:-medical-router-qwen35-9b}"

MAIN_CONTEXT_LENGTH="${LM_STUDIO_MAIN_CONTEXT_LENGTH:-8192}"
ROUTER_CONTEXT_LENGTH="${LM_STUDIO_ROUTER_CONTEXT_LENGTH:-4096}"
MAIN_PARALLEL="${LM_STUDIO_MAIN_PARALLEL:-2}"
MAIN_TTL_SECONDS="${LM_STUDIO_MAIN_TTL_SECONDS:-3600}"
ROUTER_TTL_SECONDS="${LM_STUDIO_ROUTER_TTL_SECONDS:-3600}"

if [[ ! -x "${LMS_EXECUTABLE}" ]]; then
    echo "LM Studio CLI 不可执行: ${LMS_EXECUTABLE}" >&2
    exit 1
fi

"${LMS_EXECUTABLE}" server start --port "${LM_STUDIO_PORT}"

loaded_instance_for() {
    "${LMS_EXECUTABLE}" ps --json \
        | sed 's/},{/}\
{/g' \
        | grep -F "\"identifier\":\"$1\"" \
        || true
}

MAIN_INSTANCE="$(loaded_instance_for "${MAIN_IDENTIFIER}")"
if [[ -n "${MAIN_INSTANCE}" ]] \
    && is_9b_model_key "${MAIN_MODEL_KEY}" \
    && [[ "${MAIN_INSTANCE}" != *"\"selectedVariant\":\"${MAIN_MODEL_KEY}\""* ]]; then
    echo "主模型是 9B 但未使用 4bit，正在重载: ${MAIN_IDENTIFIER}"
    "${LMS_EXECUTABLE}" unload "${MAIN_IDENTIFIER}"
    MAIN_INSTANCE=""
fi

if [[ -n "${MAIN_INSTANCE}" ]]; then
    echo "主模型已加载，跳过重复加载: ${MAIN_IDENTIFIER}"
else
    "${LMS_EXECUTABLE}" load "${MAIN_MODEL_KEY}" \
        --identifier "${MAIN_IDENTIFIER}" \
        --context-length "${MAIN_CONTEXT_LENGTH}" \
        --parallel "${MAIN_PARALLEL}" \
        --gpu max \
        --ttl "${MAIN_TTL_SECONDS}" \
        --yes
fi

ROUTER_INSTANCE="$(loaded_instance_for "${ROUTER_IDENTIFIER}")"
if [[ -n "${ROUTER_INSTANCE}" ]] \
    && is_9b_model_key "${ROUTER_MODEL_KEY}" \
    && [[ "${ROUTER_INSTANCE}" != *"\"selectedVariant\":\"${ROUTER_MODEL_KEY}\""* ]]; then
    echo "Router 是 9B 但未使用 4bit，正在重载: ${ROUTER_IDENTIFIER}"
    "${LMS_EXECUTABLE}" unload "${ROUTER_IDENTIFIER}"
    ROUTER_INSTANCE=""
fi

if [[ -n "${ROUTER_INSTANCE}" ]]; then
    echo "Router 已加载，跳过重复加载: ${ROUTER_IDENTIFIER}"
else
    "${LMS_EXECUTABLE}" load "${ROUTER_MODEL_KEY}" \
        --identifier "${ROUTER_IDENTIFIER}" \
        --context-length "${ROUTER_CONTEXT_LENGTH}" \
        --parallel 1 \
        --gpu max \
        --ttl "${ROUTER_TTL_SECONDS}" \
        --yes
fi

echo "LM Studio 服务与模型已准备完成。"
echo "主模型 identifier: ${MAIN_IDENTIFIER}"
echo "Router identifier: ${ROUTER_IDENTIFIER}"
echo "OpenAI-compatible base URL: http://127.0.0.1:${LM_STUDIO_PORT}/v1"
"${LMS_EXECUTABLE}" ps
