#!/usr/bin/env bash
# One-shot rebuild of the WiSP experiment environment on a fresh GPU pod
# (RunPod 3090-class or similar). Rebuilding a lost pod should cost one
# command and ~30 minutes, never an afternoon.
#
# Layout: heavy, disposable state lives on the LOCAL container disk (fast);
# canonical /workspace/* names are preserved as symlinks so every script in
# this repo works unchanged. If /workspace is a network volume, results
# written there survive pod loss.
#
#   $BASE/venv  -> /workspace/venv     vLLM + WiSP virtualenv
#   $BASE/hf    -> /workspace/hf       HF cache (also linked from ~/.cache)
#   $BASE/wisp  -> /workspace/wisp     this repo (BRANCH, default e1-baselines)
#
# Usage:  bash pod_setup.sh            # full: env + models + bandwidth probe
#         SKIP_MODELS=1 bash pod_setup.sh
# Idempotent: every step checks before doing.
set -uo pipefail
BASE=${BASE:-/root}
BRANCH=${BRANCH:-e1-baselines}
REPO_URL=${REPO_URL:-https://github.com/nokia-applied-research/WiSP.git}
VLLM_PIN=${VLLM_PIN:-0.11.2}
echo "=== pod_setup: BASE=$BASE BRANCH=$BRANCH vllm==$VLLM_PIN ==="

command -v cmake >/dev/null || (apt-get update -qq && apt-get install -y -qq cmake)

mkdir -p "$BASE/hf"
for name in venv hf wisp; do
  [ -e "/workspace/$name" ] || ln -s "$BASE/$name" "/workspace/$name"
done
# Kill the image's HF_HOME preset so nothing downloads to a second cache.
rm -rf /workspace/.cache/huggingface 2>/dev/null
mkdir -p /workspace/.cache && ln -sfn "$BASE/hf" /workspace/.cache/huggingface
mkdir -p "$HOME/.cache" && ln -sfn "$BASE/hf" "$HOME/.cache/huggingface"
export HF_HOME="$BASE/hf"

if [ ! -x "$BASE/venv/bin/python" ]; then
  python3 -m venv "$BASE/venv"
  "$BASE/venv/bin/pip" install -q --upgrade pip
fi
"$BASE/venv/bin/python" -c "import vllm" 2>/dev/null || \
  "$BASE/venv/bin/pip" install -q "vllm==$VLLM_PIN" hf_transfer pytest || { echo "SETUP-FAIL-VLLM"; exit 1; }

if [ ! -d "$BASE/wisp/.git" ]; then
  GIT_TERMINAL_PROMPT=0 git clone -q "$REPO_URL" "$BASE/wisp" 2>/dev/null || true
fi
if [ -d "$BASE/wisp/.git" ]; then
  GIT_TERMINAL_PROMPT=0 git -C "$BASE/wisp" fetch -q origin && git -C "$BASE/wisp" checkout -q "$BRANCH" \
    && git -C "$BASE/wisp" reset -q --hard "origin/$BRANCH"
else
  # Some datacenters proxy-break git smart-HTTP while plain HTTPS works
  # (seen on RunPod eu-cz: "could not read Username" on a public repo).
  # Fall back to the codeload tarball; no .git, but experiments don't care.
  echo "[setup] git clone failed; falling back to tarball of $BRANCH"
  TARBALL="${REPO_URL%.git}/tarball/$BRANCH"
  rm -rf "$BASE/wisp" && mkdir -p "$BASE/wisp"
  curl -sL "$TARBALL" | tar -xz -C "$BASE/wisp" --strip-components=1 \
    || { echo "SETUP-FAIL-CLONE"; exit 1; }
fi
"$BASE/venv/bin/pip" install -q -e "$BASE/wisp"
"$BASE/venv/bin/python" -c "import wisp, vllm; print('IMPORT-OK', vllm.__version__)" || { echo "SETUP-FAIL-IMPORT"; exit 1; }

if [ "${SKIP_MODELS:-0}" != "1" ]; then
  H="$BASE/venv/bin/hf"
  $H download Qwen/Qwen3-30B-A3B            >/dev/null 2>&1 && echo MODEL-QWEN-OK   || echo MODEL-QWEN-FAIL
  $H download allenai/OLMoE-1B-7B-0924      >/dev/null 2>&1 && echo MODEL-OLMOE-OK  || echo MODEL-OLMOE-FAIL
  $H download Qwen/Qwen3-30B-A3B-GGUF --include "*Q8_0*.gguf" >/dev/null 2>&1 && echo MODEL-GGUF-OK || echo MODEL-GGUF-FAIL
fi

"$BASE/venv/bin/python" "$BASE/wisp/scripts/m3_11_anvil_bandwidth_probe.py" \
    --out "$BASE/probe.json" 2>&1 | tail -6
echo "=== reminder: healthy PCIe 4.0 x16 pinned H2D is ~23-25 GB/s; below ~15 the pod is miswired — get a different one ==="
echo SETUP-DONE
