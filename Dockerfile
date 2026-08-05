# Reproduction image for WiSP. Starts from the vLLM image whose version
# matches the paper, then installs the plug-in. The plug-in auto-registers
# with vLLM via its entry point, so `vllm serve ...` is WiSP-enabled.
FROM vllm/vllm-openai:v0.11.2

WORKDIR /opt/wisp
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY reproduce.sh ./

# Editable install so the entry point is registered; vLLM/torch already
# present in the base image, so we don't reinstall them.
RUN pip install --no-deps -e .

ENV VLLM_USE_V1=1 \
    WISP_MODE=paged

# Default: drop into a shell. To reproduce the paper claims:
#   docker run --gpus all -e HF_TOKEN=... <image> bash reproduce.sh
ENTRYPOINT ["/bin/bash"]
