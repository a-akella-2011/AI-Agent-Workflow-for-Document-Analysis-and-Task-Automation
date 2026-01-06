# Local Llama2 Agent (llama-cpp-python)

This folder provides a local, charge-free agent variant that runs a Llama2-style model using llama-cpp-python.

Requirements and notes
- Requires a local GGML-compatible model file (e.g. `ggml-model-q4_0.bin`). Downloading Llama2 models requires agreeing to the model license from the provider.
- CPU inference can be slow; for acceptable speed use a machine with AVX/AVX2/AVX512 support or use a GPU-backed runtime.

Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements-local.txt
   ```
2. Download or obtain a GGML-compatible Llama2 model and set `LLAMA_MODEL_PATH` to the file path.
3. Add `.txt`/`.md` documents to `documents/` in the repo root.
4. Run:
   ```bash
   export AGENT_MODE=local
   export LLAMA_MODEL_PATH=/path/to/ggml-model.bin
   python agents/agent.py
   ```

Security
- The agent will not call external APIs when running in local mode (no API keys needed).
- Tools that modify resources require interactive confirmation by default.
