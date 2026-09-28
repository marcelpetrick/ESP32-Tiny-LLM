# Third-party notices

ESP32-Tiny-LLM is licensed under GPL-3.0-or-later (see [LICENSE](LICENSE)). It includes
or uses the third-party material listed below under their own licences. Ideas and results
that we cite without reusing code or data are credited in
[docs/07-prior-art.md](docs/07-prior-art.md).

Each entry records: what, origin (URL + version), licence, where it lives in this repo,
and how it is used. Entries are added in the same commit that introduces the material.

| Material | Origin | Licence | Location | Use |
|---|---|---|---|---|
| `run.c` (llama2.c inference) | [karpathy/llama2.c](https://github.com/karpathy/llama2.c) @ `350e04fe35433e6d2941dce5a1f53308f87058eb` | MIT, © 2023 Andrej Karpathy — [`third_party/llama2c/LICENSE`](third_party/llama2c/LICENSE) | `third_party/llama2c/run.c` (unmodified) | built only by tests as a numerical oracle (`tests/integration/test_llama2c_oracle.py`) |
| llama2.c tokenizer algorithm (idea + behaviour) | same repository, `encode()` / `decode()` in `run.c` | MIT | re-implemented in `runtime/src/tokenizer.c` (scored mode) and `training/tokenizer/llama2c.py` | load llama2.c checkpoints; attribution in both file headers |
| checkpoint format (layout) | llama2.c `export.py` legacy format | MIT | read by `tools/convert_llama2c.py` | conversion to `.tllm` |
| `stories260K.bin`, `tok512.bin`, `readme.md` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) (`stories260K/`) | MIT (model card) | `models/third_party/stories260K/` | vision M1 reproduction, oracle tests, story demo |
