# Optional models

Downloaded by `python tools/fetch_models.py`; nothing here is required.
Without these files the app falls back to heuristics / Windows SAPI.

| file | source | license | used by |
|---|---|---|---|
| `yamnet.onnx` + `yamnet_class_map.csv` | YAMNet (TF Hub / AudioSet), ONNX export from `hf-mirror.com/niobures/YAMNet` | Apache-2.0 | `screenlight/ai/scene_yamnet.py` |
| `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/` | [k2-fsa/sherpa-onnx KWS](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html) | Apache-2.0 | `screenlight/ai/kws.py` |

`kws_keywords.txt` (generated) contains the wake phrases: `亮起来` (pinyin tokens) and `LIGHT UP`.
