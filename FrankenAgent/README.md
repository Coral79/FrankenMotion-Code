# FrankenAgent

LLM-based annotation pipeline that produced the [Frankenstein Dataset](https://huggingface.co/datasets/Coral79/frankenstein-dataset) used by FrankenMotion.

FrankenAgent reads a unified motion-annotation JSON (BABEL + HumanML3D + KIT-ML, merged via [AMASS-Annotation-Unifier](https://github.com/Mathux/AMASS-Annotation-Unifier)) and asks an LLM to re-annotate each sequence at **per-body-part granularity with temporal boundaries**, producing the structured JSON shipped on HuggingFace.

```
unified annotation JSON (per source_path: text + start/end ranges from BABEL/HumanML3D/KIT-ML)
        ↓
[FrankenAgent: parallel LLM calls, one per source_path]
        ↓
per-source-path JSONs   →   combined annotations.json
                            (head_text, left_arm_text, ..., trajectory_text + per-part start/end ranges)
```

## Input

The script consumes the unified annotation JSON produced by Mathieu Petrovich's [AMASS-Annotation-Unifier](https://github.com/Mathux/AMASS-Annotation-Unifier) — clone that repo, follow its README to download BABEL + HumanML3D + KIT-ML annotations under their respective licenses, and run its `merge.py`. The output looks like:

```json
{
  "ACCAD/Female1General_c3d/A1 - Stand_poses": {
    "duration": 3.0,
    "annotations": [
      {"seg_id": "babel_10890_seq_0", "babel_id": "...", "text": "stand",      "start": 0.0, "end": 3.0},
      {"seg_id": "humanml3d_004501_0",                     "text": "facing forward", "start": 0.0, "end": 3.0}
    ]
  },
  ...
}
```

The Frankenstein dataset on HuggingFace was built from such a JSON. A 2-sequence sample is shipped here as `example_unified.json` for quick smoke-testing.

## LLM provider

We used **DeepSeek-R1-0521**. Supply your own `--api_base_url`, `--api_key`, and `--model` for whichever OpenAI-compatible provider you choose. **Never commit your API key.**

## Setup

```bash
pip install openai tqdm
```

Tested with `openai>=1.0`.

## Run

```bash
# Smoke test on the shipped 2-sequence sample:
python motion_annotation_processor.py \
    --input_json   FrankenAgent/example_unified.json \
    --output_dir   ./out \
    --api_base_url <YOUR_PROVIDER_BASE_URL> \
    --api_key      <YOUR_API_KEY> \
    --model        <YOUR_MODEL_ID>
```

Optional:
- `--prompt_file path/to/other.txt` — override the default prompt (see below).
- `--source_paths_file paths.txt` — only process source_paths listed in this file (one per line). Useful for re-running failed items or a subset.
- `--max_workers N` — parallel LLM calls; respect your provider's rate limits.
- `--max_tokens N` — generation cap (default 9000; raise if your model can return larger annotations).
- `--temperature T` — default 0 for reproducibility.

Re-runs resume automatically from the most recent `checkpoint_*.json` in `--output_dir`.

## Prompts

Two prompts are provided. Both produce the same annotation schema (per-body-part text descriptions with `start`/`end` time ranges); they differ in how they handle long motion sequences.

### `label_process_sep_head.txt` — whole-sequence mode

The LLM annotates the entire motion sequence in a single call. The output covers the full duration `[0, motion_duration]` for each body part.

Recommended for: short motions, simple inputs, debugging the prompt.

### `label_process_sep_head_batch.txt` — windowed-batch mode (default)

*Used to produce the released FrankenMotion dataset.* The LLM annotates only within a specified time window `[start_time, end_time]`. Long motions are pre-split into batches; each batch is annotated independently and the per-batch outputs are then concatenated. The prompt enforces:

- annotations are **hard-clipped** to the batch window (no overflow into adjacent batches)
- segments outside the window are passed as **context only** (the LLM sees them but does not annotate them)
- per-body-part time coverage has **no gaps** within the window

Recommended for: long motions (> 20s), production runs, reproducing the FrankenMotion dataset.

## Output

```
<output_dir>/
├── checkpoint_<timestamp>.json     # resumable checkpoint
├── answers.jsonl                   # raw LLM responses (one line per source_path)
├── individual_jsons/               # one JSON per source_path with the extracted annotation
│   └── <source_path>.json
└── combined_annotations.json       # all per-source-path JSONs merged into one keyid → annotation dict
```

The format of each per-keyid annotation matches the schema documented at https://huggingface.co/datasets/Coral79/frankenstein-dataset.

## Adapting to a different dataset

`motion_annotation_processor.py` is dataset-agnostic in its iteration: it walks every key in `--input_json`. To adapt to a non-AMASS source you only need to:

1. Build a unified JSON in the AMASS-Annotation-Unifier output schema (see Input above), where each top-level key is a stable source identifier and the value contains `duration` and an `annotations` list with `text`/`start`/`end` per existing annotation.
2. Decide which prompt fits — adapt the body-part vocabulary or temporal-window rules in the prompt text if your dataset has different parts or different temporal granularity than what the released prompts assume.
3. Run as above.

## Notes

- Failed items (status starting `failed_*` or `incomplete`) are tracked in the checkpoint and re-tried on the next run.
- For reference, the production run that produced the shipped dataset used `temperature=0`, `max_tokens=9000`, `max_workers=10`, and the `_sep_head_batch` prompt.
