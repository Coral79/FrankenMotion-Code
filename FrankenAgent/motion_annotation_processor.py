"""FrankenAgent motion-annotation processor.

Sends each entry of a unified annotation JSON (from
github.com/Mathux/AMASS-Annotation-Unifier) to an LLM, asking it to
re-annotate the motion sequence at per-body-part granularity with temporal
boundaries. Per-keyid responses are saved as individual JSON files and then
combined into a single JSON whose keys are source_paths.

The LLM provider is user-supplied: pass ``--api_base_url`` (any
OpenAI-compatible endpoint) + ``--api_key`` + ``--model``. The released
FrankenMotion dataset was produced with DeepSeek-R1-0521 (model id
``deepseek-reasoner``) at temperature 0.

Usage:

    python motion_annotation_processor.py \\
        --input_json   path/to/babel_humanml3d_kitml.json \\
        --output_dir   ./out \\
        --api_base_url <YOUR_PROVIDER_BASE_URL> \\
        --api_key      <YOUR_API_KEY> \\
        --model        <YOUR_MODEL_ID>

By default uses ``prompts/label_process_sep_head_batch.txt`` (the production
prompt). Pass ``--prompt_file path/to/other.txt`` to override.
"""
import argparse
import concurrent.futures
import json
import logging
import os
import time
from datetime import datetime

from openai import OpenAI
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Question construction
# ---------------------------------------------------------------------------

def load_questions(input_json, prompt_file, source_paths_filter=None):
    """Build one question per source_path in the unified annotation JSON.

    Parameters
    ----------
    input_json : str
        Path to the unified annotation JSON. The schema is ``{source_path: {...}}``;
        see https://github.com/Mathux/AMASS-Annotation-Unifier for how to build it.
    prompt_file : str
        Path to a plain-text prompt template. The text is concatenated with a
        formatted JSON dump of the per-keyid entry; the LLM must respond with
        a fenced ```json block ending with the literal ``ANNOTATION COMPLETE``.
    source_paths_filter : Optional[Iterable[str]]
        If provided, only build questions for these source_paths (subset run).
        ``None`` means iterate every key in ``input_json``.
    """
    with open(prompt_file, "r") as f:
        prompt = f.read()

    with open(input_json, "r") as f:
        annotations = json.load(f)

    if source_paths_filter is not None:
        paths = [p for p in source_paths_filter if p in annotations]
        if len(paths) < len(source_paths_filter):
            missing = set(source_paths_filter) - set(paths)
            logging.warning(f"{len(missing)} requested source_paths not in input_json (e.g. {next(iter(missing))!r})")
    else:
        paths = list(annotations.keys())

    questions = []
    for source_path in paths:
        entry = annotations[source_path]
        formatted = json.dumps({source_path: entry}, indent=2)
        questions.append({
            "question": prompt + " " + formatted,
            "source_path": source_path,
        })
    return questions


# ---------------------------------------------------------------------------
# Response extraction (handles partial / malformed JSON)
# ---------------------------------------------------------------------------

def extract_json_from_response(response):
    """Extract a JSON object from a model response.

    The expected format is::

        ```json
        { ... }
        ```
        ANNOTATION COMPLETE

    The function tolerates: extra text after ``ANNOTATION COMPLETE``,
    ``Extra data`` errors (truncates at the offending position), unclosed
    structures (rewinds to the last balanced brace), missing markers (best
    effort).

    Returns
    -------
    (extracted_json, is_valid, error_message, status)
    """
    extracted_json = None
    is_valid = False
    error_message = None
    status = "unknown"

    try:
        start_marker = "```json"
        end_marker = "ANNOTATION COMPLETE"
        start_idx = response.find(start_marker)

        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = response.find(end_marker, start_idx)

            if end_idx != -1:
                json_str = response[start_idx:end_idx].strip()
                try:
                    extracted_json = json.loads(json_str)
                    is_valid, status = True, "success"
                except json.JSONDecodeError as e:
                    original_error = str(e)
                    error_message = f"Invalid JSON: {original_error}"
                    if "Extra data" in original_error:
                        try:
                            char_pos = int(original_error.split("char ")[1].split(")")[0])
                            extracted_json = json.loads(json_str[:char_pos])
                            is_valid, status = True, "truncated_success"
                        except Exception:
                            pass
                    elif any(err in original_error for err in
                             ["Expecting", "Unterminated string", "Missing value"]):
                        # Rewind to last balanced brace.
                        open_count = close_count = 0
                        in_string = escape_next = False
                        last_valid_pos = -1
                        for i, char in enumerate(json_str):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == "\\":
                                escape_next = True
                            elif char == '"' and not escape_next:
                                in_string = not in_string
                            elif not in_string:
                                if char == "{":
                                    open_count += 1
                                elif char == "}":
                                    close_count += 1
                                    if open_count == close_count:
                                        last_valid_pos = i + 1
                        if last_valid_pos > 0:
                            try:
                                extracted_json = json.loads(json_str[:last_valid_pos])
                                is_valid, status = True, "repaired_success"
                            except Exception:
                                pass
                    if not is_valid:
                        status = "failed_json_error"
            else:
                error_message = "End marker 'ANNOTATION COMPLETE' not found"
                status = "failed_incomplete"
                # Try to parse anyway (response may be cut off).
                try:
                    extracted_json = json.loads(response[start_idx:].strip())
                    is_valid, status = True, "partial_no_end_marker"
                except Exception:
                    pass
        else:
            error_message = "Start marker '```json' not found"
            status = "failed_format"
            try:
                brace_start = response.find("{")
                if brace_start != -1:
                    extracted_json = json.loads(response[brace_start:].strip())
                    is_valid, status = True, "recovered_no_markers"
            except Exception:
                pass
    except Exception as e:
        error_message = f"Error processing response: {e}"
        status = "exception"

    # Round-trip validation.
    if extracted_json is not None:
        try:
            json.loads(json.dumps(extracted_json))
        except Exception as e:
            extracted_json = None
            is_valid = False
            error_message = f"JSON validation failed: {e}"
            status = "failed_validation"

    return extracted_json, is_valid, error_message, status


# ---------------------------------------------------------------------------
# Single LLM call with retry + completion-marker check
# ---------------------------------------------------------------------------

def process_question_item(item, args, client):
    """Send one question to the LLM with up to 10 retries.

    Detects truncated responses (``ANNOTATION COMPLETE`` missing) and asks the
    model to continue once before retrying from scratch.
    """
    max_retries = 10
    retry_delay = 10

    answer = None
    question = item["question"]
    source_path = item.get("source_path", "unknown")

    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": question}]
            is_continuation = (
                answer is not None
                and "```json" in answer
                and "ANNOTATION COMPLETE" not in answer
            )
            if is_continuation:
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content":
                    "You were providing a JSON annotation but didn't complete it. "
                    "Please continue where you left off, and return the full json, "
                    "if it is complete end with ANNOTATION COMPLETE:\n\n"})

            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=0.7,
                max_tokens=args.max_tokens * (2 if is_continuation else 1),
                stream=True,
            )

            chunks = []
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    chunks.append(chunk.choices[0].delta.content)
            current_response = "".join(chunks)
            answer = current_response

            if "```json" in current_response and "ANNOTATION COMPLETE" in current_response:
                end_marker_pos = current_response.rfind("ANNOTATION COMPLETE")
                remainder = current_response[end_marker_pos + len("ANNOTATION COMPLETE"):].strip()
                _, is_valid, error_message, _ = extract_json_from_response(current_response)
                if is_valid:
                    status = "success" if not remainder else "partial_success"
                    logging.info(f"[{source_path}] {status}")
                    break
                if attempt < max_retries - 1:
                    logging.warning(f"[{source_path}] Invalid JSON: {error_message}. Retrying ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                status = "failed_json_error"
            else:
                if attempt < max_retries - 1:
                    logging.warning(f"[{source_path}] Incomplete answer. Retrying ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    status = "incomplete"
                else:
                    status = "failed_incomplete"

        except Exception as e:
            logging.error(f"[{source_path}] Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                backoff = retry_delay * (2 ** attempt)
                logging.info(f"[{source_path}] Retrying in {backoff}s...")
                time.sleep(backoff)
                status = "error_retrying"
            else:
                answer = "Error: " + str(e)
                status = "failed_error"

    return {
        "question": question,
        "answer": answer if answer else "Error: No answer generated after all attempts",
        "source_path": source_path,
        "status": status,
        "complete": "ANNOTATION COMPLETE" in (answer or "") and status in ["success", "partial_success"],
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Parallel orchestration + checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint_path, processed_items, failed_items):
    data = {
        "processed": processed_items,
        "failed": failed_items,
        "timestamp": datetime.now().isoformat(),
    }
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_answers(questions, args, checkpoint_path):
    """Run questions through the LLM in parallel (ThreadPoolExecutor).

    Resumes from ``checkpoint_path`` if it exists. Items with status starting
    ``failed`` are re-tried.
    """
    client = OpenAI(base_url=args.api_base_url, api_key=args.api_key)

    processed_items = {}
    failed_items = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                ckpt = json.load(f)
            processed_items = ckpt.get("processed", {})
            failed_items = ckpt.get("failed", [])
            logging.info(f"Resuming from checkpoint: {len(processed_items)} done, {len(failed_items)} failed")
        except Exception as e:
            logging.error(f"Could not read checkpoint ({e}); starting fresh")

    todo = []
    for q in questions:
        sp = q.get("source_path", "unknown")
        if sp in processed_items and sp not in failed_items:
            continue
        todo.append(q)
    logging.info(f"Processing {len(todo)} of {len(questions)} questions")

    results = []
    new_failed = []
    checkpoint_interval = max(1, min(10, len(todo) // 10))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_q = {executor.submit(process_question_item, q, args, client): q for q in todo}
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_q), total=len(todo)), start=1):
            q = future_to_q[future]
            sp = q.get("source_path", "unknown")
            try:
                r = future.result()
                results.append(r)
                processed_items[sp] = r["status"]
                if r["status"].startswith("failed") or not r.get("complete", False):
                    new_failed.append(sp)
            except Exception as e:
                logging.error(f"[{sp}] exception in worker: {e}")
                results.append({
                    "question": q["question"],
                    "answer": f"Error: {e}",
                    "source_path": sp,
                    "status": "exception",
                    "complete": False,
                    "timestamp": datetime.now().isoformat(),
                })
                processed_items[sp] = "exception"
                new_failed.append(sp)
            if checkpoint_path and i % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, processed_items, new_failed + failed_items)

    if checkpoint_path:
        save_checkpoint(checkpoint_path, processed_items, new_failed + failed_items)

    return results, {
        "total": len(questions),
        "processed": len(processed_items),
        "successful": sum(1 for s in processed_items.values() if s in ["success", "partial_success"]),
        "failed": len(new_failed + failed_items),
        "failed_items": new_failed + failed_items,
    }


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def save_individual_jsons(answers, output_dir):
    """Write one JSON file per source_path under ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    valid = 0
    failed = []
    for item in tqdm(answers, desc="Saving individual JSONs"):
        sp = item.get("source_path", "unknown")
        if not sp.endswith(".json"):
            out = os.path.join(output_dir, f"{sp}.json")
        else:
            out = os.path.join(output_dir, sp)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        extracted, is_valid, err, status = extract_json_from_response(item.get("answer", ""))
        if is_valid:
            with open(out, "w") as f:
                json.dump(extracted, f, indent=2)
            valid += 1
        else:
            failed.append({"source_path": sp, "error": err, "status": status})
    if failed:
        with open(os.path.join(output_dir, "failed_extractions.json"), "w") as f:
            json.dump({"failed_items": failed}, f, indent=2)
    return {"total": len(answers), "valid": valid, "failed": len(failed)}


def combine_individual_jsons(input_dir, output_file):
    """Walk ``input_dir`` and combine every per-source-path JSON into one file."""
    combined = {}
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.endswith(".json") or name == "failed_extractions.json":
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, input_dir)
            sp = os.path.splitext(rel)[0]
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and sp in data:
                    combined[sp] = data[sp]  # un-nest if double-keyed
                else:
                    combined[sp] = data
            except Exception as e:
                logging.error(f"Could not read {path}: {e}")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2)
    return {"files_seen": sum(1 for _ in os.walk(input_dir) for _ in _[2]), "combined_entries": len(combined)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Inputs.
    p.add_argument("--input_json", required=True,
                   help="Unified annotation JSON (from AMASS-Annotation-Unifier).")
    p.add_argument("--output_dir", required=True,
                   help="Directory for per-source-path JSON outputs + checkpoints.")
    p.add_argument("--prompt_file", default=None,
                   help="Prompt template. Default: prompts/label_process_sep_head_batch.txt next to this script.")
    p.add_argument("--source_paths_file", default=None,
                   help="Optional .txt of source_paths to subset (one per line).")

    # LLM provider — user-supplied; no defaults.
    p.add_argument("--api_base_url", required=True,
                   help="OpenAI-compatible API base URL (e.g. your provider's /v1 endpoint).")
    p.add_argument("--api_key", required=True,
                   help="API key for the provider above. Get your own; never commit a key.")
    p.add_argument("--model", required=True,
                   help='Model identifier on your provider (e.g. "deepseek-reasoner").')

    # Generation knobs.
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--max_tokens", type=int, default=9000)
    p.add_argument("--max_workers", type=int, default=10,
                   help="Parallel LLM calls (respect your provider's rate limits).")
    args = p.parse_args()

    # Resolve default prompt to the one shipped next to this script.
    if args.prompt_file is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.prompt_file = os.path.join(here, "prompts", "label_process_sep_head_batch.txt")
        logging.info(f"Using default prompt: {args.prompt_file}")

    # Load source-path filter if provided.
    source_paths_filter = None
    if args.source_paths_file:
        with open(args.source_paths_file, "r") as f:
            source_paths_filter = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(source_paths_filter)} source_paths from filter")

    # Build questions.
    questions = load_questions(args.input_json, args.prompt_file, source_paths_filter)
    logging.info(f"Built {len(questions)} questions")

    os.makedirs(args.output_dir, exist_ok=True)
    individual_dir = os.path.join(args.output_dir, "individual_jsons")
    os.makedirs(individual_dir, exist_ok=True)
    checkpoint_file = os.path.join(args.output_dir, f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Run.
    answers, summary = generate_answers(questions, args, checkpoint_path=checkpoint_file)

    # Persist raw answers.
    raw_jsonl = os.path.join(args.output_dir, "answers.jsonl")
    with open(raw_jsonl, "w") as f:
        for ans in answers:
            f.write(json.dumps(ans) + "\n")

    # Per-source-path JSONs + combined.
    save_stats = save_individual_jsons(answers, individual_dir)
    combined_file = os.path.join(args.output_dir, "combined_annotations.json")
    combine_stats = combine_individual_jsons(individual_dir, combined_file)

    print("=" * 60)
    print("FrankenAgent annotation complete")
    print("=" * 60)
    print(f"Questions:         {summary['total']}")
    print(f"Successful:        {summary['successful']}")
    print(f"Failed:            {summary['failed']}")
    print(f"Per-keyid JSONs:   {save_stats['valid']}/{save_stats['total']}  ({individual_dir})")
    print(f"Combined output:   {combined_file}  ({combine_stats['combined_entries']} entries)")
    print(f"Raw responses:     {raw_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    main()
