#!/usr/bin/env python3
"""
Stream The Pile (uncopyrighted parquet mirror) and save per-domain JSONL files.
Each line: {"text": "<raw text>"}

Stops after collecting ~target_chars per domain (approx proxy for later tokens).

Default dataset: monology/pile-uncopyrighted-parquet
(Recommended by dataset owner as a workable parquet mirror for partial use.) :contentReference[oaicite:2]{index=2}
Note: "pile-uncopyrighted" removes Books3, BookCorpus2, OpenSubtitles, YTSubtitles, OWT2. :contentReference[oaicite:3]{index=3}
"""

import argparse
import json
import os
import re
import signal
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

DEFAULT_DOMAINS = [
    "USPTO Backgrounds",
    "PubMed Central",
    "FreeLaw",
    "DM Mathematics",
    "PhilPapers",
    "Ubuntu IRC",
]


def sanitize_domain(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


class GracefulStop(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="monology/pile-uncopyrighted-parquet",
        help="HF dataset repo (default: monology/pile-uncopyrighted-parquet).",
    )
    p.add_argument("--split", type=str, default="train", help="Split to stream (default: train).")
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision (e.g., refs/convert/parquet if using monology/pile-uncopyrighted).",
    )
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for domain JSONL files.")
    p.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=None,
        help="List of meta['pile_set_name'] domains. If omitted, uses DEFAULT_DOMAINS.",
    )
    p.add_argument("--target_chars", type=int, default=80_000_000, help="Target total characters per domain.")
    p.add_argument(
        "--max_samples", type=int, default=None, help="Optional global cap on number of samples processed (safety)."
    )
    p.add_argument("--shuffle_buffer", type=int, default=0, help="Streaming shuffle buffer (0 disables shuffle).")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    p.add_argument("--resume", action="store_true", default=True, help="Resume if outputs already exist (default: on).")
    p.add_argument("--no_resume", dest="resume", action="store_false", help="Overwrite existing outputs.")
    p.add_argument(
        "--list_domains_only",
        action="store_true",
        default=False,
        help="Scan first N samples and print pile_set_name counts, then exit.",
    )
    p.add_argument(
        "--list_domains_max_samples", type=int, default=200_000, help="Max samples to scan in --list_domains_only mode."
    )
    args = p.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains: List[str] = args.domains if args.domains else list(DEFAULT_DOMAINS)
    domain_set = set(domains)

    # Load dataset (streaming always, since you want partial extraction)
    ds = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        revision=args.revision,
    )

    if args.shuffle_buffer and args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    # Domain-counting mode (helps discover exact pile_set_name strings)
    if args.list_domains_only:
        counts: Dict[str, int] = {}
        for i, ex in enumerate(ds):
            meta = ex.get("meta", {}) or {}
            name = meta.get("pile_set_name", None)
            if name is not None:
                counts[name] = counts.get(name, 0) + 1
            if i + 1 >= args.list_domains_max_samples:
                break
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        print(f"Observed pile_set_name counts in first {args.list_domains_max_samples} samples:")
        for k, v in top[:80]:
            print(f"{k}\t{v}")
        return

    # Resume bookkeeping via meta.json per domain
    writers = {}
    char_counts = {}
    line_counts = {}

    def load_resume_meta(dom_dir: Path) -> Dict:
        meta_path = dom_dir / "meta.json"
        if args.resume and meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception:
                return {}
        return {}

    for dom in domains:
        dom_dir = out_dir / sanitize_domain(dom)
        dom_dir.mkdir(parents=True, exist_ok=True)
        out_path = dom_dir / "data.jsonl"
        meta = load_resume_meta(dom_dir)

        if (not args.resume) and out_path.exists():
            out_path.unlink()

        mode = "a" if (args.resume and out_path.exists()) else "w"
        f = open(out_path, mode, encoding="utf-8")

        # Use meta counts if available; otherwise 0
        char_counts[dom] = int(meta.get("chars_written", 0))
        line_counts[dom] = int(meta.get("lines_written", 0))
        writers[dom] = f

    stop_now = {"flag": False}

    def handle_sigint(sig, frame):
        stop_now["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    total_target = len(domains) * args.target_chars
    already_done = sum(min(char_counts[d], args.target_chars) for d in domains)

    pbar = tqdm(total=total_target, initial=already_done, unit="ch", desc="Collecting text")

    def domain_done(dom: str) -> bool:
        return char_counts[dom] >= args.target_chars

    def write_meta(dom: str):
        dom_dir = out_dir / sanitize_domain(dom)
        meta_path = dom_dir / "meta.json"
        payload = {
            "domain": dom,
            "dataset": args.dataset,
            "revision": args.revision,
            "split": args.split,
            "target_chars": int(args.target_chars),
            "chars_written": int(char_counts[dom]),
            "lines_written": int(line_counts[dom]),
            "shuffle_buffer": int(args.shuffle_buffer),
            "seed": int(args.seed),
        }
        meta_path.write_text(json.dumps(payload, indent=2))

    try:
        for i, ex in enumerate(ds):
            if stop_now["flag"]:
                raise GracefulStop()

            if args.max_samples is not None and i >= args.max_samples:
                break

            meta = ex.get("meta", {}) or {}
            dom = meta.get("pile_set_name", None)
            if dom is None or dom not in domain_set:
                continue

            if domain_done(dom):
                if all(domain_done(d) for d in domains):
                    break
                continue

            text = ex.get("text", "")
            if not isinstance(text, str) or not text:
                continue

            # Write as JSONL: only {"text": "..."} per your encoder requirements
            rec = {"text": text}
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            writers[dom].write(line)

            n_chars = len(text)
            char_counts[dom] += n_chars
            line_counts[dom] += 1
            pbar.update(n_chars)

            # Periodically flush + meta snapshot for robust resume
            if line_counts[dom] % 2000 == 0:
                writers[dom].flush()
                write_meta(dom)

        pbar.close()

    except GracefulStop:
        pbar.close()
        print("\nInterrupted. Flushing partial outputs...")

    finally:
        for dom in domains:
            try:
                writers[dom].flush()
                writers[dom].close()
            except Exception:
                pass
            write_meta(dom)

        print("\nSummary:")
        for dom in domains:
            print(f"- {dom:20s}  chars={char_counts[dom]:,}/{args.target_chars:,}  lines={line_counts[dom]:,}")


if __name__ == "__main__":
    main()
