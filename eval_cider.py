"""Evaluate eval_attack-style JSON outputs with CIDEr and CIDEr-D."""

import argparse
import contextlib
import importlib
import importlib.resources as importlib_resources_std
import io
import json
import os
import re
import sys
import types
from pathlib import Path


def _install_local_cider_shims() -> None:
    """Make the vendored cider package importable in lightweight environments."""
    sys.modules.setdefault("importlib_resources", importlib_resources_std)

    try:
        importlib.import_module("spacy")
        return
    except ModuleNotFoundError:
        pass

    spacy = types.ModuleType("spacy")
    spacy.load = lambda _name: None
    lang = types.ModuleType("spacy.lang")
    char_classes = types.ModuleType("spacy.lang.char_classes")
    char_classes.ALPHA = "A-Za-z"
    char_classes.ALPHA_LOWER = "a-z"
    char_classes.ALPHA_UPPER = "A-Z"
    char_classes.CONCAT_QUOTES = "'\""
    char_classes.LIST_ELLIPSES = []
    char_classes.LIST_ICONS = []
    util = types.ModuleType("spacy.util")
    util.compile_infix_regex = lambda infixes: re.compile("|".join(infixes) if infixes else r"$^")
    sys.modules.setdefault("spacy", spacy)
    sys.modules.setdefault("spacy.lang", lang)
    sys.modules.setdefault("spacy.lang.char_classes", char_classes)
    sys.modules.setdefault("spacy.util", util)


def _load_cider_functions():
    repo_root = Path(__file__).resolve().parent
    cider_root = repo_root / "cider"
    if str(cider_root) not in sys.path:
        sys.path.insert(0, str(cider_root))
    _install_local_cider_shims()
    from cidereval import cider, ciderD
    return cider, ciderD


def _normalize_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _collect_pairs(rows, field: str, limit: int | None):
    pairs = []
    skipped = 0
    ignored_after_limit = 0
    for row in rows:
        gt = _normalize_text(row.get("gt"))
        pred = _normalize_text(row.get(field))
        if gt is None or pred is None:
            skipped += 1
            continue
        if limit is not None and len(pairs) >= limit:
            ignored_after_limit += 1
            continue
        pairs.append((pred, [gt]))
    return pairs, skipped, ignored_after_limit


def _evaluate_field(rows, field: str, df_mode: str, limit: int | None, cider_fn, ciderD_fn):
    pairs, skipped, ignored_after_limit = _collect_pairs(rows, field, limit)
    if not pairs:
        return {
            "n": 0,
            "skipped": skipped,
            "ignored_after_limit": ignored_after_limit,
            "CIDEr": None,
            "CIDErD": None,
        }

    predictions = [pred for pred, _ in pairs]
    references = [refs for _, refs in pairs]
    with contextlib.redirect_stdout(io.StringIO()):
        cider_result = cider_fn(predictions=predictions, references=references, df=df_mode)
        ciderD_result = ciderD_fn(predictions=predictions, references=references, df=df_mode)
    return {
        "n": len(pairs),
        "skipped": skipped,
        "ignored_after_limit": ignored_after_limit,
        "CIDEr": float(cider_result["avg_score"]),
        "CIDErD": float(ciderD_result["avg_score"]),
    }


def _default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}.cider{ext or '.json'}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/eval_attack.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fields", type=str, nargs="+", default=["original", "randnoise", "pgd"])
    parser.add_argument("--df", type=str, default="corpus", choices=["corpus", "coco-val"])
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N valid rows per field.")
    return parser.parse_args()


def main():
    args = parse_args()
    cider_fn, ciderD_fn = _load_cider_functions()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = json.load(f)

    summary = {
        "input": args.input,
        "df": args.df,
        "total_rows": len(rows),
        "limit": args.limit,
        "fields": {},
    }

    print("=" * 58)
    print(f"Input : {args.input}")
    print(f"Rows  : {len(rows)}")
    print(f"DF    : {args.df}")
    print("-" * 58)
    print(f"{'Field':<12} {'CIDEr':>12} {'CIDEr-D':>12} {'n':>8} {'skip':>8}")
    print("-" * 58)

    for field in args.fields:
        result = _evaluate_field(rows, field, args.df, args.limit, cider_fn, ciderD_fn)
        summary["fields"][field] = result
        cider_text = "N/A" if result["CIDEr"] is None else f"{result['CIDEr']:.4f}"
        ciderD_text = "N/A" if result["CIDErD"] is None else f"{result['CIDErD']:.4f}"
        print(f"{field:<12} {cider_text:>12} {ciderD_text:>12} {result['n']:>8d} {result['skipped']:>8d}")

    print("=" * 58)

    output_path = args.output or _default_output_path(args.input)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()