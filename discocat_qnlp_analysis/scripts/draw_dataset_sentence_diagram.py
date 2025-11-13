#!/usr/bin/env python3
"""
Extract a simple subject-verb-object triple from the dataset and draw a DisCoCat diagram.

The script heuristically picks the first three English tokens (ASCII words) it finds in the
selected text. For better control, pass `--words` with a comma-separated list of three tokens.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from discopy.grammar.pregroup import Ty, Word, Cup, Diagram  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "discopy with grammar.pregroup is required. Install/update with `pip install --upgrade discopy`."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH_DEFAULT = REPO_ROOT / "data" / "dataseet.xlsx"
OUTPUT_DEFAULT = REPO_ROOT / "20251112_collapose" / "string_diagram_dataset.png"


def extract_words(text: str, override: str | None = None) -> tuple[str, str, str]:
    """
    Return a (subject, verb, object) triple either from an override string or by heuristics.

    Override must be a comma-separated list of three tokens.
    """
    if override:
        parts = [token.strip() for token in override.split(",") if token.strip()]
        if len(parts) != 3:
            raise ValueError("Override must contain exactly three comma-separated tokens.")
        return parts[0], parts[1], parts[2]

    tokens = re.findall(r"[A-Za-z']+", text)
    if len(tokens) < 3:
        raise ValueError(
            "Could not find at least three ASCII tokens in the selected text. "
            "Pass --words to specify the tokens manually."
        )

    subject, verb, *rest = tokens
    obj = " ".join(rest) if rest else "object"
    return subject, verb, obj


def build_diagram(subject: str, verb: str, obj: str) -> Diagram:
    """Build a simple SVO DisCoCat diagram."""
    noun, sentence = Ty("n"), Ty("s")
    subj_word = Word(subject, noun)
    verb_word = Word(verb, noun.r @ sentence @ noun.l)
    obj_word = Word(obj, noun)
    return subj_word @ verb_word @ obj_word >> Cup(noun, noun.r) @ sentence @ Cup(noun.l, noun)


def main() -> None:
    # Set a CJK-capable font if available to avoid missing glyph warnings.
    candidates = [
        "PingFang TC",
        "PingFang HK",
        "PingFang SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Heiti TC",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name]
            break
        except Exception:
            continue

    parser = argparse.ArgumentParser(description="Draw a dataset-based DisCoCat string diagram.")
    parser.add_argument("--row", type=int, default=0, help="Row index in the dataset (0-based).")
    parser.add_argument(
        "--column",
        type=str,
        default="新聞標題",
        help="Column name to extract text from (default: 新聞標題).",
    )
    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="Comma-separated subject,verb,object override (e.g., 'dogs,chase,cats').",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH_DEFAULT,
        help="Path to the dataset Excel file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Where to save the rendered diagram (PNG/PDF).",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise SystemExit(f"Dataset not found at {args.data_path}")

    df = pd.read_excel(args.data_path)
    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found. Available: {df.columns.tolist()}")
    if not (0 <= args.row < len(df)):
        raise SystemExit(f"Row index {args.row} out of range (dataset has {len(df)} rows).")

    text = str(df.iloc[args.row][args.column])
    subject, verb, obj = extract_words(text, args.words)

    diagram = build_diagram(subject, verb, obj)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagram.draw(path=str(output_path))

    print("Extracted tokens:", subject, verb, obj)
    print(f"Diagram saved to {output_path}")


if __name__ == "__main__":
    main()

