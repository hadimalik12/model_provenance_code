"""Verify the paper fragment against pinned result snapshots (standard library only)."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RUNS = [f"run_{i:02}" for i in range(3)]
STEPS = [512, 1024, 2048, 3125, 6250, 12500]
M = 8000
GAMMA = math.sqrt(2 * math.log(2 / 0.05) / M)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gap(row, prefix=""):
    return Decimal(row[prefix + "a"]) - Decimal(row[prefix + "a_prime"])


def check_metric(row, prefix=""):
    delta = gap(row, prefix)
    a, ap = (Decimal(row[prefix + key]) for key in ("a", "a_prime"))
    require(0 <= a <= 1 and 0 <= ap <= 1, "Invalid acceptance rates")
    require(a * M == int(a * M) and ap * M == int(ap * M), "Nonintegral counts")
    require(int(row[prefix + "audit_pairs"]) == M, "Unexpected audit size")
    require(math.isclose(float(delta), float(row[prefix + "signed_gap"]),
                         abs_tol=1e-12), "Signed gap disagrees with acceptance rates")
    require(math.isclose(abs(float(delta)), float(row[prefix + "advantage"]),
                         abs_tol=1e-12), "Advantage disagrees with signed gap")
    # Independently recompute the two-sided sign-swap p-value from saved counts.
    discordant = M - int(row[prefix + "decision_ties"])
    difference = int(delta * M)
    require((discordant + difference) % 2 == 0, "Inconsistent pair counts")
    positive = (discordant + difference) // 2
    negative = discordant - positive
    require(min(positive, negative) >= 0, "Invalid discordant-pair counts")
    if prefix:
        for key, value in (("discordant_pairs", discordant),
                           ("positive_pairs", positive), ("negative_pairs", negative)):
            require(int(row[prefix + key]) == value, "Pair counts disagree")
    numerator = 2 * sum(math.comb(discordant, k)
                        for k in range(min(positive, negative) + 1))
    log_p = min(0.0, math.log10(numerator) - discordant * math.log10(2))
    require(math.isclose(log_p, float(row[prefix + "log10_p_shard"]),
                         abs_tol=1e-8), "Paired log-p disagrees with saved counts")
    require((abs(float(delta)) > GAMMA) == (log_p < math.log10(0.05)),
            "Paired threshold and exact-test verdicts differ")


def number(value, digits):
    return format(value.quantize(Decimal(10) ** -digits, rounding=ROUND_HALF_UP), "f")


def cells(row, prefix=""):
    delta = gap(row, prefix)
    return [number(50 * (1 + delta), 2) + r"\%", "$" + number(delta, 4) + "$"]


def table_row(run, exposure, control, target, prefix=""):
    require(abs(float(gap(control, prefix))) <= GAMMA, "Unexpected control rejection")
    require(abs(float(gap(target, prefix))) > GAMMA, "Unexpected target nonrejection")
    return " & ".join([run[-2:], exposure, *cells(control, prefix),
                       *cells(target, prefix), "0.0304", r"Reject $H_0$ / Yes"]) + r" \\"


def load_and_verify():
    manifest = json.loads((ROOT / "data/source_manifest.json").read_text())
    for name, record in manifest["files"].items():
        require(hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
                == record["sha256"], f"Source snapshot changed: {name}")
    snapshot = json.loads((ROOT / "data/ultrachat_snapshot.json").read_text())
    require(snapshot["matched_steps"] == STEPS, "Unexpected matched coverage")
    require(snapshot["completed_checkpoint_audits"] == 36
            and snapshot["planned_checkpoint_audits"] == 42
            and snapshot["all_scheduled_audits_complete"] is False,
            "Snapshot must remain partial")
    require(hashlib.sha256((ROOT / "data/ultrachat.csv").read_bytes()).hexdigest()
            == snapshot["files_sha256"]["curve.csv"], "Upstream snapshot hash differs")
    with (ROOT / "data/dolly.csv").open(newline="") as stream:
        dolly = list(csv.DictReader(stream))
    with (ROOT / "data/ultrachat.csv").open(newline="") as stream:
        ultra = list(csv.DictReader(stream))
    d = {(r["run"], r["checkpoint"]): r for r in dolly}
    u = {(r["run_id"], r["stage"], int(r["steps"])): r for r in ultra}
    require(len(dolly) == len(d) == 18 and set(d) ==
            {(run, cp) for run in RUNS for cp in ("B", "P", "T1", "T2", "T", "T0")},
            "Dolly must contain the six original checkpoints per run")
    require(len(ultra) == len(u) == 42 and set(u) ==
            {(run, stage, step) for run in RUNS for stage in ("target", "control")
             for step in [0, *STEPS]}, "Unexpected UltraChat rows or missing matched pairs")
    for row in dolly:
        require(row["score"] == "watermark", "Wrong detector")
        check_metric(row)
    for row in ultra:
        check_metric(row, "watermark_")
    dolly_lines, ultra_lines = [], []
    for run in RUNS:
        if dolly_lines:
            dolly_lines.append(r"\midrule")
            ultra_lines.append(r"\midrule")
        for epoch, checkpoint in enumerate(("T1", "T2", "T"), 1):
            dolly_lines.append(table_row(run, str(epoch), d[run, "T0"], d[run, checkpoint]))
        for step in [0, *STEPS]:
            target, control = u[run, "target", step], u[run, "control", step]
            for field in ("examples_seen", "unique_examples_seen", "response_tokens_seen",
                          "input_tokens_seen"):
                require(target[field] == control[field], "Mismatched target/control exposure")
            require(int(target["examples_seen"]) == int(target["unique_examples_seen"])
                    == step * 32, "Repeated or incorrect example exposure")
            for stage, row in (("target", target), ("control", control)):
                expected = ("P" if stage == "target" else "B") if step == 0 else f"{stage}_step{step}"
                require(row["checkpoint"] == expected, "Wrong checkpoint identity")
                if step == 0:
                    require(gap(row, "watermark_") == gap(d[run, expected]),
                            "UltraChat baseline differs from original parent/base")
            exposure = f"{step * 32:,}".replace(",", r"\, ").replace(" ", "")
            ultra_lines.append(table_row(run, exposure, control, target, "watermark_"))
    return {"DOLLY": "\n".join(dolly_lines), "ULTRACHAT": "\n".join(ultra_lines)}


def preview(section):
    # A self-contained check, not a replacement for the unavailable manuscript class.
    return r"""\documentclass[10pt,twocolumn]{article}
\usepackage[letterpaper,margin=0.75in]{geometry}
\usepackage{amsmath,booktabs,graphicx}
\usepackage[numbers]{natbib}
\usepackage{hyperref}
\usepackage{cleveref}
\newcommand{\myparatight}[1]{\par\smallskip\noindent\textbf{#1.}\ }
\setcounter{secnumdepth}{3}
\begin{document}
\setcounter{section}{5}
\setcounter{subsection}{3}
\setcounter{table}{3}
""" + section + r"""
\clearpage
\begin{thebibliography}{24}
\bibitem[Sander et~al.(2024)]{sander2024watermarking}
Tom Sander, Pierre Fernandez, Alain Durmus, Matthijs Douze, and Teddy Furon.
\newblock Watermarking makes language models radioactive.
\newblock \emph{Advances in Neural Information Processing Systems},
37:21079--21113, 2024.
\end{thebibliography}
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-tables", action="store_true",
                        help="Regenerate only the two marked table bodies")
    parser.add_argument("--preview", type=Path, help="Write a standalone LaTeX preview")
    args = parser.parse_args()
    rows = load_and_verify()
    path = ROOT / "section.tex"
    section = path.read_text()
    for name, body in rows.items():
        pattern = rf"(% BEGIN {name} ROWS\n)(.*?)(\n% END {name} ROWS)"
        match = re.search(pattern, section, flags=re.S)
        require(match is not None, f"Missing {name} table markers")
        if args.write_tables:
            section = section[:match.start(2)] + body + section[match.end(2):]
        else:
            require(match.group(2) == body, f"{name} table differs from source data")
    labels = re.findall(r"\\label\{([^}]+)\}", section)
    require(len(labels) == len(set(labels)), "Duplicate labels")
    for refs in re.findall(r"\\Cref\{([^}]+)\}", section):
        require(all(ref in labels for ref in refs.split(",")), "Unresolved table reference")
    require(set(re.findall(r"\\citep\{([^}]+)\}", section)) ==
            {"sander2024watermarking"}, "Unexpected bibliography dependency")
    if args.write_tables:
        path.write_text(section)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        args.preview.write_text(preview(section))
    print(f"Verified 60 source measurements, 9 Dolly rows, and 21 UltraChat rows; "
          f"paired threshold = {GAMMA:.10f}. All signed values and paired verdicts agree.")


if __name__ == "__main__":
    main()
