"""Extract embedded notebook figures into an Overleaf-friendly folder.

This script reads executed Jupyter notebooks and writes embedded image outputs
and markdown attachments to:

    latex_build/figures/notebook_outputs/

It also creates:

    latex_build/figures/notebook_outputs/figure_manifest.csv
    latex_build/figures/notebook_outputs/figure_manifest.md
    latex_build/figure_snippets.tex

The script does not modify notebooks and does not delete existing files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from pathlib import Path


NOTEBOOKS = [
    ("phase1_forensic", "phase1_ai_shield_oop_colab (1).ipynb"),
    ("phase2_semantic", "phase2_semantic_layer_vit_attention_colab.ipynb"),
    ("phase3_fusion", "phase3_forensic_semantic_glm_fusion_colab.ipynb"),
]

MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
}


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "figure"


def decode_image_payload(payload: str | list[str]) -> bytes:
    if isinstance(payload, list):
        payload = "".join(payload)
    return base64.b64decode(payload)


def write_image_if_needed(path: Path, content: bytes) -> Path:
    """Write image bytes without overwriting different existing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
        return path

    if path.read_bytes() == content:
        return path

    digest = hashlib.sha256(content).hexdigest()[:10]
    alternate = path.with_name(f"{path.stem}_{digest}{path.suffix}")
    if not alternate.exists():
        alternate.write_bytes(content)
    return alternate


def markdown_first_line(cell: dict) -> str:
    source = "".join(cell.get("source", [])).strip()
    if not source:
        return ""
    first = source.splitlines()[0]
    return re.sub(r"^#+\s*", "", first).strip()


def extract_notebook(slug: str, notebook_path: Path, output_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not notebook_path.exists():
        print(f"Skipping missing notebook: {notebook_path}")
        return rows

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_output_dir = output_root / slug

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type", "")
        heading = markdown_first_line(cell)

        if cell_type == "code":
            for output_index, output in enumerate(cell.get("outputs", [])):
                for mime, extension in MIME_EXT.items():
                    payload = output.get("data", {}).get(mime)
                    if not payload:
                        continue
                    content = decode_image_payload(payload)
                    filename = f"{slug}_cell{cell_index:03d}_output{output_index:02d}{extension}"
                    path = write_image_if_needed(notebook_output_dir / filename, content)
                    rows.append(
                        {
                            "notebook": notebook_path.name,
                            "slug": slug,
                            "kind": "code_output",
                            "cell_index": str(cell_index),
                            "output_index": str(output_index),
                            "mime": mime,
                            "filename": path.name,
                            "relative_path_from_latex_build": str(path.relative_to(output_root.parent.parent)).replace("\\", "/"),
                            "size_bytes": str(len(content)),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "caption_hint": f"{slug} code cell {cell_index} output {output_index}",
                        }
                    )

        attachments = cell.get("attachments", {})
        for attachment_name, attachment_data in attachments.items():
            for mime, extension in MIME_EXT.items():
                payload = attachment_data.get(mime)
                if not payload:
                    continue
                content = decode_image_payload(payload)
                stem = safe_stem(Path(attachment_name).stem)
                filename = f"{slug}_cell{cell_index:03d}_attachment_{stem}{extension}"
                path = write_image_if_needed(notebook_output_dir / filename, content)
                rows.append(
                    {
                        "notebook": notebook_path.name,
                        "slug": slug,
                        "kind": "markdown_attachment",
                        "cell_index": str(cell_index),
                        "output_index": "",
                        "mime": mime,
                        "filename": path.name,
                        "relative_path_from_latex_build": str(path.relative_to(output_root.parent.parent)).replace("\\", "/"),
                        "size_bytes": str(len(content)),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "caption_hint": heading or stem,
                    }
                )

    return rows


def write_manifest(rows: list[dict[str, str]], output_root: Path, latex_build_root: Path) -> None:
    fieldnames = [
        "notebook",
        "slug",
        "kind",
        "cell_index",
        "output_index",
        "mime",
        "filename",
        "relative_path_from_latex_build",
        "size_bytes",
        "sha256",
        "caption_hint",
    ]

    manifest_csv = output_root / "figure_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_md = output_root / "figure_manifest.md"
    lines = [
        "# Extracted Notebook Figures",
        "",
        "These files were extracted from executed notebook outputs and markdown attachments.",
        "Paths below are relative to `latex_build/`, so they can be used directly in Overleaf after uploading the `figures/` folder.",
        "",
        "| Notebook | Kind | Cell | File | Caption hint |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['slug']} | {row['kind']} | {row['cell_index']} | "
            f"`{row['relative_path_from_latex_build']}` | {row['caption_hint']} |"
        )
    manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    snippets = [
        "% Auto-generated LaTeX figure snippets.",
        "% Paths are relative to latex_build/main.tex.",
        "% Edit captions/labels before final submission.",
        "",
    ]
    for row in rows:
        label = safe_stem(f"fig:{row['slug']}:cell{row['cell_index']}:{row['filename']}")
        snippets.extend(
            [
                "\\begin{figure}[htbp]",
                "    \\centering",
                f"    \\includegraphics[width=0.92\\linewidth]{{{row['relative_path_from_latex_build']}}}",
                f"    \\caption{{{row['caption_hint']}}}",
                f"    \\label{{{label}}}",
                "\\end{figure}",
                "",
            ]
        )
    (latex_build_root / "figure_snippets.tex").write_text("\n".join(snippets), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex-build-root", default="latex_build")
    parser.add_argument("--output-subdir", default="figures/notebook_outputs")
    args = parser.parse_args()

    repo_root = Path.cwd()
    latex_build_root = repo_root / args.latex_build_root
    output_root = latex_build_root / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for slug, notebook_name in NOTEBOOKS:
        rows.extend(extract_notebook(slug, repo_root / notebook_name, output_root))

    rows.sort(key=lambda row: (row["slug"], int(row["cell_index"]), row["kind"], row["filename"]))
    write_manifest(rows, output_root, latex_build_root)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["slug"]] = counts.get(row["slug"], 0) + 1

    print(f"Extracted {len(rows)} figures to {output_root}")
    for slug, count in counts.items():
        print(f"  {slug}: {count}")
    print(f"Wrote manifest: {output_root / 'figure_manifest.csv'}")
    print(f"Wrote LaTeX snippets: {latex_build_root / 'figure_snippets.tex'}")


if __name__ == "__main__":
    main()
