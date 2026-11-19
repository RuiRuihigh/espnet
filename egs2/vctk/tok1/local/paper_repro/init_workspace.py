#!/usr/bin/env python3
"""Render cluster-specific launchers in a fresh workspace. Never submit jobs."""
import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = (
        "root", "base", "vctk", "libri", "conda-sh", "cpu-account",
        "gpu-account", "cpu-partition", "gpu-partitions", "preflight-partition",
    )
    for name in names:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    values = vars(args)
    for key, value in values.items():
        if not re.fullmatch(r"[A-Za-z0-9_./,:+\-]+", value):
            parser.error(f"{key}: spaces and shell metacharacters are not supported")
    for key in ("root", "base", "vctk", "libri", "conda_sh"):
        if not Path(values[key]).is_absolute():
            parser.error(f"{key} must be absolute")
    root = Path(args.root)
    if root.exists() and any(root.iterdir()):
        parser.error("root must be absent or empty; refusing to overwrite an experiment")
    package = Path(__file__).resolve().parent
    repo = package.parents[4]
    if not (repo / "espnet2/tasks/gan_tok.py").is_file():
        parser.error("run this helper inside the ESPnet checkout")
    rendered = {}
    for source in sorted((package / "templates").iterdir()):
        content = source.read_text()
        for key, value in values.items():
            content = content.replace("@" + key.upper() + "@", value)
        if re.search(r"@[A-Z_]+@", content):
            parser.error(f"unresolved placeholder in {source.name}")
        if source.suffix == ".py":
            ast.parse(content, filename=source.name)
        elif source.suffix == ".sh":
            subprocess.run(["bash", "-n"], input=content, text=True, check=True)
        rendered[source.name] = content
    root.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        target = root / name
        target.write_text(content)
        if target.suffix == ".sh":
            target.chmod(0o755)
    (root / "espnet").symlink_to(repo, target_is_directory=True)
    (root / "logs").mkdir()
    (root / "PORTING.json").write_text(json.dumps({
        "settings": values,
        "repository": str(repo),
        "preflight_auto_submit": False,
        "jobs_submitted": False,
        "note": "Use a dedicated checkout per workspace: configure.py writes recipe-local paths.",
    }, indent=2) + "\n")
    print(f"Created {root}. No environments installed and no jobs submitted.")


if __name__ == "__main__":
    main()
