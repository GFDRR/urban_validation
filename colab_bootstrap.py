"""
Colab bootstrap: code from GitHub, data from Drive.

Why this exists
---------------
Every notebook used to do::

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.validator import UrbanValidator   # <-- imported src from DRIVE

PROJECT_ROOT points at Google Drive, so ``src`` resolved to a hand-copied
snapshot on Drive rather than the committed code. That copy silently drifted
weeks behind the repo (missing the SpaceNet IoU split, ``total_area_bias``,
``mae_area_m2`` ...), so notebooks opened fresh from GitHub still executed
stale library code.

This module clones the repo into the Colab VM and puts *that* on ``sys.path``,
while leaving the working directory on Drive (the configs resolve
``aoi_tracker`` and ``data_dir`` relative to the CWD, so it must stay there).

Usage in a notebook::

    !wget -q -O /content/colab_bootstrap.py \
      https://raw.githubusercontent.com/GFDRR/urban_validation/fix/pipeline-audit/colab_bootstrap.py
    import sys; sys.path.insert(0, '/content')
    from colab_bootstrap import setup
    setup(PROJECT_ROOT)          # after PROJECT_ROOT is defined

``setup()`` is idempotent — safe to re-run in a warm runtime.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/GFDRR/urban_validation.git"
REPO_DIR = Path("/content/repo")
DEFAULT_BRANCH = "fix/pipeline-audit"


def _run(cmd: list[str]) -> str:
    """Run a subprocess and return its stdout, raising on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{res.stderr.strip()}")
    return res.stdout.strip()


def setup(
    project_root,
    branch: str = DEFAULT_BRANCH,
    repo_dir: Path = REPO_DIR,
    chdir: bool = True,
) -> Path:
    """Clone/refresh the repo, put it on sys.path, keep the CWD on Drive.

    Parameters
    ----------
    project_root : str | Path
        The Drive project root (data, configs, outputs live here).
    branch : str
        Branch to check out. Pin this deliberately — it decides which code runs.
    chdir : bool
        Keep the working directory on ``project_root``. Required: the configs
        resolve ``aoi_tracker`` / ``data_dir`` relative to the CWD.

    Returns
    -------
    Path to the cloned repo.
    """
    import os

    project_root = Path(project_root)
    if not project_root.exists():
        raise FileNotFoundError(
            f"PROJECT_ROOT does not exist: {project_root}\n"
            "Is Drive mounted, and is the path correct?"
        )

    repo_dir = Path(repo_dir)
    if (repo_dir / ".git").is_dir():
        _run(["git", "-C", str(repo_dir), "fetch", "--quiet", "origin", branch])
        _run(["git", "-C", str(repo_dir), "checkout", "--quiet", "-B", branch,
              f"origin/{branch}"])
        _run(["git", "-C", str(repo_dir), "reset", "--quiet", "--hard",
              f"origin/{branch}"])
    else:
        _run(["git", "clone", "--quiet", "--branch", branch, REPO_URL, str(repo_dir)])

    # The repo must win over any leftover Drive copy of src/.
    drop = {str(project_root), str(repo_dir), ""}
    sys.path = [p for p in sys.path if p not in drop]
    sys.path.insert(0, str(repo_dir))

    # Data paths in the configs are relative to the CWD, so it stays on Drive.
    if chdir:
        os.chdir(project_root)

    for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[mod]

    import src  # noqa: F401  (import here so the print below proves the origin)

    sha = _run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"])
    print(f"code : {repo_dir}  @ {branch} ({sha})")
    print(f"src  : {src.__file__}")
    print(f"data : {project_root}  (cwd)")
    if str(project_root) in (src.__file__ or ""):
        raise RuntimeError(
            "src is still being imported from Drive — the stale copy is shadowing "
            "the repo. Delete/rename PROJECT_ROOT/src and re-run this cell."
        )
    return repo_dir
