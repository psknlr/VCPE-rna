"""Every file a figure reads must be in the repository.

`results/real_*/` is ignored by design (live run directories), and the final
outputs a figure depends on are committed deliberately with `git add -f`. When one
is missed, the figures still build on the machine that ran the experiments and
nowhere else -- which is how the per-seed reports and training logs behind
Figs 1, 4 and 5 and Extended Data Figs 2-4 went uncommitted until CI, running on a
fresh clone, failed on them. This test fails first, and names the file.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "figures"))


def test_every_figure_input_is_tracked(monkeypatch):
    import figdata as F
    read = set()
    real_open = open

    def recording_open(path, *a, **k):
        p = Path(path).resolve()
        if ROOT in p.parents:
            read.add(p.relative_to(ROOT).as_posix())
        return real_open(path, *a, **k)

    monkeypatch.setattr(F, "open", recording_open, raising=False)
    F.collect_ed()
    assert read, "figdata read nothing; the recorder is not wired in"
    tracked = set(subprocess.run(["git", "ls-files", "--", *sorted(read)],
                                 cwd=ROOT, capture_output=True, text=True,
                                 check=True).stdout.split())
    missing = sorted(read - tracked)
    assert not missing, (
        f"figures read {len(missing)} file(s) that are not committed, so they "
        f"cannot be rebuilt from a clone: {missing}. Commit them with `git add -f`.")
