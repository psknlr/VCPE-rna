"""Tests for result-file provenance stamping.

These matter because the repository's pre-v4 result files cannot be tied to any
revision or environment: the history is a single squashed commit, nothing but
torch was version-pinned, and at least one committed JSON does not match the
schema of the script said to have produced it (docs/ERRATA.md, E11). Every
result written from now on must carry enough to reproduce it.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance import package_versions, stamp, write_json  # noqa: E402


def test_stamp_records_the_essentials():
    p = stamp()
    for key in ("recorded_at", "git", "command", "argv", "python", "platform",
                "packages"):
        assert key in p, f"missing {key}"


def test_git_revision_is_recorded():
    g = stamp()["git"]
    assert g["revision"] is None or len(g["revision"]) == 40
    assert "dirty" in g


def test_args_are_flattened_into_the_stamp():
    """The protocol switches that decide a number must travel with it."""
    args = Namespace(split_by="target_gene", seed=7, inner_val_frac=0.15,
                     data_dirs=["a.h5ad", "b.h5ad"], use_region=True)
    p = stamp(args=args)
    assert p["args"]["split_by"] == "target_gene"
    assert p["args"]["seed"] == 7
    assert p["args"]["data_dirs"] == ["a.h5ad", "b.h5ad"]


def test_non_serialisable_arg_values_are_stringified_not_dropped():
    args = Namespace(device=object(), path=Path("/tmp/x"))
    p = stamp(args=args)
    assert isinstance(p["args"]["device"], str)
    assert p["args"]["path"] == "/tmp/x"
    json.dumps(p)                      # must round-trip


def test_package_versions_reports_absent_packages_as_none():
    v = package_versions(names=("numpy", "definitely_not_a_real_package_xyz"))
    assert v["numpy"] is not None
    assert v["definitely_not_a_real_package_xyz"] is None


def test_package_versions_does_not_use_deprecated_dunder_version():
    """anndata (among others) deprecated `__version__`; metadata is authoritative.

    Reading versions must not emit a FutureWarning, because a warning on every
    result write trains people to ignore warnings.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        v = package_versions(names=("numpy", "scipy", "anndata"))
    assert isinstance(v["numpy"], str) and isinstance(v["scipy"], str)


def test_write_json_attaches_provenance_and_preserves_payload(tmp_path):
    out = tmp_path / "nested" / "result.json"
    write_json(str(out), {"pearson_dev": 0.42, "baselines": {"zero": 0.0}},
               args=Namespace(seed=1))
    d = json.loads(out.read_text())
    assert d["pearson_dev"] == 0.42
    assert d["baselines"]["zero"] == 0.0
    assert d["provenance"]["args"]["seed"] == 1


def test_write_json_refuses_a_non_dict_payload(tmp_path):
    """A list payload has nowhere to put provenance, so this must be loud."""
    with pytest.raises(TypeError):
        write_json(str(tmp_path / "x.json"), [1, 2, 3])


def test_dirty_tree_is_flagged(monkeypatch):
    """A dirty tree means the recorded revision does not describe the code."""
    import provenance
    monkeypatch.setattr(provenance, "git_state",
                        lambda: {"revision": "a" * 40, "branch": "b",
                                 "dirty": True, "n_modified_files": 3})
    assert "warning" in provenance.stamp()


def test_clean_tree_is_not_flagged(monkeypatch):
    import provenance
    monkeypatch.setattr(provenance, "git_state",
                        lambda: {"revision": "a" * 40, "branch": "b",
                                 "dirty": False, "n_modified_files": 0})
    assert "warning" not in provenance.stamp()
