"""Provenance stamping for result files.

The problem this solves
-----------------------
None of the result files committed before v4 can be tied to the code or the
environment that produced them:

* The repository is a single squashed commit, so there is no history linking a
  number to a revision.
* `requirements.txt` pinned nothing but `torch>=2.1`, and no lockfile or
  `pip freeze` was recorded, so the resolved versions are unknown.
* At least one committed result file does not match the schema written by the
  committed script that supposedly produced it (docs/ERRATA.md, E11), and there
  is no way to tell which script actually did.

Every result written from now on carries a `provenance` block: the git revision
and whether the tree was dirty, the exact command line, the resolved versions of
the packages that affect numerics, the platform, and the seeds. That is the
minimum needed for someone else to reproduce a number, and for us to tell later
which code produced it.

Usage
-----
    from provenance import stamp, write_json
    write_json(path, {"metrics": ...}, args=args)      # adds "provenance"
"""
import json
import os
import platform
import subprocess
import sys
import time

# Packages whose version can change a number. Recorded when importable.
_NUMERIC_PACKAGES = ("torch", "numpy", "scipy", "pandas", "sklearn", "xgboost",
                     "anndata", "h5py")


def _git(*args, repo_root=None):
    try:
        out = subprocess.run(["git", *args], cwd=repo_root or _repo_root(),
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_state():
    """Revision, branch and dirtiness. A dirty tree makes a result unreproducible."""
    rev = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return dict(
        revision=rev,
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        dirty=bool(status) if status is not None else None,
        n_modified_files=(len([l for l in status.splitlines() if l.strip()])
                          if status else 0),
    )


#  distribution name != import name for some of these
_DIST_NAME = {"sklearn": "scikit-learn"}


def package_versions(names=_NUMERIC_PACKAGES):
    """Resolved versions, read from installed metadata rather than `__version__`.

    Several of these packages have deprecated their `__version__` attribute, and
    metadata is the authoritative record of what is installed anyway.
    """
    from importlib import metadata, util
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(_DIST_NAME.get(n, n))
        except metadata.PackageNotFoundError:
            # importable but not installed as a distribution (vendored, editable)
            out[n] = "present-no-metadata" if util.find_spec(n) else None
        except Exception:
            out[n] = None
    return out


def stamp(args=None, extra=None):
    """Build the provenance block.

    `args` may be an argparse.Namespace; its fields are recorded verbatim, which
    captures the split, the seeds and every protocol switch in one place.
    """
    g = git_state()
    prov = dict(
        recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git=g,
        command=" ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        argv=sys.argv,
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages=package_versions(),
    )
    if args is not None:
        prov["args"] = {k: _jsonable(v) for k, v in vars(args).items()}
    if extra:
        prov.update(extra)
    if g.get("dirty"):
        prov["warning"] = (
            "the working tree was dirty when this result was produced, so the "
            "recorded revision does not fully describe the code that ran")
    return prov


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def write_json(path, payload, args=None, extra=None, indent=2):
    """Write `payload` with a `provenance` block attached, creating parent dirs."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict so that provenance can be added")
    out = dict(payload)
    out["provenance"] = stamp(args=args, extra=extra)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=indent, default=str)
    return path
