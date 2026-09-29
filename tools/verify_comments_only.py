#!/usr/bin/env python
"""Verify that a change touched only comments and docstrings, not code.

Comments and docstrings do not survive into the abstract syntax tree in any form
that affects execution, so comparing the ASTs of two revisions of a file -- with
docstrings stripped -- is a definitive check rather than a heuristic. Grepping the
diff is not: a line whose only change is a trailing comment still shows up as a
modified code line, and a genuine one-token edit inside a long line can hide.

Used when translating comments, where the whole point is that behaviour is
untouched.

Usage:
    python tools/verify_comments_only.py <git-ref> [paths...]
    python tools/verify_comments_only.py HEAD src/efficacy/predict_service.py
    python tools/verify_comments_only.py HEAD          # all tracked changed .py

Exit code 0 means every file's code is byte-identical in structure; non-zero
names the files that differ and where.
"""
import ast
import subprocess
import sys


def strip_docstrings(tree):
    """Remove docstring expression statements in place."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            # a lone docstring must be replaced, not deleted, or the body empties
            if len(body) == 1:
                node.body = [ast.Pass()]
            else:
                node.body = body[1:]
    return tree


def normalise(src):
    """AST dump with docstrings and position information removed."""
    tree = strip_docstrings(ast.parse(src))
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def git_show(ref, path):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True,
                       text=True)
    if r.returncode != 0:
        return None                      # new file: nothing to compare against
    return r.stdout


def changed_py_files(ref):
    r = subprocess.run(["git", "diff", "--name-only", ref], capture_output=True,
                       text=True, check=True)
    return [p for p in r.stdout.split() if p.endswith(".py")]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ref = sys.argv[1]
    paths = sys.argv[2:] or changed_py_files(ref)
    if not paths:
        print("no changed .py files to check")
        return 0

    bad, skipped, ok = [], [], []
    for p in paths:
        before = git_show(ref, p)
        if before is None:
            skipped.append(p)
            continue
        try:
            with open(p) as f:
                after = f.read()
        except FileNotFoundError:
            bad.append((p, "file deleted"))
            continue
        try:
            a, b = normalise(before), normalise(after)
        except SyntaxError as e:
            bad.append((p, f"does not parse: {e}"))
            continue
        if a == b:
            ok.append(p)
        else:
            # locate the first divergence to make the failure actionable
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                     min(len(a), len(b)))
            bad.append((p, f"AST differs near offset {i}: "
                           f"...{a[max(0, i-90):i+90]!r} != ...{b[max(0, i-90):i+90]!r}"))

    for p in ok:
        print(f"OK        {p}  (comments/docstrings only)")
    for p in skipped:
        print(f"SKIP      {p}  (new file, nothing to compare)")
    for p, why in bad:
        print(f"CODE-DIFF {p}\n          {why}")
    print(f"\n{len(ok)} comment-only, {len(skipped)} new, {len(bad)} with code changes")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
