#!/usr/bin/env python3
"""Regenerate the AUTO sections of the repo map (CLAUDE.md and docs/map/*.md).

    python tools/gen_repo_map.py            # rewrite AUTO sections, warn if diagrams may be stale
    python tools/gen_repo_map.py --check    # report staleness only, write nothing
    python tools/gen_repo_map.py --stamp-diagrams   # after re-checking the diagrams, record their sources' hash

Standard library only. Read-only git (ls-files, log, diff, rev-parse, status, all with GIT_OPTIONAL_LOCKS=0 so
no index refresh is written). Never imports or runs project code: every metric is static (ast + git history).
Hand-written text outside <!-- AUTO:START name --> ... <!-- AUTO:END name --> markers is never touched.
"""
import argparse
import ast
import datetime
import hashlib
import io
import os
import re
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVED = ("experiments/phase1/",)  # archived one-off scripts: listed, not measured
IGNORED_DIRS = {".git", "venv", ".venv", "node_modules", "build", "dist", "__pycache__", ".ipynb_checkpoints",
                "dataset", "models", "results", "temp", "Claude outputs"}
IGNORED_SUFFIXES = (".egg-info",)
MAP_FILES = ["CLAUDE.md", "docs/map/directory.md", "docs/map/symbols.md", "docs/map/metrics.md"]
ARCH_FILES = ["docs/architecture/architecture.md", "docs/architecture/data-flow.md"]
AUTO_RE = re.compile(r"(<!-- AUTO:START (?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- AUTO:END (?P=name) -->)", re.S)


# ----------------------------------------------------------------------------------------------- git (read-only)
def git(*args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        return subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True, text=True, env=env,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def ignored(path):
    parts = path.split("/")
    return any(p in IGNORED_DIRS or p.endswith(IGNORED_SUFFIXES) for p in parts[:-1])


def repo_files():
    """Tracked files plus untracked-but-not-ignored ones, minus ignored directories."""
    files = set(git("ls-files").splitlines()) | set(git("ls-files", "--others", "--exclude-standard").splitlines())
    return sorted(f for f in files if f and not ignored(f) and os.path.isfile(os.path.join(ROOT, f)))


# ----------------------------------------------------------------------------------------------- static analysis
BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.Assert,
            ast.IfExp)


def complexity(fn):
    """Cyclomatic complexity (1 + decision points) and max nesting depth of a function."""
    cc = 1
    for node in ast.walk(fn):
        if isinstance(node, BRANCHES):
            cc += 1
        elif isinstance(node, ast.BoolOp):
            cc += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            cc += len(node.ifs)

    def depth(node, d=0):
        best = d
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            nd = d + 1 if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)) else d
            best = max(best, depth(child, nd))
        return best
    return cc, depth(fn)


def sloc(text):
    """Non-blank, non-comment lines (docstrings count as code)."""
    lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                                tokenize.ENDMARKER):
                lines.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError):
        return sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    return len(lines)


def analyse(py_files):
    mods = {}
    for f in py_files:
        text = open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        defs, funcs, imports, used = [], [], set(), Counter()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc, nest = complexity(node)
                funcs.append((node.name, node.lineno, cc, nest))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node in tree.body:
                defs.append((node.name, node.lineno, "class" if isinstance(node, ast.ClassDef) else "function"))
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("__"):
                        defs.append((node.name + "." + item.name, item.lineno, "method"))
            if isinstance(node, ast.Import):
                imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Name):
                used[node.id] += 1
            elif isinstance(node, ast.Attribute):
                used[node.attr] += 1
        mods[f] = {"name": name, "sloc": sloc(text), "defs": defs, "funcs": funcs, "imports": imports, "used": used,
                   "n_class": sum(1 for d in defs if d[2] == "class"), "n_func": len(funcs),
                   "is_test": f.startswith("tests/") or os.path.basename(f).startswith("test_")}
    return mods


def todos(files):
    out = []
    for f in files:
        if (not f.endswith((".py", ".sh", ".yml", ".txt")) or f.startswith(("docs/", "tools/")) or f.startswith(ARCHIVED)):
            continue
        for i, ln in enumerate(open(os.path.join(ROOT, f), encoding="utf-8", errors="replace"), 1):
            if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", ln):
                out.append((f, i, ln.strip()[:90]))
    return out


def churn():
    commits, lines = Counter(), Counter()
    log = git("log", "--since=12 months ago", "--numstat", "--format=tformat:@@")
    seen = set()
    for ln in log.splitlines():
        if ln == "@@":
            seen = set(); continue
        parts = ln.split("\t")
        if len(parts) == 3:
            a, d, f = parts
            if f not in seen:
                commits[f] += 1; seen.add(f)
            lines[f] += (int(a) if a.isdigit() else 0) + (int(d) if d.isdigit() else 0)
    return commits, lines


def pct_rank(values):
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    return {k: (i + 1) / n for i, (k, _) in enumerate(items)} if n else {}


def bar(x, top, width=10):
    k = 0 if top <= 0 else int(round(width * x / top))
    return "▮" * k + "▯" * (width - k)


# ----------------------------------------------------------------------------------------------- renderers
def tree_block(files):
    dirs = defaultdict(list)
    for f in files:
        d = os.path.dirname(f)
        if d.count("/") < 3:
            dirs[d].append(os.path.basename(f))
    out = ["```"]
    for d in sorted(dirs):
        names = sorted(dirs[d])
        shown = ", ".join(names[:8]) + (", … (+%d)" % (len(names) - 8) if len(names) > 8 else "")
        out.append("%s/  %s" % (d or ".", shown))
    out.append("```")
    out.append("Archived (listed, not measured): " + ", ".join("`%s`" % a for a in ARCHIVED) + ".")
    out.append("Ignored (not mapped): " + ", ".join("`%s/`" % d for d in sorted(IGNORED_DIRS)) + ", `*.egg-info`.")
    return "\n".join(out)


def symbols_block(mods, limit=50):
    total_used, n_defs = Counter(), Counter()
    for m in mods.values():
        total_used.update(m["used"])
        n_defs.update(d[0].split(".")[-1] for d in m["defs"])
    rows = []
    for f, m in mods.items():
        if m["is_test"] or f.startswith(("tools/", "experiments/")):
            continue
        for name, line, kind in m["defs"]:
            short = name.split(".")[-1]
            refs = total_used[short] // max(1, n_defs[short])  # same-named defs (main, parse_args) share the count
            rows.append((refs, name, f, line, kind))
    rows.sort(key=lambda r: (-r[0], r[2], r[3]))
    out = ["| symbol | file:line | kind | refs |", "|---|---|---|---|"]
    for refs, name, f, line, kind in rows[:limit]:
        out.append("| `%s` | `%s:%d` | %s | %d |" % (name, f, line, kind, refs))
    out.append("")
    out.append("Ranked by intra-repo reference count: ast `Name`/`Attribute` occurrences of the symbol name across "
               "all mapped Python files, divided by the number of same-named definitions (so `main`/`parse_args` in each "
               "entry script share one count). Experiment and tool scripts are excluded from the ranking.")
    return "\n".join(out)


def metrics_blocks(mods, files):
    src = {f: m for f, m in mods.items() if not m["is_test"] and not f.startswith(("tools/",) + ARCHIVED) and m["sloc"] > 0}
    tests = {f: m for f, m in mods.items() if m["is_test"]}
    names = {m["name"]: f for f, m in src.items()}
    commits, lines = churn()
    blocks = {}

    # size + complexity
    rows = []
    for f, m in sorted(src.items()):
        cc_sum = sum(fn[2] for fn in m["funcs"]); cc_max = max([fn[2] for fn in m["funcs"]] or [0])
        rows.append((f, m["sloc"], m["n_func"], m["n_class"], cc_sum, cc_max))
    top_sloc = max([r[1] for r in rows] or [1])
    out = ["| module | SLOC | size | functions | classes | complexity sum | max CC |", "|---|---|---|---|---|---|---|"]
    for f, s, nf, nc, cs, cm in rows:
        out.append("| `%s` | %d | %s | %d | %d | %d | %d |" % (f, s, bar(s, top_sloc), nf, nc, cs, cm))
    tot = sum(r[1] for r in rows)
    out.append("| **total (%d modules)** | **%d** | | **%d** | **%d** | **%d** | |" % (
        len(rows), tot, sum(r[2] for r in rows), sum(r[3] for r in rows), sum(r[4] for r in rows)))
    blocks["metrics-size"] = "\n".join(out)

    fns = sorted(((fn[2], fn[3], f, fn[0], fn[1]) for f, m in src.items() for fn in m["funcs"]), reverse=True)[:10]
    out = ["| function | file:line | cyclomatic complexity | max nesting |", "|---|---|---|---|"]
    for cc, nest, f, name, line in fns:
        out.append("| `%s` | `%s:%d` | %d %s | %d |" % (name, f, line, cc, bar(cc, fns[0][0]), nest))
    blocks["metrics-complexity"] = "\n".join(out)

    # static test mapping
    tested = defaultdict(lambda: {"files": 0, "funcs": 0, "sloc": 0})
    for tf, tm in tests.items():
        targets = {names[i] for i in tm["imports"] if i in names}
        stem = tm["name"][5:] if tm["name"].startswith("test_") else None
        if stem in names:
            targets.add(names[stem])
        n_tests = sum(1 for fn in tm["funcs"] if fn[0].startswith("test"))
        for t in targets:
            tested[t]["files"] += 1; tested[t]["funcs"] += n_tests; tested[t]["sloc"] += tm["sloc"]
    out = ["| module | test files | test functions | test SLOC : source SLOC |", "|---|---|---|---|"]
    for f, m in sorted(src.items()):
        t = tested.get(f)
        out.append("| `%s` | %d | %d | %s |" % (f, t["files"] if t else 0, t["funcs"] if t else 0,
                                                ("%.2f" % (t["sloc"] / max(1, m["sloc"]))) if t else "—"))
    untested = sorted((f for f in src if f not in tested), key=lambda f: -src[f]["sloc"])
    out.append("")
    out.append("**Untested modules** (no test file imports them), by SLOC: " +
               (", ".join("`%s` (%d)" % (f, src[f]["sloc"]) for f in untested) or "none"))
    out.append("")
    out.append("*Static mapping, not measured line coverage.*")
    blocks["metrics-tests"] = "\n".join(out)

    # churn + hotspots
    cc_sum = {f: sum(fn[2] for fn in m["funcs"]) for f, m in src.items()}
    ch = {f: commits.get(f, 0) for f in src}
    pr_c, pr_h = pct_rank(cc_sum), pct_rank(ch)
    hot = sorted(src, key=lambda f: -(pr_c[f] * pr_h[f]))[:10]
    out = ["| file | commits (12 mo) | lines changed | complexity sum | hotspot score | tested |", "|---|---|---|---|---|---|"]
    for f in hot:
        out.append("| `%s` | %d | %d | %d | %.2f %s | %s |" % (f, ch[f], lines.get(f, 0), cc_sum[f], pr_c[f] * pr_h[f],
                                                            bar(pr_c[f] * pr_h[f], 1), "yes" if f in tested else "**no ⚠**"))
    out.append("")
    out.append("`hotspot = pct_rank(complexity sum) × pct_rank(commits in 12 months)`. Churn counts come from "
               "`git log --numstat` (commit counts only, no authors). Uncommitted edits are not counted.")
    blocks["metrics-hotspots"] = "\n".join(out)

    # coupling
    fan_out = {f: {names[i] for i in m["imports"] if i in names and names[i] != f} for f, m in src.items()}
    fan_in = Counter(t for deps in fan_out.values() for t in deps)
    out = ["| module | fan-in | fan-out | instability | imports |", "|---|---|---|---|---|"]
    for f in sorted(src, key=lambda f: (-fan_in[f], f)):
        fi, fo = fan_in[f], len(fan_out[f])
        inst = fo / (fi + fo) if fi + fo else 0.0
        out.append("| `%s` | %d | %d | %.2f | %s |" % (f, fi, fo, inst, ", ".join("`%s`" % src[d]["name"] for d in sorted(fan_out[f])) or "—"))
    cycles = sorted({tuple(sorted((a, b))) for a in fan_out for b in fan_out[a] if a in fan_out.get(b, set())})
    out.append("")
    out.append("Import cycles: " + (", ".join("`%s` ↔ `%s`" % c for c in cycles) if cycles else "none"))
    blocks["metrics-coupling"] = "\n".join(out)

    td = todos(files)
    out = ["| file:line | text |", "|---|---|"] + ["| `%s:%d` | %s |" % (f, i, t.replace("|", "\\|")) for f, i, t in td[:10]]
    if not td:
        out = ["No TODO / FIXME / HACK / XXX markers found."]
    blocks["todo"] = "\n".join(out)

    pct_tested = 100.0 * len(tested) / max(1, len(src))
    blocks["health"] = ("**Code health:** %d source modules, %d SLOC; %.0f%% of modules have a test file importing them "
                        "(static mapping); top hotspots %s; complexity by stdlib `ast` (radon not used). Detail: "
                        "[docs/map/metrics.md](docs/map/metrics.md)." % (
                            len(src), tot, pct_tested, ", ".join("`%s`" % f for f in hot[:3])))
    return blocks


# ----------------------------------------------------------------------------------------------- staleness
def stamp_commit():
    try:
        m = re.search(r"at commit `?([0-9a-f]{7,40})", open(os.path.join(ROOT, "CLAUDE.md")).read())
        return m.group(1) if m else None
    except OSError:
        return None


def diagram_sources():
    srcs = set()
    for a in ARCH_FILES:
        p = os.path.join(ROOT, a)
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8").read()
        for sect in re.findall(r"\*\*Sources[^\n]*\n((?:[-*] .*\n?)+)", text):
            srcs.update(re.findall(r"`([^`]+\.(?:py|sh|yml|txt|md|ipynb)|Dockerfile)(?::\d+)?`", sect))
    return srcs


def fingerprint(paths):
    """Short hash of the current content of the given repo files (missing files hash as empty)."""
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.encode())
        try:
            h.update(open(os.path.join(ROOT, p), "rb").read())
        except OSError:
            pass
    return h.hexdigest()[:12]


FP_RE = re.compile(r"Sources fingerprint: `([0-9a-f]{12})`")


def staleness_report(stamp_diagrams=False):
    """Warn when a file listed under a diagram's **Sources** changed since that diagram was last reviewed.

    Each diagram file carries `Sources fingerprint: `<hash>`` (content hash of its sources at review time), so
    uncommitted edits are caught too. After re-checking a diagram, run with --stamp-diagrams to record the
    new fingerprint.
    """
    stale = []
    for a in ARCH_FILES:
        p = os.path.join(ROOT, a)
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8").read()
        srcs = set()
        for sect in re.findall(r"\*\*Sources[^\n]*\n((?:[-*] .*\n?)+)", text):
            srcs.update(re.findall(r"`([^`]+\.(?:py|sh|yml|txt|md|ipynb)|Dockerfile)(?::\d+)?`", sect))
        now = fingerprint(srcs)
        m = FP_RE.search(text)
        if stamp_diagrams:
            new = FP_RE.sub("Sources fingerprint: `%s`" % now, text) if m else text.replace(
                "**Sources**", "Sources fingerprint: `%s`\n\n**Sources**" % now, 1)
            if new != text:
                open(p, "w", encoding="utf-8").write(new)
            print("stamped %s: %s" % (a, now))
        elif not m or m.group(1) != now:
            stale.append(a)
    if stale:
        print("diagrams may be stale: " + ", ".join(stale) + " (a Sources file changed since the last review; "
              "re-check the diagram, then run --stamp-diagrams)")
    elif not stamp_diagrams:
        print("diagrams: sources unchanged since last review")
    head = git("rev-parse", "--short", "HEAD").strip(); stamp = stamp_commit()
    if stamp and head and not head.startswith(stamp) and not stamp.startswith(head):
        changed = git("diff", "--name-only", stamp, "HEAD").split()
        dirs = sorted({os.path.dirname(f) or "." for f in changed})
        print("map stamp %s != HEAD %s; re-scan: %s" % (stamp, head, ", ".join(dirs) or "-"))
    return stale


# ----------------------------------------------------------------------------------------------- main
def replace_blocks(path, blocks):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return False
    text = open(full, encoding="utf-8").read()

    def sub(m):
        name = m.group("name")
        if name not in blocks:
            return m.group(0)
        return m.group(1) + blocks[name].rstrip("\n") + "\n" + m.group(4)
    new = AUTO_RE.sub(sub, text)
    if new != text:
        open(full, "w", encoding="utf-8").write(new)
        return True
    return False


def current_blocks(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return {}
    return {m.group("name"): m.group("body") for m in AUTO_RE.finditer(open(full, encoding="utf-8").read())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report staleness only; write nothing")
    ap.add_argument("--stamp-diagrams", action="store_true",
                    help="after reviewing the diagrams, record the current fingerprint of their Sources")
    args = ap.parse_args()
    if args.check:
        staleness_report(); return 0
    if args.stamp_diagrams:
        staleness_report(stamp_diagrams=True); return 0

    files = repo_files()
    py = [f for f in files if f.endswith(".py")]
    mods = analyse(py)
    blocks = {"tree": tree_block(files), "symbols": symbols_block(mods)}
    blocks.update(metrics_blocks(mods, files))

    head = git("rev-parse", "--short", "HEAD").strip() or "unknown"
    dirty = [ln for ln in git("status", "--porcelain").splitlines() if ln]
    existing = {}
    for p in MAP_FILES:
        existing.update(current_blocks(p))
    content_changed = any(existing.get(k, "").rstrip("\n") != v.rstrip("\n") for k, v in blocks.items() if k in existing)
    old_stamp = existing.get("stamp", "")
    if content_changed or head not in old_stamp:
        blocks["stamp"] = "Generated: %s at commit `%s` (working tree had %d uncommitted/untracked paths)." % (
            datetime.date.today().isoformat(), head, len(dirty))
    wrote = [p for p in MAP_FILES if replace_blocks(p, blocks)]
    print("updated: " + (", ".join(wrote) if wrote else "nothing (already current)"))
    staleness_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
