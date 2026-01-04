# auto_annotator.py
# Run in pre-commit or CI to propose/insert annotations automatically.
import ast, os, sys, io, re
from pathlib import Path

BEHAVIOR_TEMPLATE = '@BehaviorTag("{tag}")\n@Flavor("{flavor}")\n'
DEFAULT_TAG = "default_behavior"
DEFAULT_FLAVOR = "neutral"

def find_py_files(root="."):
    for p in Path(root).rglob("*.py"):
        yield p

def parse_and_patch_file(path: Path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    inserts = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # skip if already annotated (simple textual check)
            start_line = node.lineno - 1
            # get leading lines to check for decorators
            prev_lines = src.splitlines()[max(0,start_line-5):start_line]
            joined = "\n".join(prev_lines)
            if re.search(r'@BehaviorTag\(', joined) or re.search(r'@Flavor\(', joined):
                continue
            # propose insertion before function
            tag = DEFAULT_TAG
            flavor = DEFAULT_FLAVOR
            inserts.append((node.lineno, BEHAVIOR_TEMPLATE.format(tag=tag, flavor=flavor)))
    if not inserts:
        return False
    # apply inserts in reverse order to not shift lines
    lines = src.splitlines()
    for lineno, text in sorted(inserts, key=lambda x: -x[0]):
        lines.insert(lineno-1, text.rstrip())
    path.write_text("\n".join(lines), encoding="utf-8")
    return True

if __name__ == "__main__":
    changed = False
    for f in find_py_files("."):
        if parse_and_patch_file(f):
            print(f"Patched annotations in {f}")
            changed = True
    if changed:
        print("Files modified with default annotations. Please review before commit.")
        sys.exit(1)  # non-zero to surface in CI; in pre-commit make optional
    print("No annotation patches required.")
