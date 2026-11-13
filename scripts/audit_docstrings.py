from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def has_type_hints(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, bool]:
    """Returns (has_args_hints, has_return_hint)."""
    args = node.args
    all_args = args.posonlyargs + args.args + args.kwonlyargs
    # exclude self/cls
    relevant = [a for a in all_args if a.arg not in {"self", "cls"}]
    has_args = all(a.annotation is not None for a in relevant) if relevant else True
    has_return = node.returns is not None
    return has_args, has_return


def check_file(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"path": str(path), "error": f"SyntaxError: {exc}"}

    module_doc = ast.get_docstring(tree) is not None

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # skip private
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue
            doc = ast.get_docstring(node) is not None
            args_ok, ret_ok = has_type_hints(node)
            if not (doc and args_ok and ret_ok):
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "docstring": doc,
                    "args_hints": args_ok,
                    "return_hint": ret_ok,
                })

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) is not None
            if not doc:
                classes.append({"name": node.name, "line": node.lineno})

    return {
        "path": str(path.relative_to(ROOT)),
        "module_docstring": module_doc,
        "missing_function_docs": functions,
        "missing_class_docs": classes,
    }


def main() -> None:
    targets = sorted(ROOT.glob("src/**/*.py")) + sorted(ROOT.glob("scripts/**/*.py"))

    total_issues = 0
    print(f"Auditing {len(targets)} files\n")

    for path in targets:
        if path.name == "__init__.py":
            continue
        result = check_file(path)
        issues = 0

        if "error" in result:
            print(f"[ERROR] {result['path']}: {result['error']}")
            total_issues += 1
            continue

        if not result["module_docstring"]:
            print(f"[MISSING MODULE DOC] {result['path']}")
            issues += 1

        for fn in result["missing_function_docs"]:
            flags = []
            if not fn["docstring"]:
                flags.append("no docstring")
            if not fn["args_hints"]:
                flags.append("missing arg hints")
            if not fn["return_hint"]:
                flags.append("missing return hint")
            print(f"  [FN] {result['path']}:{fn['line']} {fn['name']}() — {', '.join(flags)}")
            issues += 1

        for cls in result["missing_class_docs"]:
            print(f"  [CLASS] {result['path']}:{cls['line']} {cls['name']} — no docstring")
            issues += 1

        total_issues += issues

    print(f"\n{'='*60}")
    print(f"Total issues: {total_issues}")
    if total_issues == 0:
        print("✅ Every public function has a docstring and full type hints.")
    else:
        print("Fix the issues above before submission.")


if __name__ == "__main__":
    main()
