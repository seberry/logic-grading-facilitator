from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = ("app", "scripts", "tests")
LOCAL_PACKAGE_PREFIXES = ("app", "scripts", "tests")
ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}
WORKFLOW_LABELS = {
    "/classwork": "classwork",
    "/problem/": "problem_strips",
    "/grade/": "legacy_exam",
    "/name_review": "legacy_exam",
    "/work_units": "legacy_exam",
    "/setup": "setup",
    "/gradebook_merge": "setup",
    "/flashcards": "flashcards",
    "/debug/fitch": "fitch_debug",
    "/export": "exports",
    "/finish": "exports",
    "/finalize": "exports",
    "/session/summary": "debug",
}


@dataclass
class RouteInfo:
    method: str
    path: str
    function: str
    lineno: int


@dataclass
class FileInfo:
    path: Path
    module: str
    line_count: int
    top_level_functions: List[str] = field(default_factory=list)
    top_level_classes: List[str] = field(default_factory=list)
    local_imports: List[str] = field(default_factory=list)
    routes: List[RouteInfo] = field(default_factory=list)
    function_calls: Dict[str, Set[str]] = field(default_factory=dict)


def rel_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def module_name_for_path(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def iter_python_files(directories: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for directory in directories:
        root = REPO_ROOT / directory
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.py")))
    return [p for p in files if "__pycache__" not in p.parts]


def resolve_import(current_module: str, node: ast.AST) -> List[str]:
    resolved: List[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.name
            if name.startswith(LOCAL_PACKAGE_PREFIXES):
                resolved.append(name)
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level:
            current_parts = current_module.split(".")
            if current_parts:
                current_parts = current_parts[:-1]
            if node.level > len(current_parts) + 1:
                return resolved
            base_parts = current_parts[: len(current_parts) - (node.level - 1)]
            if module:
                base_parts.extend(module.split("."))
            target = ".".join(part for part in base_parts if part)
            if target.startswith(LOCAL_PACKAGE_PREFIXES):
                resolved.append(target)
        elif module.startswith(LOCAL_PACKAGE_PREFIXES):
            resolved.append(module)
    return resolved


def route_info_from_decorator(decorator: ast.AST) -> Optional[tuple[str, str]]:
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if not isinstance(func, ast.Attribute):
        return None
    method = func.attr.lower()
    if method not in ROUTE_METHODS:
        return None
    if not decorator.args:
        return None
    first_arg = decorator.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return method.upper(), first_arg.value
    return None


def gather_called_names(function_node: ast.AST, local_defs: Set[str]) -> Set[str]:
    called: Set[str] = set()
    for node in ast.walk(function_node):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id in local_defs:
                called.add(target.id)
    return called


def classify_route(route_path: str) -> str:
    for prefix, label in WORKFLOW_LABELS.items():
        if route_path.startswith(prefix):
            return label
    return "other"


def analyze_file(path: Path) -> FileInfo:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    info = FileInfo(
        path=path,
        module=module_name_for_path(path),
        line_count=len(source.splitlines()),
    )

    top_level_defs: Dict[str, ast.AST] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.top_level_functions.append(node.name)
            top_level_defs[node.name] = node
            for decorator in node.decorator_list:
                route_meta = route_info_from_decorator(decorator)
                if route_meta:
                    method, route_path = route_meta
                    info.routes.append(
                        RouteInfo(method=method, path=route_path, function=node.name, lineno=node.lineno)
                    )
        elif isinstance(node, ast.ClassDef):
            info.top_level_classes.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            info.local_imports.extend(resolve_import(info.module, node))

    local_def_names = set(top_level_defs.keys())
    for name, node in top_level_defs.items():
        info.function_calls[name] = gather_called_names(node, local_def_names)

    info.local_imports = sorted(set(info.local_imports))
    return info


def invert_dependencies(files: Dict[str, FileInfo]) -> Dict[str, List[str]]:
    reverse: Dict[str, Set[str]] = defaultdict(set)
    for module, info in files.items():
        for dep in info.local_imports:
            reverse[dep].add(module)
    return {key: sorted(value) for key, value in reverse.items()}


def reachable_helpers(info: FileInfo, route_filter: Optional[str] = None) -> Dict[str, List[str]]:
    route_functions = [
        route.function
        for route in info.routes
        if route_filter is None or classify_route(route.path) == route_filter or route_filter in route.path
    ]
    reachable: Dict[str, List[str]] = {}
    for route_fn in route_functions:
        seen: Set[str] = set()
        queue = deque(sorted(info.function_calls.get(route_fn, set())))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            for child in sorted(info.function_calls.get(current, set())):
                if child not in seen:
                    queue.append(child)
        reachable[route_fn] = sorted(seen)
    return reachable


def build_report(files: Dict[str, FileInfo], focus: Optional[str]) -> Dict[str, object]:
    reverse_deps = invert_dependencies(files)
    summaries = []
    workflow_counter: Counter[str] = Counter()
    for module, info in sorted(files.items(), key=lambda item: rel_path(item[1].path)):
        route_labels = Counter(classify_route(route.path) for route in info.routes)
        workflow_counter.update(route_labels)
        summaries.append(
            {
                "file": rel_path(info.path),
                "module": module,
                "line_count": info.line_count,
                "function_count": len(info.top_level_functions),
                "class_count": len(info.top_level_classes),
                "route_count": len(info.routes),
                "local_imports": info.local_imports,
                "imported_by": reverse_deps.get(module, []),
                "routes": [
                    {
                        "method": route.method,
                        "path": route.path,
                        "function": route.function,
                        "workflow": classify_route(route.path),
                    }
                    for route in info.routes
                ],
                "top_level_functions": info.top_level_functions,
                "top_level_classes": info.top_level_classes,
            }
        )

    hotspot_files = sorted(
        summaries,
        key=lambda item: (item["route_count"], item["function_count"], item["line_count"]),
        reverse=True,
    )

    focus_report = None
    if focus:
        focus_files = []
        for module, info in files.items():
            matching_routes = [
                {
                    "method": route.method,
                    "path": route.path,
                    "function": route.function,
                    "lineno": route.lineno,
                }
                for route in info.routes
                if focus in route.path or classify_route(route.path) == focus
            ]
            if not matching_routes:
                continue
            focus_files.append(
                {
                    "file": rel_path(info.path),
                    "module": module,
                    "routes": matching_routes,
                    "reachable_local_helpers": reachable_helpers(info, focus),
                    "local_imports": info.local_imports,
                    "imported_by": reverse_deps.get(module, []),
                }
            )
        focus_report = {
            "focus": focus,
            "files": sorted(focus_files, key=lambda item: item["file"]),
        }

    return {
        "generated_from": [rel_path(info.path) for info in files.values()],
        "workflow_route_counts": dict(sorted(workflow_counter.items())),
        "files": summaries,
        "hotspots": hotspot_files[:15],
        "focus_report": focus_report,
    }


def markdown_report(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Code Map")
    lines.append("")
    workflow_counts = report.get("workflow_route_counts", {})
    if workflow_counts:
        lines.append("## Workflow Route Counts")
        lines.append("")
        for key, value in workflow_counts.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")

    lines.append("## Hotspots")
    lines.append("")
    for item in report["hotspots"]:
        lines.append(
            f"- `{item['file']}`: {item['line_count']} LOC, "
            f"{item['function_count']} functions, {item['route_count']} routes"
        )
    lines.append("")

    focus = report.get("focus_report")
    if focus:
        lines.append(f"## Focus: `{focus['focus']}`")
        lines.append("")
        for item in focus["files"]:
            lines.append(f"### `{item['file']}`")
            lines.append("")
            for route in item["routes"]:
                lines.append(f"- Route `{route['method']} {route['path']}` -> `{route['function']}`")
                helpers = item["reachable_local_helpers"].get(route["function"], [])
                helper_text = ", ".join(f"`{name}`" for name in helpers) if helpers else "(none)"
                lines.append(f"  Local helper reach: {helper_text}")
            if item["local_imports"]:
                deps = ", ".join(f"`{dep}`" for dep in item["local_imports"])
                lines.append(f"- Local imports: {deps}")
            if item["imported_by"]:
                rev = ", ".join(f"`{dep}`" for dep in item["imported_by"])
                lines.append(f"- Imported by: {rev}")
            lines.append("")

    lines.append("## Files")
    lines.append("")
    for item in report["files"]:
        lines.append(f"### `{item['file']}`")
        lines.append("")
        lines.append(
            f"- {item['line_count']} LOC, {item['function_count']} functions, "
            f"{item['class_count']} classes, {item['route_count']} routes"
        )
        if item["local_imports"]:
            lines.append("- Local imports: " + ", ".join(f"`{dep}`" for dep in item["local_imports"]))
        if item["imported_by"]:
            lines.append("- Imported by: " + ", ".join(f"`{dep}`" for dep in item["imported_by"]))
        if item["routes"]:
            lines.append("- Routes:")
            for route in item["routes"]:
                lines.append(
                    f"  - `{route['method']} {route['path']}` -> `{route['function']}` "
                    f"({route['workflow']})"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local code/dependency map.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=list(DEFAULT_DIRS),
        help="Top-level directories to scan.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "md"),
        default="md",
        help="Output format.",
    )
    parser.add_argument(
        "--out",
        help="Optional output file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--focus",
        help="Optional workflow/path focus such as classwork, problem_strips, flashcards, /classwork, /problem/.",
    )
    args = parser.parse_args()

    files = {
        info.module: info
        for info in (analyze_file(path) for path in iter_python_files(args.dirs))
    }
    report = build_report(files, args.focus)
    if args.format == "json":
        output = json.dumps(report, indent=2)
    else:
        output = markdown_report(report)

    if args.out:
        out_path = (REPO_ROOT / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
