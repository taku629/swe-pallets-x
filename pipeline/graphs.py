#!/usr/bin/env python3
"""Generate AST call/dependency graphs in the competition's NetworkX
node-link JSON schema:

  directed=True, multigraph=True,
  nodes: [{id, name, text}]  id = fully qualified symbol path
  edges: [{source, target, type, key}]  type in {calls, imports, contains}

Usage: python graphs.py <snapshot_dir> <repo_prefix> <out.json>
"""
import ast
import json
import os
import sys


def qualname_stack(module, names):
    return module + "." + ".".join(names) if module else ".".join(names)


class Indexer(ast.NodeVisitor):
    """Collect symbol definitions: modules, classes, functions, methods."""

    def __init__(self, module):
        self.module = module
        self.stack = []
        self.nodes = {}          # id -> {id, name, text, file}
        self.contains = []       # (parent_id, child_id)
        self.bodies = {}         # id -> ast node (for call extraction)

    def _add(self, node, name, src):
        qn = qualname_stack(self.module, self.stack + [name])
        self.nodes[qn] = {"id": qn, "name": qn, "text": src, "file": ""}
        if self.stack:
            self.contains.append((qualname_stack(self.module, self.stack), qn))
        self.bodies[qn] = node
        return qn

    def visit_ClassDef(self, node):
        src = ast.get_source_segment(self.src, node) or ""
        self._add(node, node.name, src)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _func(self, node):
        src = ast.get_source_segment(self.src, node) or ""
        self._add(node, node.name, src)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func


def collect_calls(node):
    """Yield dotted names referenced inside a definition body."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            parts = []
            while isinstance(f, ast.Attribute):
                parts.append(f.attr)
                f = f.value
            if isinstance(f, ast.Name):
                parts.append(f.id)
            if parts:
                yield ".".join(reversed(parts))
        elif isinstance(sub, ast.Name):
            yield sub.id


def build_graph(root):
    root = os.path.abspath(root)
    indexer = Indexer("")
    indexer.src = ""
    # pass 1: index all symbols per module
    for dirpath, _, files in os.walk(root):
        if any(s in dirpath for s in ("/.git", "/.venv", "/tests", "/test/")):
            pass  # tests still indexed (symbols exist), nothing special
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root)
            module = rel[:-3].replace(os.sep, ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            try:
                src = open(p, encoding="utf-8", errors="replace").read()
                tree = ast.parse(src)
            except Exception:
                continue
            ix = Indexer(module)
            ix.src = src
            ix.visit(tree)
            for k, v in ix.nodes.items():
                v["file"] = rel
                indexer.nodes.setdefault(k, v)
            indexer.contains += ix.contains
            indexer.bodies.update(ix.bodies)

    nodes = indexer.nodes
    edges = []
    # containment edges
    for a, b in indexer.contains:
        if a in nodes and b in nodes:
            edges.append({"source": a, "target": b, "type": "contains", "key": 0})

    # call edges (resolve suffix-match against indexed ids)
    ids = set(nodes)
    short = {}
    for i in ids:
        short.setdefault(i.split(".")[-1], []).append(i)
    for src_id, body in indexer.bodies.items():
        seen = set()
        for call in collect_calls(body):
            cand = None
            if call in ids:
                cand = call
            else:
                # resolve by last component
                last = call.split(".")[-1]
                opts = short.get(last, [])
                if len(opts) == 1:
                    cand = opts[0]
                elif call.split(".")[0] in short:
                    # module.attr form: match longest id suffix
                    for i in ids:
                        if i.endswith("." + call):
                            cand = i
                            break
            if cand and cand != src_id and cand not in seen:
                seen.add(cand)
                edges.append({"source": src_id, "target": cand,
                              "type": "calls", "key": len(seen) - 1})

    return {
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": [{"id": n["id"], "name": n["name"], "text": n["text"]}
                  for n in nodes.values()],
        "edges": edges,
    }


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[3]
    g = build_graph(root)
    json.dump(g, open(out, "w"))
    print(f"{len(g['nodes'])} nodes, {len(g['edges'])} edges -> {out}",
          file=sys.stderr)
