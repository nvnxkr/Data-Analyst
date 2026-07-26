"""
safe_exec.py

Executes LLM-generated pandas code safely. We NEVER just eval()/exec() raw
LLM output. Instead:
  1. Parse the code into an AST.
  2. Walk the AST and reject anything not on a strict allow-list
     (no imports, no attribute access to dunders, no calls to
     eval/exec/open/compile/__import__, no access to os/sys/subprocess, etc.)
  3. Only if the code passes validation do we exec it, and even then in a
     restricted namespace that exposes only `df`, `pd`, `np`, and `result`.
"""

import ast
import pandas as pd
import numpy as np

FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "shutil", "socket", "importlib", "builtins",
    "__import__", "eval", "exec", "compile", "open", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
    "__builtins__", "__loader__", "__spec__",
}

ALLOWED_NODE_TYPES = (
    ast.Module, ast.Expr, ast.Load, ast.Store,
    ast.Assign, ast.AugAssign, ast.Name, ast.Attribute, ast.Call,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    ast.Subscript, ast.Index, ast.Slice, ast.List, ast.Tuple, ast.Dict,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.keyword, ast.Starred, ast.comprehension, ast.ListComp, ast.GeneratorExp,
    ast.For, ast.If, ast.Return,
)


class UnsafeCodeError(Exception):
    pass


def validate_code(code: str) -> None:
    """Raise UnsafeCodeError if the code contains anything outside the allow-list."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"Generated code has a syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Import statements are not allowed.")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            raise UnsafeCodeError("Function/class definitions are not allowed.")
        if isinstance(node, ast.With) or isinstance(node, ast.Try):
            raise UnsafeCodeError("with/try blocks are not allowed.")
        if not isinstance(node, ALLOWED_NODE_TYPES):
            raise UnsafeCodeError(f"Disallowed syntax: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Use of forbidden name: {node.id}")

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCodeError("Dunder attribute access is not allowed.")
            if node.attr in {"to_csv", "to_excel", "to_pickle", "to_sql", "eval", "query"}:
                # to_csv/to_excel could write files; query/eval on df can run arbitrary
                # expressions we haven't vetted, so keep the surface area small.
                raise UnsafeCodeError(f"Use of disallowed method: {node.attr}")

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                raise UnsafeCodeError(f"Call to forbidden function: {func.id}")


def run_generated_code(code: str, df: pd.DataFrame):
    """
    Validate then execute LLM-generated pandas code.
    The code must assign its final answer to a variable named `result`.
    Returns the value of `result` (typically a DataFrame, Series, or scalar).
    """
    validate_code(code)

    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "sorted": sorted, "abs": abs, "round": round, "list": list,
        "dict": dict, "tuple": tuple, "str": str, "int": int, "float": float,
        "bool": bool, "enumerate": enumerate, "zip": zip,
    }
    local_ns = {"df": df.copy(), "pd": pd, "np": np}
    global_ns = {"__builtins__": safe_builtins}

    exec(compile(code, "<llm_generated>", "exec"), global_ns, local_ns)

    if "result" not in local_ns:
        raise UnsafeCodeError("Generated code did not assign an output to `result`.")

    return local_ns["result"]
