"""
Verbatim state ledger for history compression.

An LLM summary can drop or rewrite exact runtime values (access tokens, ids, page
indices); ACON's optimized AppWorld guideline even instructs the compressor to truncate
tokens to "<token>" and to remove page_index/page_limit. The ledger is extracted with no
LLM call from the raw interactions being compressed and appended verbatim to the summary,
so those values survive however the summary is written.

AppWorld agents act by writing Python into a persistent REPL, so the state worth keeping
is mostly the variables they assigned, the APIs they already called, and the credentials
and errors they saw. Extraction is from the code's syntax tree plus a few regexes over
the execution output; nothing is executed.
"""
import ast
import re
from collections import OrderedDict
from typing import Dict, List, Optional

JWT_RE = re.compile(r"eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NO_API_RE = re.compile(r"No API named '(\w+)' found in the (\w+) app")
ERROR_RE = re.compile(r"\w*(?:Exception|Error)\b[^\n]{0,160}")

PREVIEW_CHARS = 80
MAX_VARS = 25
MAX_APIS = 25
MAX_CREDENTIALS = 6
MAX_ERRORS = 3


def _api_name(func: ast.AST) -> Optional[str]:
    """'app.api' for a call to apis.<app>.<api>(...), else None."""
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name) and func.value.value.id == "apis"):
        return f"{func.value.attr}.{func.attr}"
    return None


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _literal(node: ast.AST) -> Optional[str]:
    """Source of a small literal value (str/number/list/dict of literals), else None."""
    try:
        ast.literal_eval(node)
    except Exception:
        return None
    src = ast.unparse(node)
    return src if len(src) <= 200 else None


class StateLedger:
    """Accumulates state across all compressions within one task."""

    SECTIONS = ("values", "errors")

    def __init__(self, sections=SECTIONS):
        # "values": variables, credentials, APIs already called.
        # "errors": APIs that do not exist, recent errors. Used for the ablation.
        unknown = set(sections) - set(self.SECTIONS)
        if unknown:
            raise ValueError(f"Unknown ledger sections: {sorted(unknown)}")
        self.sections = set(sections)
        self.vars: "OrderedDict[str, str]" = OrderedDict()
        self.apis: "OrderedDict[str, Dict]" = OrderedDict()
        self.credentials: "OrderedDict[str, None]" = OrderedDict()
        self.errors: List[str] = []
        # APIs the environment said do not exist (the agent guessing names is a common loop)
        self.missing_apis: "OrderedDict[str, None]" = OrderedDict()

    # ---- extraction -------------------------------------------------------------
    def update(self, raw_history: List[Dict[str, str]]) -> None:
        """Read (assistant code, user observation) pairs from the messages being compressed.

        The system message and the first user message are skipped: they hold the task
        prompt, whose few-shot example contains dummy tokens and emails, and after a
        previous compression the previous summary (whose ledger is already in self).
        """
        msgs = raw_history[2:] if len(raw_history) > 2 else []
        for i, msg in enumerate(msgs):
            if msg.get("role") != "assistant":
                continue
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            obs = nxt["content"] if nxt and nxt.get("role") == "user" else ""
            self._update_from_step(msg.get("content", ""), obs or "")

    def _update_from_step(self, code: str, obs: str) -> None:
        failed = bool(re.search(r"Traceback|Exception|Error", obs))
        for m in ERROR_RE.findall(obs) if failed else []:
            self._add_error(m)
        for api, app in NO_API_RE.findall(obs):
            self.missing_apis[f"{app}.{api}"] = None
            self.apis.pop(f"{app}.{api}", None)
        for tok in JWT_RE.findall(obs) + JWT_RE.findall(code):
            self.credentials[tok] = None
        for em in EMAIL_RE.findall(obs):
            self.credentials[em] = None

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _api_name(node.func)
                if name and name not in self.missing_apis:
                    entry = self.apis.pop(name, {"n": 0, "args": {}})
                    entry["n"] += 1
                    for kw in node.keywords:
                        if kw.arg and _literal(kw.value) is not None:
                            entry["args"][kw.arg] = _preview(ast.unparse(kw.value), 40)
                    self.apis[name] = entry  # re-insert: most recent last

        # A step that raised may not have defined everything it assigns, so only
        # successful steps add variables.
        if failed:
            return
        printed = self._printed_names(tree)
        mutated = self._mutated_names(tree)
        for stmt in tree.body:
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
                value = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets = [stmt.target]
                value = stmt.value
            for t in targets:
                for name in self._names(t):
                    lit = _literal(value)
                    if name in mutated:
                        # e.g. all_songs = [] then appended in a loop: the literal is stale
                        val = "changed later in the same step" + (f" (started as {lit})" if lit else "")
                    elif lit is not None:
                        val = lit
                    elif name in printed and obs.strip():
                        val = "printed: " + _preview(obs)
                    else:
                        val = "= " + _preview(ast.unparse(value), 60)
                    self.vars.pop(name, None)
                    self.vars[name] = val

    @staticmethod
    def _names(target: ast.AST) -> List[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for elt in target.elts for n in StateLedger._names(elt)]
        return []

    MUTATING_METHODS = {"append", "extend", "insert", "update", "pop", "remove", "clear", "add", "setdefault"}

    @staticmethod
    def _mutated_names(tree: ast.Module) -> set:
        """Names changed after their top-level assignment: augmented assignment,
        in-place method calls, or reassignment nested inside a loop/if/with block."""
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign):
                out.update(StateLedger._names(node.target))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.attr in StateLedger.MUTATING_METHODS):
                out.add(node.func.value.id)
        top = set(map(id, tree.body))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and id(node) not in top:
                for t in node.targets:
                    out.update(StateLedger._names(t))
        return out

    @staticmethod
    def _printed_names(tree: ast.AST) -> set:
        out = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                out.update(a.id for a in node.args if isinstance(a, ast.Name))
        return out

    def _add_error(self, msg: str) -> None:
        msg = _preview(msg, 160)
        if msg in self.errors:
            self.errors.remove(msg)
        self.errors.append(msg)
        del self.errors[:-MAX_ERRORS]

    # ---- rendering --------------------------------------------------------------
    def render(self) -> str:
        body = []
        if "values" in self.sections:
            if self.vars:
                body.append("VARIABLES:")
                body += [f"- {k}: {v}" for k, v in list(self.vars.items())[-MAX_VARS:]]
            if self.credentials:
                body.append("CREDENTIALS / ACCOUNTS SEEN (verbatim):")
                body += [f"- {c}" for c in list(self.credentials)[-MAX_CREDENTIALS:]]
            if self.apis:
                body.append("APIS ALREADY CALLED (count, last literal arguments):")
                for name, e in list(self.apis.items())[-MAX_APIS:]:
                    args = ", ".join(f"{k}={v}" for k, v in e["args"].items())
                    body.append(f"- apis.{name} x{e['n']}" + (f" ({args})" if args else ""))
        if "errors" in self.sections:
            if self.missing_apis:
                body.append("APIS THAT DO NOT EXIST (do not call again; check api_docs instead):")
                body += [f"- apis.{a}" for a in self.missing_apis]
            if self.errors:
                body.append("RECENT ERRORS:")
                body += [f"- {e}" for e in self.errors]
        if not body:
            return ""
        header = "Exact values extracted automatically from the code already executed."
        if "values" in self.sections:
            header += " Variables listed here still exist in the Python session and can be used directly."
        return "\n".join(["<STATE_LEDGER>", header] + body + ["</STATE_LEDGER>"])
