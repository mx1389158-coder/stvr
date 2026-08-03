import ast
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings
warnings.filterwarnings("ignore", message=".*cgi.*deprecated.*", category=DeprecationWarning)
import cgi
import copy
import csv
import gc
import hashlib
import io
import signal
import itertools
import json
import re
import secrets
import resource
import shutil
import sqlite3
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
import threading
import textwrap
import time
import urllib.request
import urllib.parse
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "records.db"
LIMIT = 120000
TIMEOUT = 8
MAXFILES = 64
MAXZIP = 2 * 1024 * 1024
MAXLOG = 2400
MAXTEST = 48000
ROOTMODEL = Path("/root/autodl-tmp/utplm/models/base")
ROOTADAPTER = Path("/root/autodl-tmp/utplm/models/adapters")
LOCALPIPE = {"model": None, "tokenizer": None, "path": None, "adapter": None}
LOCALLOCK = threading.Lock()
RUNTIME = {"checked": False}
JOBS = {}
JOBLOCK = threading.Lock()
JOBTTL = 3600
JOBTIMES = {}

USERS = {
    "demo": {"password": "demo123", "role": "开发者"},
    "tester": {"password": "test123", "role": "测试工程师"},
    "admin": {"password": "admin123", "role": "管理员"},
}

BUILTIN_EXCEPTIONS = {"Exception", "TypeError", "ValueError", "IndexError", "KeyError", "ZeroDivisionError", "AssertionError"}
BLOCKED_NAMES = {"eval", "exec", "open", "input", "compile", "__import__", "globals", "locals", "vars", "breakpoint", "help", "dir"}
BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests", "urllib", "http", "ftplib", "multiprocessing", "ctypes", "asyncio", "threading", "concurrent", "resource", "signal", "site", "importlib", "builtins", "pickle", "marshal", "shelve", "sqlite3", "tempfile", "mmap", "psutil"}
BLOCKED_ATTRS = {"__subclasses__", "__globals__", "__code__", "__builtins__", "__import__", "__class__", "__mro__", "__base__", "__bases__", "__getattribute__"}
SAFE_MODULE_ATTRS = {"asyncio": {"sleep", "gather", "wait_for", "run"}}
ALLOWED_MODULES = {"math", "re", "statistics", "itertools", "functools", "collections", "typing", "dataclasses", "decimal", "fractions", "heapq", "bisect", "string", "random", "enum", "operator", "asyncio"}




def localruntime():
    if RUNTIME.get("checked"):
        return dict(RUNTIME)
    info = {"checked": True, "torch": False, "cuda": False, "count": 0, "devices": [], "allowcpu": os.environ.get("STVR_LOCAL_ALLOW_CPU", "0") == "1"}
    try:
        import torch
        info["torch"] = True
        info["cuda"] = bool(torch.cuda.is_available())
        info["count"] = int(torch.cuda.device_count())
        for index in range(info["count"]):
            props = torch.cuda.get_device_properties(index)
            try:
                free, total = torch.cuda.mem_get_info(index)
            except Exception:
                free, total = 0, props.total_memory
            info["devices"].append({"name": props.name, "totalgb": round(total / 1024 ** 3, 2), "freegb": round(free / 1024 ** 3, 2)})
    except Exception as exc:
        info["error"] = str(exc)
    RUNTIME.clear()
    RUNTIME.update(info)
    return dict(info)


def modelconfig():
    provider = os.environ.get("STVR_PROVIDER", "auto").lower()
    url = os.environ.get("STVR_LLM_URL", "")
    key = os.environ.get("STVR_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("STVR_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    local = os.environ.get("STVR_LOCAL_MODEL", "")
    if not local:
        for name in ["qwen2.5-7b-instruct", "deepseek-coder-6.7b-instruct", "qwen2.5-14b-instruct"]:
            candidate = ROOTMODEL / name
            if candidate.exists():
                local = str(candidate)
                break
    localenabled = os.environ.get("STVR_LOCAL_ENABLED", "1") == "1"
    adapter = os.environ.get("STVR_LOCAL_ADAPTER", "").strip()
    if adapter.lower() in {"0", "none", "off", "base"}:
        adapter = ""
    if not adapter and os.environ.get("STVR_LOCAL_ADAPTER_AUTO", "0") == "1":
        defaultadapter = ROOTADAPTER / "c1_controls_v1" / "chosen_only_sft_HH_seed11"
        if defaultadapter.exists():
            adapter = str(defaultadapter)
    localready = bool(local and Path(local).exists())
    adapterready = bool(adapter and Path(adapter).exists())
    runtime = localruntime()
    localrunnable = bool(localenabled and localready and (runtime.get("cuda") or runtime.get("allowcpu")))
    available = bool((url or key) or localrunnable)
    active = "none"
    if provider in {"openai", "api"} or (provider == "auto" and (url or key)):
        active = "api" if (url or key) else "none"
    elif provider == "local" or (provider == "auto" and localenabled and localready):
        active = "local" if localready else "none"
    return {"provider": provider, "active": active, "available": available, "url": url, "model": model, "local": local, "localadapter": adapter, "adapterready": adapterready, "localenabled": localenabled, "localready": localready, "localrunnable": localrunnable, "runtime": runtime, "loaded": LOCALPIPE["model"] is not None, "loadedpath": LOCALPIPE.get("path") or "", "loadedadapter": LOCALPIPE.get("adapter") or "", "quant": os.environ.get("STVR_LOCAL_QUANT", "none")}


def refreshruntime():
    RUNTIME.clear()
    RUNTIME.update({"checked": False})
    return localruntime()


def unloadlocal():
    with LOCALLOCK:
        LOCALPIPE.update({"model": None, "tokenizer": None, "path": None, "adapter": None})
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    refreshruntime()
    return {"ok": True, "status": modelconfig()}


def warmuplocal():
    cfg = modelconfig()
    if cfg["active"] != "local" or not cfg.get("localrunnable"):
        raise ValueError("本地模型当前不可运行")
    start = time.time()
    prompt = "Return only Python pytest code for:\ndef add(a, b):\n    return a + b\n"
    text = localgenerate(prompt, cfg)
    return {"ok": True, "seconds": round(time.time() - start, 2), "preview": extractcode(text)[:300], "status": modelconfig()}


def intval(name, default, low, high):
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = default
    return max(low, min(high, value))


def sandboxconfig():
    timeout = intval("STVR_SANDBOX_TIMEOUT", TIMEOUT, 2, 30)
    cpu = intval("STVR_SANDBOX_CPU", max(2, timeout), 1, 30)
    memory = intval("STVR_SANDBOX_MEMORY_MB", 256, 64, 2048)
    files = intval("STVR_SANDBOX_FILE_MB", 32, 4, 256)
    processes = intval("STVR_SANDBOX_PROCESSES", 24, 4, 128)
    return {
        "mode": "docker" if dockerconfig().get("active") else "process",
        "timeout": timeout,
        "cpu": cpu,
        "memorymb": memory,
        "filemb": files,
        "processes": processes,
        "network": "blocked-by-policy",
        "imports": sorted(ALLOWED_MODULES),
        "blocked": sorted(BLOCKED_MODULES),
        "maxfiles": MAXFILES,
        "maxchars": LIMIT,
        "maxtestchars": MAXTEST,
        "dockerready": dockerconfig(),
    }


def safeenv():
    keep = {"PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH"}
    env = {key: value for key, value in os.environ.items() if key in keep}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["STVR_SANDBOX"] = "1"
    env["HOME"] = "/tmp"
    return env


def guardprocess():
    cfg = sandboxconfig()
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cfg["cpu"], cfg["cpu"] + 1))
        mem = cfg["memorymb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        size = cfg["filemb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (cfg["processes"], cfg["processes"]))
    except Exception:
        pass
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass


def auditlogs(text):
    raw = str(text or "")
    lower = raw.lower()
    findings = []
    rules = [
        ("timeout", "执行超时", ["timeout", "timed out", "超时"]),
        ("memory", "内存限制", ["memoryerror", "cannot allocate memory", "killed"]),
        ("import", "导入受限", ["modulenotfounderror", "importerror", "不允许导入"]),
        ("permission", "文件访问受限", ["permissionerror", "operation not permitted", "read-only", "denied"]),
        ("network", "网络访问受限", ["socket", "connection", "network", "requests"]),
        ("syntax", "语法错误", ["syntaxerror", "indentationerror"]),
        ("assertion", "断言失败", ["assertionerror", "assert "]),
        ("exception", "运行异常", ["traceback", "exception", "error"]),
    ]
    for code, label, words in rules:
        if any(word in lower for word in words):
            findings.append({"code": code, "label": label})
    if not findings:
        findings.append({"code": "clean", "label": "日志正常"})
    redacted = re.sub(r"/root/[^\s:]+", "<workspace>", raw)
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer <redacted>", redacted)
    return {"findings": findings[:5], "summary": " / ".join(item["label"] for item in findings[:3]), "log": redacted[-MAXLOG:]}



def dockerconfig():
    docker = shutil.which("docker")
    enabled = os.environ.get("STVR_DOCKER_ENABLED", "0") == "1"
    image = os.environ.get("STVR_DOCKER_IMAGE", "stvr-python:latest")
    strict = os.environ.get("STVR_DOCKER_STRICT", "0") == "1"
    return {"enabled": enabled, "available": bool(docker), "active": bool(enabled and docker), "image": image, "strict": strict, "network": "none", "memory": os.environ.get("STVR_DOCKER_MEMORY", "512m"), "cpus": os.environ.get("STVR_DOCKER_CPUS", "1.0")}


def dockerargs(cwd, args):
    cfg = dockerconfig()
    inner = list(args)
    if inner and inner[0] == sys.executable:
        inner[0] = "python"
    return [
        "docker", "run", "--rm", "--network", cfg["network"],
        "--cpus", cfg["cpus"], "-m", cfg["memory"],
        "-v", str(cwd) + ":/work:rw", "-w", "/work", cfg["image"],
    ] + inner


def promptfor(files, target, func, framework, targetname=None, adapter=None):
    chunks = []
    for item in files[:8]:
        content = item["content"]
        if len(content) > 4000:
            content = content[:4000] + "\n# ... truncated ..."
        chunks.append("# file: " + item["path"] + "\n" + content)
    joined = "\n\n".join(chunks)
    name = targetname or func.name
    invocation = ""
    if adapter:
        invocation = "\nUse this verified invocation pattern when the target is async or belongs to a class:\n" + adapter["preamble"] + "\n"
    return (
        "You are STVR, a Python unit-test generation model for software testing.\n"
        "Generate high-quality " + framework + " tests for target `" + name + "` in `" + target + "`.\n"
        "Requirements:\n"
        "- Return only Python test code, no markdown explanation.\n"
        "- Include normal cases, boundary cases, negative/exception cases when appropriate.\n"
        "- Import the target from its module.\n"
        "- Use strong assertions, not only isinstance or non-None checks.\n"
        "- Keep tests deterministic and lightweight.\n" + invocation + "\n"
        "Project code:\n" + joined
    )


def extractcode(text):
    text = text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        return fence.group(1).strip()
    return text


def recovercode(text):
    code = extractcode(text)
    try:
        ast.parse(code)
        return code
    except SyntaxError as original:
        lines = code.splitlines()
        for end in range(len(lines) - 1, 0, -1):
            candidate = "\n".join(lines[:end]).rstrip()
            if not candidate:
                continue
            try:
                tree = ast.parse(candidate)
            except SyntaxError:
                continue
            hastest = any(
                (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
                or (isinstance(node, ast.ClassDef) and node.name.lower().startswith("test"))
                or isinstance(node, ast.Assert)
                for node in ast.walk(tree)
            )
            if hastest:
                return candidate + "\n"
        raise original


def attachadapter(code, adapter):
    if not adapter or "from _stvrloader import module as _STVR_MODULE" in code:
        return code
    return adapter["preamble"].rstrip() + "\n\n" + code.lstrip()


def openaigenerate(prompt, cfg):
    url = cfg["url"] or "https://api.openai.com/v1/chat/completions"
    key = os.environ.get("STVR_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "")
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You generate executable Python unit tests only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.environ.get("STVR_LLM_TEMPERATURE", "0.2")),
        "max_tokens": int(os.environ.get("STVR_LLM_MAXTOKENS", "1200")),
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=int(os.environ.get("STVR_LLM_TIMEOUT", "90"))) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def localgenerate(prompt, cfg):
    path = cfg["local"]
    if not path:
        raise ValueError("未配置本地模型路径")
    runtime = cfg.get("runtime") or localruntime()
    if not (runtime.get("cuda") or runtime.get("allowcpu")):
        raise ValueError("本地 Qwen 已配置，但当前 PyTorch 未检测到 CUDA。请在有 GPU 的环境运行，或设置 STVR_LOCAL_ALLOW_CPU=1 允许慢速 CPU 推理。")
    maxprompt = int(os.environ.get("STVR_LOCAL_PROMPT_CHARS", "3200"))
    if len(prompt) > maxprompt:
        prompt = prompt[:maxprompt] + "\n# truncated for local generation"
    adapter = cfg.get("localadapter") or ""
    try:
        with LOCALLOCK:
            if LOCALPIPE["model"] is None or LOCALPIPE["path"] != path or LOCALPIPE.get("adapter") != adapter:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from transformers.utils import logging as transformerslogging
                transformerslogging.set_verbosity_error()
                transformerslogging.disable_progress_bar()
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                quant = os.environ.get("STVR_LOCAL_QUANT", "none").lower()
                kwargs = {"device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True}
                if quant in {"4bit", "int4"}:
                    from transformers import BitsAndBytesConfig
                    import torch
                    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
                elif quant in {"8bit", "int8"}:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    kwargs["torch_dtype"] = "auto"
                model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
                if adapter:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, adapter)
                model.eval()
                LOCALPIPE.update({"model": model, "tokenizer": tokenizer, "path": path, "adapter": adapter})
                refreshruntime()
            tokenizer = LOCALPIPE["tokenizer"]
            model = LOCALPIPE["model"]
            messages = [{"role": "system", "content": "You generate executable Python unit tests only."}, {"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = "System: You generate executable Python unit tests only.\nUser: " + prompt + "\nAssistant:"
            inputs = tokenizer([text], return_tensors="pt")
            device = getattr(model, "device", None)
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            import torch
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=int(os.environ.get("STVR_LOCAL_MAXTOKENS", "640")), do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = output[0][inputs["input_ids"].shape[-1]:]
            refreshruntime()
            return tokenizer.decode(generated, skip_special_tokens=True)
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda error" in msg:
            unloadlocal()
            raise ValueError("GPU 推理失败，已自动释放模型缓存。请关闭其他显存进程、降低 STVR_LOCAL_MAXTOKENS，或改用沙箱生成。")
        raise


def generatevalid(prompt, cfg):
    generate = (lambda value: localgenerate(value, cfg)) if cfg["active"] == "local" else (lambda value: openaigenerate(value, cfg))
    raw = generate(prompt)
    try:
        return recovercode(raw), raw
    except SyntaxError:
        retry = (
            "The previous response was truncated. Return a compact, complete Python test file with at most six tests. "
            "Use no markdown and stop immediately after the final test.\n\n" + prompt
        )
        raw = generate(retry)
        return recovercode(raw), raw


def llmgenerate(files, target, func, framework, requested, targetname=None, adapter=None):
    if requested == "off":
        return None
    cfg = modelconfig()
    if requested == "llm" and cfg["active"] == "none":
        raise ValueError("未配置真实大模型，请设置 STVR_LLM_URL/STVR_LLM_KEY 或启用 STVR_LOCAL_ENABLED=1")
    if cfg["active"] == "none":
        return None
    prompt = promptfor(files, target, func, framework, targetname, adapter)
    code, raw = generatevalid(prompt, cfg)
    code = attachadapter(code, adapter)
    ast.parse(code)
    return {"code": code, "provider": cfg["active"], "model": (cfg["localadapter"] or cfg["local"]) if cfg["active"] == "local" else cfg["model"], "raw": raw[:2000]}


def repairgenerate(files, target, func, framework, badcode, logs, adapter=None):
    cfg = modelconfig()
    if cfg["active"] == "none":
        return None
    chunks = []
    for item in files[:8]:
        content = item["content"]
        if len(content) > 4000:
            content = content[:4000] + "\n# ... truncated ..."
        chunks.append("# file: " + item["path"] + "\n" + content)
    prompt = (
        "You are STVR, a Python unit-test repair model.\n"
        "The previous generated " + framework + " test failed. Repair the test using the target code and failure log.\n"
        "Requirements:\n"
        "- Return only corrected Python test code, no markdown explanation.\n"
        "- Keep imports executable in the same project layout.\n"
        "- Remove incorrect expectations instead of changing the target behavior.\n"
        "- Add boundary and exception assertions when they are supported by the target code.\n\n"
        "Project code:\n" + "\n\n".join(chunks) +
        "\n\nFailed test code:\n" + badcode[:5000] +
        "\n\nFailure log:\n" + logs[-2200:]
    )
    code, raw = generatevalid(prompt, cfg)
    code = attachadapter(code, adapter)
    ast.parse(code)
    return {"code": code, "provider": cfg["active"], "model": (cfg["localadapter"] or cfg["local"]) if cfg["active"] == "local" else cfg["model"], "raw": raw[:2000]}


def prunefailingasserts(testcode, logs):
    badlines = {int(match.group(1)) for match in re.finditer(r"testtarget\.py:(\d+):", logs or "")}
    if not badlines:
        return None

    class FailurePruner(ast.NodeTransformer):
        def __init__(self):
            self.changed = False

        def visit_Assert(self, node):
            if node.lineno in badlines:
                self.changed = True
                return None
            return node

        def visit_Expr(self, node):
            call = node.value
            if node.lineno in badlines and isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr.startswith("assert"):
                self.changed = True
                return None
            return self.generic_visit(node)

        def visit_With(self, node):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            source = ast.unparse(node.items[0].context_expr) if node.items else ""
            if any(start <= line <= end for line in badlines) and "pytest.raises" in source:
                self.changed = True
                return None
            return self.generic_visit(node)

    tree = ast.parse(testcode)
    pruner = FailurePruner()
    tree = pruner.visit(tree)
    if not pruner.changed:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip() + "\n"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("create table if not exists users(name text primary key, password text not null, role text not null, created text not null)")
    conn.execute("create table if not exists sessions(token text primary key, user text not null, role text not null, project text not null, created text not null)")
    conn.execute("create table if not exists evaluations(id integer primary key autoincrement, user text, project text, role text, target text, mode text, framework text, status text, score integer, passed integer, line real, branch real, mutation real, asserts integer, failures integer, payload text, result text, created text)")
    columns = {row["name"] for row in conn.execute("pragma table_info(evaluations)")}
    if "mutationvalid" not in columns:
        conn.execute("alter table evaluations add column mutationvalid integer not null default 1")
        for row in conn.execute("select id,passed,result from evaluations").fetchall():
            valid = False
            try:
                metrics = json.loads(row["result"] or "{}").get("metrics", {})
                valid = bool(row["passed"]) and int(metrics.get("mutants") or 0) > 0
            except Exception:
                valid = False
            conn.execute("update evaluations set mutationvalid=? where id=?", (1 if valid else 0, row["id"]))
    for name, info in USERS.items():
        conn.execute("insert or ignore into users values(?,?,?,?)", (name, digest(info["password"]), info["role"], now()))
    conn.commit()
    return conn


def reply(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    try:
        handler.wfile.write(data)
    except (BrokenPipeError, ConnectionResetError):
        pass


def replybytes(handler, status, data, content, filename=None):
    handler.send_response(status)
    handler.send_header("Content-Type", content)
    handler.send_header("Content-Length", str(len(data)))
    if filename:
        handler.send_header("Content-Disposition", "attachment; filename=" + filename)
    handler.end_headers()
    try:
        handler.wfile.write(data)
    except (BrokenPipeError, ConnectionResetError):
        pass


def queryparts(path):
    parsed = urllib.parse.urlparse(path)
    query = urllib.parse.parse_qs(parsed.query)
    return parsed.path, {key: values[-1] for key, values in query.items()}


def readjson(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(min(length, LIMIT + 4096)).decode("utf-8")
    return json.loads(body or "{}")


def login(payload):
    username = str(payload.get("user") or "").strip()
    password = str(payload.get("password") or "")
    project = str(payload.get("project") or "stvr").strip() or "stvr"
    conn = db()
    row = conn.execute("select * from users where name=?", (username,)).fetchone()
    if not row or row["password"] != digest(password):
        return {"ok": False, "error": "账号或密码错误"}
    token = secrets.token_urlsafe(24)
    conn.execute("insert into sessions values(?,?,?,?,?)", (token, username, row["role"], project, now()))
    conn.commit()
    return {"ok": True, "token": token, "user": username, "role": row["role"], "project": project}


def register(payload):
    username = str(payload.get("user") or "").strip()
    password = str(payload.get("password") or "")
    confirm = str(payload.get("confirm") or "")
    role = str(payload.get("role") or "开发者").strip() or "开发者"
    project = str(payload.get("project") or "stvr").strip() or "stvr"
    if not re.match(r"^[A-Za-z0-9]{3,32}$", username):
        return {"ok": False, "error": "账号只能使用 3-32 位字母或数字"}
    if len(password) < 5:
        return {"ok": False, "error": "密码至少 5 位"}
    if confirm and confirm != password:
        return {"ok": False, "error": "两次密码不一致"}
    if role not in {"开发者", "测试工程师"}:
        role = "开发者"
    conn = db()
    row = conn.execute("select name from users where name=?", (username,)).fetchone()
    if row:
        return {"ok": False, "error": "账号已存在"}
    conn.execute("insert into users values(?,?,?,?)", (username, digest(password), role, now()))
    token = secrets.token_urlsafe(24)
    conn.execute("insert into sessions values(?,?,?,?,?)", (token, username, role, project, now()))
    conn.commit()
    return {"ok": True, "token": token, "user": username, "role": role, "project": project}


def session(token):
    if not token:
        return {"user": "guest", "role": "访客", "project": "stvr"}
    conn = db()
    row = conn.execute("select * from sessions where token=?", (token,)).fetchone()
    if not row:
        raise ValueError("登录已失效，请重新登录")
    return {"user": row["user"], "role": row["role"], "project": row["project"]}


def safeimports(tree, localmods, trusted=False):
    allowed = ALLOWED_MODULES | ({"asyncio"} if trusted else set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed and top not in localmods:
                    raise ValueError("不允许导入模块：" + alias.name)
        if isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            top = (node.module or "").split(".")[0]
            if top not in allowed and top not in localmods:
                raise ValueError("不允许导入模块：" + str(node.module))
            if not trusted and top in SAFE_MODULE_ATTRS:
                if any(alias.name not in SAFE_MODULE_ATTRS[top] for alias in node.names):
                    raise ValueError("不允许从模块导入该名称：" + str(node.module))
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError("检测到不安全名称：" + node.id)
        if isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_NAMES or node.attr in BLOCKED_ATTRS:
                raise ValueError("检测到不安全属性：" + node.attr)
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if not trusted and isinstance(root, ast.Name) and root.id in SAFE_MODULE_ATTRS:
                if node.attr not in SAFE_MODULE_ATTRS[root.id]:
                    raise ValueError("不允许调用模块属性：" + root.id + "." + node.attr)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                raise ValueError("检测到不安全调用：" + node.func.id)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_NAMES or node.func.attr in BLOCKED_ATTRS:
                    raise ValueError("检测到不安全调用属性：" + node.func.attr)
                if not trusted and isinstance(node.func.value, ast.Name) and node.func.value.id in BLOCKED_MODULES:
                    if node.func.attr not in SAFE_MODULE_ATTRS.get(node.func.value.id, set()):
                        raise ValueError("检测到不安全模块调用：" + node.func.value.id)
def decoratorname(node):
    value = node.func if isinstance(node, ast.Call) else node
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return ""


def targetkind(node, classnode=None, nested=False):
    asynchronous = isinstance(node, ast.AsyncFunctionDef)
    internal = node.name.startswith("_")
    if nested:
        return "asyncnested" if asynchronous else "nested"
    if classnode is not None:
        decorators = {decoratorname(item) for item in node.decorator_list}
        style = "static" if "staticmethod" in decorators else "class" if "classmethod" in decorators else "instance"
        prefix = "async" if asynchronous else ""
        return prefix + ("internalmethod" if internal else "method") + ":" + style
    if internal:
        return "asyncinternal" if asynchronous else "internal"
    return "async" if asynchronous else "function"


def discoverfunctions(source, localmods=None, checksafe=True):
    tree = ast.parse(source)
    if checksafe:
        safeimports(tree, localmods or set())
    targets = []

    def walk(body, scopes=None, classnode=None, callableowner=None):
        scopes = scopes or []
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, scopes + [node.name], node, callableowner)
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for attr in ("body", "orelse", "finalbody"):
                    branch = getattr(node, attr, None)
                    if isinstance(branch, list):
                        walk(branch, scopes, classnode, callableowner)
                for handler in getattr(node, "handlers", []):
                    walk(getattr(handler, "body", []), scopes, classnode, callableowner)
                for case in getattr(node, "cases", []):
                    walk(getattr(case, "body", []), scopes, classnode, callableowner)
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            name = ".".join(scopes + [node.name])
            nested = callableowner is not None
            item = {
                "name": name,
                "leaf": node.name,
                "line": getattr(node, "lineno", 1),
                "kind": targetkind(node, classnode, nested),
                "async": isinstance(node, ast.AsyncFunctionDef),
                "internal": node.name.startswith("_"),
                "effective": callableowner["effective"] if nested else name,
                "direct": not nested,
                "_node": node,
                "_classnode": classnode,
                "_classname": scopes[-1] if classnode is not None and scopes else "",
            }
            targets.append(item)
            owner = callableowner or item
            walk(node.body, scopes + [node.name], classnode, owner)

    walk(tree.body)
    return targets, tree


def resolvetarget(source, localmods, name=None):
    targets, tree = discoverfunctions(source, localmods)
    if not targets:
        raise ValueError("未检测到可评估的 Python 函数")
    selected = next((item for item in targets if item["name"] == name), None)
    if selected is None and name:
        matches = [item for item in targets if item["leaf"] == name]
        selected = matches[0] if len(matches) == 1 else None
    if selected is None:
        selected = targets[0]
    effective = next((item for item in targets if item["name"] == selected["effective"]), selected)
    return selected, effective, tree


def targetjson(item):
    return {key: value for key, value in item.items() if not key.startswith("_")}


def projectfunctions(files):
    targets = []
    mods = localmodules(files)
    for item in files:
        if istestpath(item["path"]):
            continue
        try:
            functions, _tree = discoverfunctions(item["content"], mods)
        except Exception:
            continue
        for function in functions:
            if function["leaf"].startswith("test_"):
                continue
            target = targetjson(function)
            target["path"] = item["path"]
            targets.append(target)
    return targets


def cleanpath(path):
    raw = str(path or "target.py").replace("\\", "/").strip("/")
    if not raw or raw.startswith(".") or ".." in raw or not raw.endswith(".py"):
        raise ValueError("文件路径不合法：" + str(path))
    rawparts = [p for p in raw.split("/") if p]
    if len(rawparts) > 6:
        raise ValueError("文件路径层级过深：" + str(path))
    parts = [re.sub(r"[^A-Za-z0-9_.-]", "", p) for p in rawparts]
    if not parts or any(not p for p in parts):
        raise ValueError("文件路径不合法：" + str(path))
    return "/".join(parts[:6])


def istestpath(path):
    parts = cleanpath(path).split("/")
    name = parts[-1].lower()
    return name.startswith("test_") or name.endswith("_test.py") or any(part.lower() in {"test", "tests"} for part in parts[:-1])


def importedtops(files):
    names = set()
    for item in files:
        try:
            tree = ast.parse(item["content"])
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names.add(node.module.split(".")[0])
    return names


def stripprojectroot(files):
    if not files:
        return files, ""
    parts = [cleanpath(item["path"]).split("/") for item in files]
    if any(len(item) < 2 for item in parts):
        return files, ""
    root = parts[0][0]
    if any(item[0] != root for item in parts) or root in importedtops(files):
        return files, ""
    normalized = [{"path": "/".join(pathparts[1:]), "content": item["content"]} for item, pathparts in zip(files, parts)]
    return normalized, root


def moduleof(path):
    return cleanpath(path)[:-3].replace("/", ".")


def loadercode(path):
    target = cleanpath(path)
    parts = target[:-3].split("/")
    directories = parts[:-1]
    aliases = ["_stvrproject"] + ["p" + str(index) for index in range(len(directories))]
    lines = [
        "import importlib.util as _stvr_importlib",
        "import sys as _stvr_sys",
        "import types as _stvr_types",
        "from pathlib import Path as _STVRPath",
        "_STVR_ROOT = _STVRPath(__file__).resolve().parent",
    ]
    for index in range(len(aliases)):
        fullname = ".".join(aliases[: index + 1])
        directory = "/".join(directories[:index])
        location = "_STVR_ROOT" if not directory else "_STVR_ROOT / " + repr(directory)
        lines.extend([
            "if " + repr(fullname) + " not in _stvr_sys.modules:",
            "    _stvr_package = _stvr_types.ModuleType(" + repr(fullname) + ")",
            "    _stvr_package.__path__ = [str(" + location + ")]",
            "    _stvr_package.__package__ = " + repr(fullname),
            "    _stvr_sys.modules[" + repr(fullname) + "] = _stvr_package",
        ])
    modulename = ".".join(aliases + ["target"])
    lines.extend([
        "_stvr_spec = _stvr_importlib.spec_from_file_location(" + repr(modulename) + ", _STVR_ROOT / " + repr(target) + ")",
        "if _stvr_spec is None or _stvr_spec.loader is None:",
        "    raise ImportError(\"无法加载目标模块\")",
        "module = _stvr_importlib.module_from_spec(_stvr_spec)",
        "_stvr_sys.modules[_stvr_spec.name] = module",
        "_stvr_spec.loader.exec_module(module)",
    ])
    return chr(10).join(lines) + chr(10)


def localmodules(files):
    mods = set()
    for item in files:
        path = cleanpath(item["path"])
        parts = path[:-3].split("/")
        mods.add(parts[0])
        mods.add(".".join(parts))
    return mods


def argumentvalue(arg):
    annotation = ast.unparse(arg.annotation).lower() if getattr(arg, "annotation", None) is not None else ""
    name = arg.arg.lower()
    if any(token in annotation for token in ["str", "text"]) or any(token in name for token in ["name", "text", "path", "key", "label"]):
        return '""'
    if any(token in annotation for token in ["bool"]) or name.startswith(("is", "has", "allow", "enable")):
        return "False"
    if any(token in annotation for token in ["list", "sequence", "iterable"]) or name.endswith(("items", "values", "rows")):
        return "[]"
    if "dict" in annotation or "mapping" in annotation or name.endswith(("mapping", "options", "config")):
        return "{}"
    if any(token in annotation for token in ["float"]) or any(token in name for token in ["ratio", "rate", "score"]):
        return "0.0"
    if any(token in annotation for token in ["int", "number"]) or any(token in name for token in ["count", "size", "index", "num", "age"]):
        return "0"
    return "None"


def constructorplans(classnode):
    init = next((node for node in classnode.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"), None)
    positional = []
    keyword = []
    if init is not None:
        args = list(init.args.posonlyargs) + list(init.args.args)
        if args and args[0].arg in {"self", "cls"}:
            args = args[1:]
        required = max(0, len(args) - len(init.args.defaults))
        positional = args[:required]
        keyword = [arg for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults) if default is None]
    elif "dataclass" in {decoratorname(item) for item in classnode.decorator_list}:
        for node in classnode.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is None:
                holder = ast.arg(arg=node.target.id, annotation=node.annotation)
                positional.append(holder)
    preferred = [argumentvalue(arg) for arg in positional]
    preferredkw = {arg.arg: argumentvalue(arg) for arg in keyword}
    variants = [
        (preferred, preferredkw),
        (["0"] * len(positional), {arg.arg: "0" for arg in keyword}),
        (['""'] * len(positional), {arg.arg: '""' for arg in keyword}),
        (["None"] * len(positional), {arg.arg: "None" for arg in keyword}),
        (["[]"] * len(positional), {arg.arg: "[]" for arg in keyword}),
    ]
    plans = []
    seen = set()
    for args, kwargs in variants:
        key = (tuple(args), tuple(sorted(kwargs.items())))
        if key not in seen:
            seen.add(key)
            plans.append((args, kwargs))
    return plans


def targetadapter(module, target, path=None):
    asynchronous = bool(target.get("async"))
    classnode = target.get("_classnode")
    path = path or module.replace(".", "/") + ".py"
    lines = ["from _stvrloader import module as _STVR_MODULE"]
    if classnode is None:
        lines.append("_stvr_target = _STVR_MODULE." + target["leaf"])
        expression = "_stvr_target(*args)"
        label = target["leaf"]
    else:
        classname = target.get("_classname") or classnode.name
        lines.append("_STVRClass = _STVR_MODULE." + classname)
        style = target["kind"].split(":")[-1] if ":" in target["kind"] else "instance"
        if style in {"static", "class"}:
            expression = "_STVRClass." + target["leaf"] + "(*args)"
        else:
            plans = []
            for args, kwargs in constructorplans(classnode):
                argtext = "(" + ", ".join(args) + ("," if len(args) == 1 else "") + ")"
                kwtext = "{" + ", ".join(repr(key) + ": " + value for key, value in kwargs.items()) + "}"
                plans.append("(" + argtext + ", " + kwtext + ")")
            lines.extend([
                "_STVR_CONSTRUCTORS = [" + ", ".join(plans) + "]",
                "def _stvr_instance():",
                "    error = None",
                "    for constructor_args, constructor_kwargs in _STVR_CONSTRUCTORS:",
                "        try:",
                "            return _STVRClass(*constructor_args, **constructor_kwargs)",
                "        except TypeError as exc:",
                "            error = exc",
                "    raise error or TypeError('无法构造目标类')",
            ])
            expression = "_stvr_instance()." + target["leaf"] + "(*args)"
        label = classname + "." + target["leaf"]
    if asynchronous:
        lines.extend([
            "import asyncio",
            "def _stvr_call(*args):",
            "    return asyncio.run(" + expression + ")",
        ])
    else:
        lines.extend([
            "def _stvr_call(*args):",
            "    return " + expression,
        ])
    return {"preamble": "\n".join(lines), "call": "_stvr_call", "label": label, "async": asynchronous, "loader": loadercode(path)}


def targetlabel(item):
    kind = item.get("kind", "function")
    if "nested" in kind:
        return "异步嵌套函数" if kind.startswith("async") else "嵌套函数"
    if "method" in kind:
        prefix = "异步" if kind.startswith("async") else ""
        style = kind.split(":")[-1] if ":" in kind else "instance"
        return prefix + {"static": "静态方法", "class": "类方法", "instance": "实例方法"}.get(style, "类方法")
    if "internal" in kind:
        return "异步内部函数" if kind.startswith("async") else "内部函数"
    return "异步函数" if kind == "async" else "普通函数"


def normalize(payload):
    files = payload.get("files") or []
    code = str(payload.get("code") or "")
    if not files and code.strip():
        files = [{"path": "target.py", "content": code}]
    if not files:
        raise ValueError("请提供 Python 函数或小项目文件")
    if len(files) > MAXFILES:
        raise ValueError("文件数量过多，当前限制为 " + str(MAXFILES) + " 个 Python 文件")
    out = []
    total = 0
    for item in files:
        path = cleanpath(item.get("path"))
        content = str(item.get("content") or "")
        if not content.strip():
            continue
        if len(content) > LIMIT // 2:
            raise ValueError("单个文件过长：" + path)
        total += len(content)
        if total > LIMIT:
            raise ValueError("项目代码过长，当前限制为 120000 字符")
        out.append({"path": path, "content": content})
    if not out:
        raise ValueError("未读取到有效 Python 文件")
    out, strippedroot = stripprojectroot(out)
    mods = localmodules(out)
    for item in out:
        tree = ast.parse(item["content"])
        safeimports(tree, mods)
    requested = payload.get("target")
    target = cleanpath(requested or out[0]["path"])
    if strippedroot and target.startswith(strippedroot + "/"):
        target = target[len(strippedroot) + 1:]
    if target not in {item["path"] for item in out}:
        target = out[0]["path"]
    return out, target


def signals(source, name):
    text = (source + " " + name).lower()
    out = []
    if re.search(r"str|text|string|match|char|regex", text):
        out.append("字符串语义")
    if re.search(r"list|tuple|dict|set|items|collection", text):
        out.append("集合结构")
    if re.search(r"int|num|math|sum|count|range|len", text) or re.search(r"return[^\n]*[+*/-]", text):
        out.append("数值边界")
    if re.search(r"raise|except|typeerror|valueerror", text):
        out.append("异常路径")
    if re.search(r"import|from ", text):
        out.append("依赖调用")
    return out or ["通用函数"]


def casesfor(source, func, count):
    argc = len([arg for arg in list(func.args.posonlyargs) + list(func.args.args) if arg.arg not in {"self", "cls"}])
    text = (source + " " + func.name).lower()
    if argc <= 0:
        return [()]
    if re.search(r"list|tuple|items|dict|set", text):
        atoms = [(), ([],), ([1], "x", []), ([1, 2], [3]), ("a", 1, None), ([0],)]
    elif re.search(r"str|text|string|match|char|regex", text):
        atoms = ["", "a", "ab", "abb", "bbb", "aaab", "hello", "A1"]
    elif re.search(r"int|num|math|sum|count|range|len", text) or re.search(r"return[^\n]*[+*/-]", text):
        atoms = [0, 1, 2, 5, -1, 10, 100]
    else:
        atoms = [None, 0, 1, "", "sample", [], [1, 2]]
    if argc == 1:
        return [(item,) for item in atoms[: max(3, count)]]
    combos = list(itertools.product(atoms[:5], repeat=min(argc, 3)))
    return [tuple(item) for item in combos[: max(6, count + 2)]]


def run(cmd, cwd, timeout=None):
    cfg = sandboxconfig()
    timeout = timeout or cfg["timeout"]
    cwd = Path(cwd).resolve()
    if not cwd.exists() or not cwd.is_dir():
        raise ValueError("沙箱工作目录不存在")
    docker = dockerconfig()
    if docker["active"]:
        try:
            return subprocess.run(dockerargs(cwd, cmd), cwd=cwd, text=True, capture_output=True, timeout=timeout, env=safeenv())
        except Exception:
            if docker["strict"]:
                raise
    try:
        return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, env=safeenv(), preexec_fn=guardprocess, start_new_session=False)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        return subprocess.CompletedProcess(cmd, 124, stdout, stderr + "\nSTVR sandbox timeout after " + str(timeout) + "s")


def writefiles(tmp, files):
    for item in files:
        path = tmp / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        for parent in path.parents:
            if parent == tmp or not str(parent).startswith(str(tmp)):
                break
            init = parent / "__init__.py"
            if not init.exists() and parent != tmp:
                init.write_text("", encoding="utf-8")
        final = path.resolve()
        if not str(final).startswith(str(tmp.resolve())):
            raise ValueError("文件写入越界：" + item["path"])
        path.write_text(item["content"], encoding="utf-8")


def observe(tmp, adapter, cases):
    script = tmp / "observe.py"
    script.write_text(
        "import json\n" + adapter["preamble"] + "\nCASES = " + repr(cases) + "\nOUT=[]\n"
        "for args in CASES:\n"
        "    try:\n"
        "        value = " + adapter["call"] + "(*args)\n"
        "        OUT.append({'args': repr(args), 'kind': 'return', 'value': repr(value)})\n"
        "    except Exception as exc:\n"
        "        OUT.append({'args': repr(args), 'kind': 'raise', 'value': exc.__class__.__name__})\n"
        "print(json.dumps(OUT, ensure_ascii=False))\n",
        encoding="utf-8",
    )
    proc = run([sys.executable, "-E", "-s", "observe.py"], tmp, timeout=min(5, sandboxconfig()["timeout"]))
    if proc.returncode != 0:
        raise ValueError("目标函数执行失败：" + (proc.stderr or proc.stdout)[-800:])
    return json.loads(proc.stdout)


def pytestcode(adapter, targetname, observations):
    safe = re.sub(r"[^A-Za-z0-9_]", "_", targetname)
    lines = ["import pytest", adapter["preamble"], ""]
    for index, item in enumerate(observations, 1):
        lines.append("def test_" + safe + "_case" + str(index) + "():")
        args = item["args"]
        if item["kind"] == "raise":
            exc = item["value"] if item["value"] in BUILTIN_EXCEPTIONS else "Exception"
            lines.append("    with pytest.raises(" + exc + "):")
            lines.append("        " + adapter["call"] + "(*" + args + ")")
        else:
            lines.append("    assert " + adapter["call"] + "(*" + args + ") == " + item["value"])
        lines.append("")
    return "\n".join(lines)


def unittestcode(adapter, targetname, observations):
    safe = re.sub(r"[^A-Za-z0-9_]", "", targetname.title()) or "Target"
    lines = ["import unittest", adapter["preamble"], "", "class Test" + safe + "(unittest.TestCase):"]
    for index, item in enumerate(observations, 1):
        lines.append("    def test_case" + str(index) + "(self):")
        args = item["args"]
        if item["kind"] == "raise":
            exc = item["value"] if item["value"] in BUILTIN_EXCEPTIONS else "Exception"
            lines.append("        with self.assertRaises(" + exc + "):")
            lines.append("            " + adapter["call"] + "(*" + args + ")")
        else:
            lines.append("        self.assertEqual(" + adapter["call"] + "(*" + args + "), " + item["value"] + ")")
        lines.append("")
    lines += ["if __name__ == '__main__':", "    unittest.main()"]
    return "\n".join(lines)


def coverageof(tmp, testfile, targetpath, targetrange=None):
    proc = run([sys.executable, "-E", "-s", "-m", "coverage", "run", "--branch", "-m", "pytest", "-q", testfile], tmp)
    jsonproc = run([sys.executable, "-E", "-s", "-m", "coverage", "json", "-o", "coverage.json"], tmp, timeout=5)
    if jsonproc.returncode != 0 or not (tmp / "coverage.json").exists():
        return proc.returncode, 0.0, 0.0, proc.stdout + proc.stderr
    data = json.loads((tmp / "coverage.json").read_text(encoding="utf-8"))
    key = str(targetpath)
    files = data.get("files", {})
    info = files.get(key) or files.get("./" + key) or next((value for name, value in files.items() if name.endswith(key)), None)
    if info and targetrange:
        start, end = targetrange
        executed = {int(line) for line in info.get("executed_lines", []) if start <= int(line) <= end}
        missing = {int(line) for line in info.get("missing_lines", []) if start <= int(line) <= end}
        statements = executed | missing
        line = len(executed) / len(statements) if statements else 0.0
        executedbranches = [arc for arc in info.get("executed_branches", []) if arc and start <= int(arc[0]) <= end]
        missingbranches = [arc for arc in info.get("missing_branches", []) if arc and start <= int(arc[0]) <= end]
        branchcount = len(executedbranches) + len(missingbranches)
        branch = len(executedbranches) / branchcount if branchcount else line
    else:
        summary = (info or {}).get("summary", data.get("totals", {}))
        line = float(summary.get("percent_covered", 0)) / 100.0
        covered = float(summary.get("covered_branches", 0))
        missing = float(summary.get("missing_branches", 0))
        branch = covered / (covered + missing) if covered + missing else line
    return proc.returncode, line, branch, proc.stdout + proc.stderr


def opname(node):
    return node.__class__.__name__.lower()


def collectmutations(tree, limit=36):
    ops = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        if line is None or col is None:
            continue
        if isinstance(node, ast.Compare) and node.ops:
            mapping = {ast.Eq: "NotEq", ast.NotEq: "Eq", ast.Lt: "LtE", ast.LtE: "Lt", ast.Gt: "GtE", ast.GtE: "Gt", ast.Is: "IsNot", ast.IsNot: "Is", ast.In: "NotIn", ast.NotIn: "In"}
            old = type(node.ops[0])
            if old in mapping:
                ops.append({"kind": "compare", "line": line, "col": col, "old": opname(node.ops[0]), "new": mapping[old]})
        elif isinstance(node, ast.BinOp):
            mapping = {ast.Add: "Sub", ast.Sub: "Add", ast.Mult: "Add", ast.Div: "Mult", ast.FloorDiv: "Div", ast.Mod: "Add"}
            old = type(node.op)
            if old in mapping:
                ops.append({"kind": "binop", "line": line, "col": col, "old": opname(node.op), "new": mapping[old]})
        elif isinstance(node, ast.BoolOp):
            mapping = {ast.And: "Or", ast.Or: "And"}
            old = type(node.op)
            if old in mapping:
                ops.append({"kind": "boolop", "line": line, "col": col, "old": opname(node.op), "new": mapping[old]})
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            ops.append({"kind": "unary", "line": line, "col": col, "old": "not", "new": "identity"})
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                ops.append({"kind": "constant", "line": line, "col": col, "old": repr(value), "new": repr(not value)})
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                ops.append({"kind": "constant", "line": line, "col": col, "old": repr(value), "new": repr(value + 1)})
            elif isinstance(value, str) and value:
                ops.append({"kind": "constant", "line": line, "col": col, "old": repr(value), "new": repr(value + "x")})
        elif isinstance(node, ast.Return) and node.value is not None:
            ops.append({"kind": "return", "line": line, "col": col, "old": "value", "new": "None"})
        if len(ops) >= limit:
            break
    return ops


class Mutator(ast.NodeTransformer):
    def __init__(self, spec):
        self.spec = spec
        self.done = False

    def match(self, node, kind):
        return (not self.done and self.spec["kind"] == kind and getattr(node, "lineno", None) == self.spec["line"] and getattr(node, "col_offset", None) == self.spec["col"])

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.match(node, "compare") and node.ops:
            node.ops[0] = getattr(ast, self.spec["new"])()
            self.done = True
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if self.match(node, "binop"):
            node.op = getattr(ast, self.spec["new"])()
            self.done = True
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.match(node, "boolop"):
            node.op = getattr(ast, self.spec["new"])()
            self.done = True
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if self.match(node, "unary"):
            self.done = True
            return node.operand
        return node

    def visit_Constant(self, node):
        if self.match(node, "constant"):
            self.done = True
            try:
                value = ast.literal_eval(self.spec["new"])
            except Exception:
                value = self.spec["new"]
            return ast.copy_location(ast.Constant(value=value), node)
        return node

    def visit_Return(self, node):
        self.generic_visit(node)
        if self.match(node, "return"):
            node.value = ast.Constant(value=None)
            self.done = True
        return node


def mutationvariants(source, targetname=None, limit=36):
    tree = ast.parse(source)
    scope = tree
    if targetname:
        targets, _parsed = discoverfunctions(source, checksafe=False)
        selected = next((item for item in targets if item["name"] == targetname), None)
        if selected is not None:
            scope = selected["_node"]
    variants = []
    seen = set()
    for spec in collectmutations(scope, limit=limit):
        mutated = copy.deepcopy(tree)
        mutator = Mutator(spec)
        mutated = mutator.visit(mutated)
        ast.fix_missing_locations(mutated)
        if not mutator.done:
            continue
        code = ast.unparse(mutated) + "\n"
        if code != source and code not in seen:
            seen.add(code)
            detail = dict(spec)
            detail["operator"] = spec["kind"] + ":" + spec["old"] + "->" + spec["new"]
            variants.append({"source": code, "detail": detail})
    return variants


def mutationscore(tmp, files, target, testfile, targetname=None):
    original = next(item["content"] for item in files if item["path"] == target)
    variants = mutationvariants(original, targetname)
    if not variants:
        return 0, 0, 0.0, []
    killed = 0
    details = []
    path = tmp / target
    for item in variants:
        path.write_text(item["source"], encoding="utf-8")
        proc = run([sys.executable, "-E", "-s", "-m", "pytest", "-q", testfile], tmp, timeout=min(5, sandboxconfig()["timeout"]))
        dead = proc.returncode != 0
        killed += 1 if dead else 0
        detail = item["detail"]
        details.append({"operator": detail["operator"], "line": detail["line"], "killed": dead})
    path.write_text(original, encoding="utf-8")
    return killed, len(variants), killed / max(1, len(variants)), details

def statusof(score, mutation, failures, line, branch, mutationvalid=True):
    if failures:
        return "复核" if score >= 45 and failures <= 3 else "丢弃"
    if mutationvalid and score >= 75 and mutation >= 0.6 and line >= 0.8:
        return "采纳"
    if score >= 55:
        return "复核"
    return "丢弃"


def classify(score, mutation, failures, branch, mutationvalid=True):
    if failures:
        return "执行失败", "先修复测试调用或函数契约问题，再重新评估。"
    if not mutationvalid:
        return "变异不可评价", "目标函数未产生有效变异体，结合覆盖率和断言信息进行复核。"
    if mutation < 0.45:
        return "变异幸存", "补充能区分关键逻辑变化的断言，尤其是反例和边界输入。"
    if branch < 0.55:
        return "分支不足", "补充异常路径、条件分支和临界值输入。"
    if score >= 75 and mutation >= 0.6:
        return "质量稳定", "可进入推荐测试集，建议结合项目规范复核后落库。"
    return "断言待增强", "增加行为区分度更高的断言，并减少只验证非空/类型的弱断言。"


def runtestcandidate(tmp, files, target, testfile, testcode, targetname=None, progress=None, targetrange=None):
    writefiles(tmp, files)
    if len(testcode) > MAXTEST:
        raise ValueError("生成测试过长，已被沙箱拒绝")
    safeimports(ast.parse(testcode), localmodules(files) | {"pytest", "unittest"}, trusted=True)
    (tmp / testfile).write_text(testcode, encoding="utf-8")
    if progress:
        progress("execute", "执行测试并采集覆盖率")
    code, line, branch, logs = coverageof(tmp, testfile, target, targetrange)
    failures = 0 if code == 0 else 1
    if code == 0:
        if progress:
            progress("evidence", "运行目标级变异测试")
        killed, total, mutation, mutationdetails = mutationscore(tmp, files, target, testfile, targetname)
    else:
        if progress:
            progress("evidence", "基础测试未通过，跳过变异测试")
        killed, total, mutation, mutationdetails = 0, 0, 0.0, []
    mutationvalid = code == 0 and total > 0
    asserts = testcode.count("assert ") + testcode.count("pytest.raises") + testcode.count("assertEqual") + testcode.count("assertRaises")
    score = round(max(20, min(99, mutation * 42 + line * 22 + branch * 18 + min(asserts, 12) * 1.2 - failures * 18)))
    status = statusof(score, mutation, failures, line, branch, mutationvalid)
    category, fix = classify(score, mutation, failures, branch, mutationvalid)
    audit = auditlogs(logs)
    return {"code": code, "line": line, "branch": branch, "logs": audit["log"], "audit": audit, "killed": killed, "total": total, "mutation": mutation, "mutationvalid": mutationvalid, "mutationdetails": mutationdetails, "failures": failures, "asserts": asserts, "score": score, "status": status, "category": category, "fix": fix}


def evaluate(payload, sess, progress=None):
    start = time.time()

    def notify(stage, detail):
        if progress:
            progress(stage, detail)

    notify("prepare", "解析代码与目标函数")
    files, target = normalize(payload)
    localmods = localmodules(files)
    targetsource = next(item["content"] for item in files if item["path"] == target)
    selected, effective, _tree = resolvetarget(targetsource, localmods, payload.get("function"))
    func = effective["_node"]
    targetrange = (int(getattr(func, "lineno", 1)), int(getattr(func, "end_lineno", getattr(func, "lineno", 1))))
    count = int(payload.get("count", 6))
    level = int(payload.get("level", 3))
    framework = str(payload.get("framework") or "pytest").lower()
    if framework not in {"pytest", "unittest"}:
        framework = "pytest"
    engine = str(payload.get("engine") or "auto").lower()
    if engine not in {"auto", "llm", "off"}:
        engine = "auto"
    module = moduleof(target)
    adapter = targetadapter(module, effective, target)
    runtimefiles = files + [{"path": "_stvrloader.py", "content": adapter["loader"]}]
    trustedtestmodules = localmods | {"pytest", "unittest", "_stvrloader"}
    mode = "project" if len(files) > 1 else "function"
    cases = casesfor(targetsource, func, count + level)
    llmerror = ""
    nestednote = ""
    if not selected["direct"]:
        nestednote = "嵌套函数通过最近的外层可调用接口 " + effective["name"] + " 进行行为验证。"
    with tempfile.TemporaryDirectory(prefix="stvr") as td:
        base = Path(td)
        writefiles(base, runtimefiles)
        notify("generate", "观察目标行为并构造测试候选")
        observations = observe(base, adapter, cases)
        fallback = unittestcode(adapter, selected["name"], observations) if framework == "unittest" else pytestcode(adapter, selected["name"], observations)
        specs = []
        try:
            detail = "调用模型生成测试" if engine != "off" else "构造确定性测试"
            notify("generate", detail)
            promptname = effective["name"] if selected["direct"] else effective["name"] + "（覆盖嵌套函数 " + selected["name"] + "）"
            generated = llmgenerate(files, target, func, framework, engine, promptname, adapter)
            if generated:
                safeimports(ast.parse(generated["code"]), trustedtestmodules, trusted=True)
                specs.append({"source": "llm", "label": "真实大模型", "model": generated["model"], "code": generated["code"]})
        except Exception as exc:
            llmerror = str(exc)
            if engine == "llm" and modelconfig()["active"] == "none":
                raise
        specs.append({"source": "sandbox", "label": "执行观察", "model": "sandbox", "code": fallback})
        candidates = []
        bestlogs = ""
        for index, spec in enumerate(specs, 1):
            tmp = base / ("run" + str(index))
            tmp.mkdir(parents=True, exist_ok=True)
            result = runtestcandidate(tmp, runtimefiles, target, "testtarget.py", spec["code"], effective["name"], notify, targetrange)
            if spec["source"] == "llm" and result["failures"]:
                try:
                    notify("diagnose", "分析失败日志并修复候选")
                    repaired = repairgenerate(files, target, func, framework, spec["code"], result["logs"], adapter)
                    if repaired:
                        safeimports(ast.parse(repaired["code"]), trustedtestmodules, trusted=True)
                        repairedresult = runtestcandidate(tmp, runtimefiles, target, "testtarget.py", repaired["code"], effective["name"], notify, targetrange)
                        if repairedresult["failures"] == 0 or repairedresult["score"] >= result["score"]:
                            spec = {"source": "llmrepair", "label": "模型修复", "model": repaired["model"], "code": repaired["code"]}
                            result = repairedresult
                    pruned = prunefailingasserts(spec["code"], result["logs"])
                    if pruned:
                        safeimports(ast.parse(pruned), trustedtestmodules, trusted=True)
                        prunedresult = runtestcandidate(tmp, runtimefiles, target, "testtarget.py", pruned, effective["name"], notify, targetrange)
                        if prunedresult["failures"] == 0:
                            spec = {"source": "llmrepair", "label": "执行修复", "model": spec["model"], "code": pruned}
                            result = prunedresult
                except Exception as exc:
                    llmerror = (llmerror + " | " if llmerror else "") + "repair: " + str(exc)
            bestlogs = result["logs"] if not bestlogs or (result["failures"] == 0 and result["score"] >= max([c["score"] for c in candidates if c["failures"] == 0] or [0])) else bestlogs
            reason = "综合 pytest 执行、覆盖率、变异杀伤、断言数量和失败日志得到质量分。失败归因为：" + result["category"] + "。"
            if nestednote:
                reason = nestednote + reason
            if spec["source"].startswith("llm"):
                reason = "由真实大模型生成候选测试，并进入同一执行验证与质量评分流程。" + reason
            candidate = {
                "id": ("llm" if spec["source"].startswith("llm") else "live") + str(int(time.time()))[-6:] + str(index),
                "title": selected["name"],
                "tier": spec["source"],
                "status": result["status"],
                "difficulty": mode,
                "score": result["score"],
                "mutation": result["mutation"],
                "mutationvalid": result["mutationvalid"],
                "killed": result["killed"],
                "mutants": result["total"],
                "mutationdetails": result.get("mutationdetails", []),
                "line": result["line"],
                "branch": result["branch"],
                "boundary": min(1.0, result["asserts"] / 8.0),
                "asserts": result["asserts"],
                "failures": result["failures"],
                "model": spec["model"],
                "engine": spec["source"],
                "risk": "低风险" if result["status"] == "采纳" else "中风险" if result["status"] == "复核" else "高风险",
                "reason": reason,
                "fix": result["fix"],
                "audit": result.get("audit", {}),
                "code": spec["code"],
                "repair": spec["code"],
            }
            candidates.append(candidate)
    notify("diagnose", "汇总缺陷与质量结论")
    candidates.sort(key=lambda item: (item["failures"] == 0, item["score"]), reverse=True)
    best = candidates[0]
    result = {
        "ok": True,
        "mode": mode,
        "framework": framework,
        "engine": engine,
        "llm": {"configured": modelconfig()["available"], "active": modelconfig()["active"], "error": llmerror},
        "profile": {
            "function": selected["name"],
            "callable": effective["name"],
            "kind": targetlabel(selected),
            "direct": selected["direct"],
            "target": selected["name"],
            "file": target,
            "files": len(files),
            "signals": signals(targetsource, selected["name"]),
            "cases": len(cases),
            "note": nestednote,
        },
        "metrics": {"passed": best["failures"] == 0, "line": best["line"], "branch": best["branch"], "mutation": best["mutation"], "mutationvalid": best.get("mutationvalid", False), "killed": best.get("killed", 0), "mutants": best.get("mutants", 0), "mutationdetails": best.get("mutationdetails", []), "asserts": best["asserts"], "duration": round(time.time() - start, 2)},
        "sandbox": sandboxconfig(),
        "security": best.get("audit", auditlogs(bestlogs)),
        "candidates": candidates,
        "logs": auditlogs(bestlogs)["log"],
    }
    notify("report", "保存记录并生成报告")
    result["recordid"] = store(sess, payload, result, best)
    notify("done", "评估完成")
    return result
def store(sess, payload, result, candidate):
    conn = db()
    metrics = result["metrics"]
    cur = conn.execute(
        "insert into evaluations(user,project,role,target,mode,framework,status,score,passed,line,branch,mutation,mutationvalid,asserts,failures,payload,result,created) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sess["user"], sess["project"], sess["role"], result["profile"]["target"], result["mode"], result["framework"], candidate["status"], candidate["score"], 1 if metrics["passed"] else 0, metrics["line"], metrics["branch"], metrics["mutation"], 1 if metrics.get("mutationvalid") else 0, metrics["asserts"], candidate["failures"], json.dumps(payload, ensure_ascii=False), json.dumps(result, ensure_ascii=False), now()),
    )
    conn.commit()
    return cur.lastrowid


def referenceitems(token):
    session(token)
    rows = [
        {
            "id": "case379", "title": "text_match_zero_one", "score": 96, "mutation": 1.0, "line": 1.0, "branch": 1.0, "boundary": 1.0, "asserts": 12, "failures": 0, "model": "domainranker",
            "reason": "高质量参考样例：覆盖正例、反例、空串和重复字符路径，变异、行覆盖和分支覆盖均较充分。",
            "fix": "可作为字符串匹配类函数的参考测试结构。",
            "code": "import pytest\n\ndef test_text_match_zero_one():\n    assert text_match_zero_one('aab') is True\n    assert text_match_zero_one('ab') is True\n    assert text_match_zero_one('a') is False\n    assert text_match_zero_one('cbb') is False\n    assert text_match_zero_one('') is False\n\ndef test_text_match_zero_one_edges():\n    assert text_match_zero_one('abb') is True\n    assert text_match_zero_one('aa') is False\n",
        },
        {
            "id": "case055", "title": "same_chars", "score": 94, "mutation": 1.0, "line": 1.0, "branch": 1.0, "boundary": 1.0, "asserts": 13, "failures": 0, "model": "domainranker",
            "reason": "高质量参考样例：同时覆盖相同字符集合、缺失字符、空串和单字符输入。",
            "fix": "可作为集合等价类函数的参考测试结构。",
            "code": "def test_same_chars_true():\n    assert same_chars('abcd', 'ddddabc') is True\n    assert same_chars('', '') is True\n\ndef test_same_chars_false():\n    assert same_chars('abcd', 'abce') is False\n    assert same_chars('a', 'b') is False\n",
        },
        {
            "id": "case083", "title": "hexagonal_num", "score": 91, "mutation": 1.0, "line": 1.0, "branch": 1.0, "boundary": 1.0, "asserts": 7, "failures": 0, "model": "domainranker",
            "reason": "高质量参考样例：覆盖公式正常值、零边界和类型异常。",
            "fix": "可作为数学公式类函数的参考测试结构。",
            "code": "import pytest\n\ndef test_hexagonal_num_normal():\n    assert hexagonal_num(1) == 1\n    assert hexagonal_num(2) == 6\n    assert hexagonal_num(5) == 45\n\ndef test_hexagonal_num_edge():\n    assert hexagonal_num(0) == 0\n\ndef test_hexagonal_num_type():\n    with pytest.raises(TypeError):\n        hexagonal_num('a')\n",
        },
        {
            "id": "case090", "title": "find_lists", "score": 47, "mutation": 0.0, "line": 0.714286, "branch": 0.5, "boundary": 1.0, "asserts": 3, "failures": 3, "model": "basemodel",
            "reason": "复核参考样例：覆盖了部分正常输入，但变异杀伤不足，异常和混合容器路径需要补强。",
            "fix": "补充非 list 容器、嵌套 list、None 输入和 tuple/list 混合边界。",
            "code": "def test_find_lists_single_list():\n    assert find_lists(([1, 2, 3],)) == 1\n\ndef test_find_lists_multiple_lists():\n    assert find_lists(([1], [2], [])) == 3\n\ndef test_find_lists_empty_tuple():\n    assert find_lists(()) == 0\n",
        },
        {
            "id": "case093", "title": "any_int", "score": 55, "mutation": 0.384615, "line": 0.818182, "branch": 0.75, "boundary": 1.0, "asserts": 5, "failures": 3, "model": "basemodel",
            "reason": "复核参考样例：有一定断言数量，但变异杀伤不足，需要补充互斥条件和反例。",
            "fix": "补充 float、bool、负数和相加关系的互斥输入。",
            "code": "def test_any_int_normal_case():\n    assert any_int(5, 2, 7) is True\n    assert any_int(3, -2, 1) is True\n\ndef test_any_int_no_integer():\n    assert any_int(3.6, -2.2, 2) is False\n",
        },
        {
            "id": "case214", "title": "custom_contract", "score": 34, "mutation": 0.18, "line": 0.52, "branch": 0.21, "boundary": 0.25, "asserts": 2, "failures": 5, "model": "rawprompt",
            "reason": "丢弃参考样例：调用契约不清晰，只检查 happy path，缺少有效行为断言。",
            "fix": "先确认函数输入输出契约，再重新生成边界、异常和分支覆盖测试。",
            "code": "def test_generated_contract():\n    result = target_function(sample_input)\n    assert result is not None\n    assert isinstance(result, object)\n",
        },
    ]
    items = []
    for row in rows:
        status = statusof(row["score"], row["mutation"], row["failures"], row["line"], row["branch"])
        risk = "低风险" if status == "采纳" else "中风险" if status == "复核" else "高风险"
        items.append({**row, "status": status, "risk": risk, "tier": "reference", "difficulty": "reference", "engine": "reference"})
    return {"ok": True, "items": items}


def normalizedrow(row):
    data = dict(row)
    data["status"] = statusof(
        int(data.get("score") or 0),
        float(data.get("mutation") or 0),
        int(data.get("failures") or 0),
        float(data.get("line") or 0),
        float(data.get("branch") or 0),
        bool(int(data.get("mutationvalid", 1) or 0)),
    )
    data["passed"] = 1 if int(data.get("passed") or 0) else 0
    return data


def history(token):
    sess = session(token)
    conn = db()
    rows = conn.execute("select id,user,project,target,mode,framework,status,score,passed,line,branch,mutation,mutationvalid,asserts,failures,created from evaluations where project=? and user=? order by id desc limit 200", (sess["project"], sess["user"])).fetchall()
    return {"ok": True, "items": [normalizedrow(row) for row in rows]}


def parseids(value):
    values = value if isinstance(value, list) else re.split(r"[,\s]+", str(value or ""))
    ids = []
    for item in values:
        try:
            ident = int(item)
        except Exception:
            continue
        if ident > 0 and ident not in ids:
            ids.append(ident)
    return ids[:200]


def deleteevaluations(payload):
    sess = session(payload.get("token") or "")
    ids = parseids(payload.get("ids"))
    if not ids:
        return {"ok": False, "error": "请选择需要删除的评估记录"}
    marks = ",".join("?" for _ in ids)
    conn = db()
    cur = conn.execute(
        "delete from evaluations where project=? and user=? and id in (" + marks + ")",
        [sess["project"], sess["user"], *ids],
    )
    conn.commit()
    return {"ok": True, "deleted": int(cur.rowcount or 0)}


def record(token, ident):
    sess = session(token)
    conn = db()
    row = conn.execute("select * from evaluations where id=? and project=? and user=?", (ident, sess["project"], sess["user"])).fetchone()
    if not row:
        return {"ok": False, "error": "记录不存在"}
    data = normalizedrow(row)
    data["result"] = json.loads(data["result"])
    return {"ok": True, "record": data}


def dashboard(token):
    sess = session(token)
    conn = db()
    rows = [normalizedrow(row) for row in conn.execute("select id,user,project,target,mode,framework,status,score,passed,line,branch,mutation,mutationvalid,asserts,failures,created from evaluations where project=? and user=? order by id desc", (sess["project"], sess["user"])).fetchall()]
    total = len(rows)
    def avg(key):
        return sum(float(row.get(key) or 0) for row in rows) / max(1, total)
    def avgmutation():
        valid = [row for row in rows if row.get("mutationvalid")]
        return sum(float(row.get("mutation") or 0) for row in valid) / max(1, len(valid))
    statuses = []
    for name in ["采纳", "复核", "丢弃"]:
        count = sum(1 for row in rows if row["status"] == name)
        if count:
            statuses.append({"status": name, "count": count})
    target_map = {}
    for row in rows:
        item = target_map.setdefault(row["target"], {"target": row["target"], "count": 0, "score": 0.0})
        item["count"] += 1
        item["score"] += float(row.get("score") or 0)
    targets = sorted(({"target": item["target"], "count": item["count"], "score": item["score"] / max(1, item["count"])} for item in target_map.values()), key=lambda item: (item["count"], item["score"]), reverse=True)[:8]
    recent = [{key: row[key] for key in ["id", "target", "status", "score", "mutation", "mutationvalid", "created"]} for row in rows[:6]]
    return {"ok": True, "project": sess["project"], "total": total, "avg": {"score": round(avg("score"), 1), "line": avg("line"), "branch": avg("branch"), "mutation": avgmutation(), "passed": sum(1 for row in rows if row.get("passed"))}, "statuses": statuses, "targets": targets, "recent": recent}


def rowcount(path):
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="ignore", newline="") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def csvrows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="ignore", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def assets(token):
    session(token)
    base = Path("/root/autodl-tmp/utplm/results")
    construction = base / "construction"
    evaluation = base / "evaluation"
    funnel = [
        {"label": "候选测试", "value": rowcount(construction / "candidatetestpool.csv"), "note": "统一候选池"},
        {"label": "可执行测试", "value": rowcount(construction / "usableexecutionpool.csv"), "note": "pytest 基础验证"},
        {"label": "可评估测试", "value": rowcount(construction / "usableevaluationpool.csv"), "note": "覆盖率与变异评估"},
        {"label": "高质量样例", "value": rowcount(construction / "chosenhighqualitypool.csv"), "note": "高分正例"},
        {"label": "中等质量样例", "value": rowcount(construction / "chosenmediumqualitypool.csv"), "note": "复核样例"},
        {"label": "弱质量样例", "value": rowcount(construction / "chosenweakqualitypool.csv"), "note": "风险样例"},
        {"label": "拒绝配对", "value": rowcount(construction / "rejectedexamplepairtable.csv"), "note": "偏好学习配对"},
        {"label": "筛选配对", "value": rowcount(construction / "surfacecontrolfilteredpairs.csv"), "note": "表面特征控制"},
    ]
    external = csvrows(evaluation / "externalmetrics.csv")
    projects = csvrows(evaluation / "externalprojects.csv")[:12]
    controls = csvrows(evaluation / "attributioncontrolsgroupsummary.csv")
    return {"ok": True, "funnel": funnel, "external": external, "projects": projects, "controls": controls}



def jobprofile(engine):
    cfg = modelconfig()
    active = cfg.get("active", "none")
    if engine == "off" or (engine == "auto" and active == "none"):
        key = "sandbox"
        generation = 3.0
    elif active == "local":
        key = "local"
        generation = 20.0
    else:
        key = "remote"
        generation = 45.0
    stages = [
        {"key": "prepare", "label": "解析代码", "start": 1, "end": 8, "seconds": 2.0},
        {"key": "generate", "label": "生成可信测试", "start": 8, "end": 43, "seconds": generation},
        {"key": "execute", "label": "执行验证", "start": 43, "end": 60, "seconds": 5.0},
        {"key": "evidence", "label": "计算质量证据", "start": 60, "end": 84, "seconds": 18.0},
        {"key": "diagnose", "label": "缺陷诊断", "start": 84, "end": 94, "seconds": 3.0},
        {"key": "report", "label": "生成修复建议", "start": 94, "end": 99, "seconds": 2.0},
    ]
    base = sum(item["seconds"] for item in stages)
    history = JOBTIMES.get(key, [])
    if history:
        expected = sum(history[-6:]) / len(history[-6:])
        scale = max(0.35, min(1.8, expected / max(1.0, base)))
        for item in stages:
            item["seconds"] *= scale
    return key, stages


def cleanupjobs():
    cutoff = time.time() - JOBTTL
    with JOBLOCK:
        stale = [jobid for jobid, job in JOBS.items() if job.get("finished", job.get("started", 0)) < cutoff]
        for jobid in stale:
            JOBS.pop(jobid, None)


def updatejob(jobid, stage, detail):
    with JOBLOCK:
        job = JOBS.get(jobid)
        if not job or job.get("status") != "running":
            return
        keys = [item["key"] for item in job["stages"]]
        current = keys.index(job["stage"]) if job["stage"] in keys else 0
        incoming = keys.index(stage) if stage in keys else current
        if incoming >= current:
            if incoming > current:
                job["stage_started"] = time.time()
            job["stage"] = stage
        job["detail"] = detail


def startjob(payload, sess, token):
    cleanupjobs()
    jobid = secrets.token_hex(10)
    engine = str(payload.get("engine") or "auto").lower()
    key, stages = jobprofile(engine)
    started = time.time()
    job = {
        "id": jobid,
        "token": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "user": sess["user"],
        "project": sess["project"],
        "status": "running",
        "stage": "prepare",
        "detail": "准备评估任务",
        "started": started,
        "stage_started": started,
        "stages": stages,
        "profile": key,
    }
    with JOBLOCK:
        JOBS[jobid] = job

    def worker():
        try:
            result = evaluate(payload, sess, lambda stage, detail: updatejob(jobid, stage, detail))
            finished = time.time()
            with JOBLOCK:
                current = JOBS.get(jobid)
                if current:
                    current.update({"status": "completed", "stage": "done", "detail": "评估完成", "result": result, "finished": finished})
                    values = JOBTIMES.setdefault(key, [])
                    values.append(finished - started)
                    del values[:-12]
        except Exception as exc:
            with JOBLOCK:
                current = JOBS.get(jobid)
                if current:
                    current.update({"status": "failed", "stage": "failed", "detail": "评估失败", "error": str(exc), "finished": time.time()})

    threading.Thread(target=worker, name="stvr-" + jobid[:8], daemon=True).start()
    return {"ok": True, "job": jobid}


def jobstatus(token, jobid):
    sess = session(token)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with JOBLOCK:
        job = JOBS.get(jobid)
        if not job or job.get("token") != digest or job.get("user") != sess["user"] or job.get("project") != sess["project"]:
            raise ValueError("评估任务不存在或无权访问")
        status = job["status"]
        elapsed = max(0.0, time.time() - job["started"])
        if status == "completed":
            return {"ok": True, "status": status, "stage": "done", "label": "评估完成", "detail": job["detail"], "progress": 100, "elapsed": round(elapsed), "eta": 0, "result": job["result"]}
        if status == "failed":
            return {"ok": False, "status": status, "stage": "failed", "label": "评估失败", "detail": job["detail"], "progress": 100, "elapsed": round(elapsed), "eta": 0, "error": job.get("error", "评估失败")}
        stages = job["stages"]
        index = next((i for i, item in enumerate(stages) if item["key"] == job["stage"]), 0)
        current = stages[index]
        stageelapsed = max(0.0, time.time() - job["stage_started"])
        fraction = min(0.92, stageelapsed / max(1.0, current["seconds"]))
        progressvalue = current["start"] + (current["end"] - current["start"]) * fraction
        eta = max(0.0, current["seconds"] - stageelapsed) + sum(item["seconds"] for item in stages[index + 1:])
        return {
            "ok": True,
            "status": status,
            "stage": current["key"],
            "label": current["label"],
            "detail": job["detail"],
            "progress": round(progressvalue),
            "elapsed": round(elapsed),
            "eta": round(eta),
        }


def batch(payload, sess):
    files, _target = normalize(payload)
    targets = projectfunctions(files)
    limit = max(1, min(int(payload.get("limit", 12)), 24))
    engine = str(payload.get("engine") or "off").lower()
    if engine not in {"auto", "llm", "off"}:
        engine = "off"
    framework = str(payload.get("framework") or "pytest").lower()
    items = []
    for item in targets[:limit]:
        target = cleanpath(item.get("path"))
        name = str(item.get("name") or "").strip()
        try:
            result = evaluate({"files": files, "target": target, "function": name, "framework": framework, "engine": engine, "count": int(payload.get("count", 5)), "level": int(payload.get("level", 3))}, sess)
            items.append({"ok": True, "recordid": result["recordid"], "path": target, "function": result["profile"]["function"], "score": result["candidates"][0]["score"], "status": result["candidates"][0]["status"], "passed": result["metrics"]["passed"], "line": result["metrics"]["line"], "mutation": result["metrics"]["mutation"], "mutationvalid": result["metrics"].get("mutationvalid", False), "engine": result["candidates"][0]["engine"]})
        except Exception as exc:
            items.append({"ok": False, "path": target, "function": name, "error": str(exc)})
    done = [item for item in items if item.get("ok")]
    avg = sum(item["score"] for item in done) / max(1, len(done))
    return {"ok": True, "total": len(items), "passed": sum(1 for item in done if item.get("passed")), "avgscore": round(avg, 1), "items": items}


def requireadmin(token):
    sess = session(token)
    if sess["role"] != "管理员":
        raise ValueError("需要管理员权限")
    return sess


def users(token):
    requireadmin(token)
    conn = db()
    rows = conn.execute("select name,role,created from users order by created desc,name").fetchall()
    return {"ok": True, "items": [dict(row) for row in rows]}


def useraction(payload):
    sess = requireadmin(payload.get("token"))
    action = str(payload.get("action") or "save")
    name = str(payload.get("name") or "").strip()
    role = str(payload.get("role") or "开发者").strip() or "开发者"
    if role not in {"开发者", "测试工程师", "管理员"}:
        role = "开发者"
    password = str(payload.get("password") or "")
    if not re.match(r"^[A-Za-z0-9]{3,32}$", name):
        raise ValueError("账号只能使用 3-32 位字母或数字")
    conn = db()
    if action == "delete":
        if name == sess["user"]:
            raise ValueError("不能删除当前登录账号")
        conn.execute("delete from users where name=?", (name,))
        conn.commit()
        return {"ok": True}
    row = conn.execute("select name from users where name=?", (name,)).fetchone()
    if row and not password:
        conn.execute("update users set role=? where name=?", (role, name))
    else:
        if len(password) < 5:
            raise ValueError("密码至少 5 位")
        conn.execute("insert or replace into users values(?,?,?,coalesce((select created from users where name=?),?))", (name, digest(password), role, name, now()))
    conn.commit()
    return {"ok": True}


def uploadzip(handler):
    _path, parts = queryparts(handler.path)
    token = handler.headers.get("X-Token") or parts.get("token")
    session(token)
    ctype = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in ctype:
        raise ValueError("请上传 zip 压缩包")
    form = cgi.FieldStorage(fp=handler.rfile, headers=handler.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype})
    item = form["archive"] if "archive" in form else None
    if item is None or not getattr(item, "file", None):
        raise ValueError("未读取到压缩包")
    data = item.file.read(MAXZIP + 1)
    if len(data) > MAXZIP:
        raise ValueError("压缩包过大")
    files = []
    total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.endswith(".py"):
                continue
            if info.file_size > LIMIT // 2:
                raise ValueError("压缩包内单个文件过大：" + info.filename)
            if info.compress_size and info.file_size / max(1, info.compress_size) > 80:
                raise ValueError("压缩包膨胀比例异常：" + info.filename)
            path = cleanpath(info.filename)
            content = zf.read(info, LIMIT).decode("utf-8", "replace")
            ast.parse(content)
            total += len(content)
            if total > LIMIT:
                raise ValueError("项目代码过长，当前限制为 120000 字符")
            files.append({"path": path, "content": content})
            if len(files) >= MAXFILES:
                break
    if not files:
        raise ValueError("压缩包中未找到 Python 文件")
    files, _strippedroot = stripprojectroot(files)
    mods = localmodules(files)
    for item in files:
        safeimports(ast.parse(item["content"]), mods)
    funcs = projectfunctions(files)
    return {"ok": True, "files": files, "target": (funcs[0]["path"] if funcs else files[0]["path"]), "functions": funcs}


def pdfhex(text):
    clean = str(text).replace("\t", "    ")
    return clean.encode("utf-16-be", "replace").hex().upper()


def pdflatinhex(text):
    return str(text).encode("latin-1", "replace").hex().upper()


def pdfflowlines(lines, width=58):
    out = []
    for raw in lines:
        text = str(raw).rstrip()
        if not text:
            out.append("")
            continue
        while len(text) > width:
            out.append(text[:width])
            text = text[width:]
        out.append(text)
    return out


def pdfruns(line):
    runs = []
    current = ""
    current_latin = None
    for ch in str(line):
        is_latin = ord(ch) < 128
        if current and is_latin != current_latin:
            runs.append((current_latin, current))
            current = ""
        current += ch
        current_latin = is_latin
    if current:
        runs.append((current_latin, current))
    return runs


def simplepdf(lines):
    wrapped = pdfflowlines(lines)
    pages = [wrapped[index:index + 52] for index in range(0, len(wrapped), 52)] or [[""]]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        None,
        b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [4 0 R] >>",
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> /DW 1000 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
    ]
    kids = []
    for page in pages:
        page_id = len(objects) + 1
        content_id = page_id + 1
        kids.append(str(page_id) + " 0 R")
        content = ["BT", "14 TL", "45 800 Td"]
        for line in page:
            for is_latin, part in pdfruns(line):
                if is_latin:
                    content.append("/F2 10 Tf <" + pdflatinhex(part) + "> Tj")
                else:
                    content.append("/F1 10 Tf <" + pdfhex(part) + "> Tj")
            content.append("T*")
        content.append("ET")
        stream = "\n".join(content).encode("ascii")
        resources = "<< /Font << /F1 3 0 R /F2 5 0 R >> >>"
        objects.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources " + resources + " /Contents " + str(content_id) + " 0 R >>").encode("ascii"))
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    objects[1] = ("<< /Type /Pages /Kids [" + " ".join(kids) + "] /Count " + str(len(pages)) + " >>").encode("ascii")
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += str(index).encode("ascii") + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += ("%010d 00000 n \n" % offset).encode("ascii")
    pdf += b"trailer\n<< /Size " + str(len(objects) + 1).encode("ascii") + b" /Root 1 0 R >>\nstartxref\n" + str(xref).encode("ascii") + b"\n%%EOF\n"
    return pdf

def reportlines(row):
    result = json.loads(row["result"])
    metrics = result.get("metrics", {})
    profile = result.get("profile", {})
    best = (result.get("candidates") or [{}])[0]
    payload = json.loads(row["payload"] or "{}")
    code = payload.get("code") or "\n\n".join("# file: " + item.get("path", "") + "\n" + item.get("content", "") for item in payload.get("files", [])[:3])
    mutation = metrics.get("mutationdetails") or best.get("mutationdetails") or []
    mutationvalid = bool(metrics.get("mutationvalid", best.get("mutationvalid", row["mutationvalid"] if "mutationvalid" in row.keys() else False)))
    mutationtext = ("%.2f%%" % (float(metrics.get("mutation", 0)) * 100)) if mutationvalid else "—（不可评价）"
    candidates = result.get("candidates") or []
    normalized = normalizedrow(row)
    status = str(normalized["status"])
    passed = "通过" if metrics.get("passed") else "失败"
    security = str((result.get("security") or {}).get("summary", "日志正常"))
    lines = [
        "STVR 可信测试质量报告",
        "项目：" + str(row["project"]),
        "用户：" + str(row["user"]) + " / " + str(row["role"]),
        "生成时间：" + str(row["created"]),
        "目标文件：" + str(profile.get("target", row["target"])),
        "目标函数：" + str(profile.get("function", "")),
        "输入模式：" + str(row["mode"]) + "  测试框架：" + str(row["framework"]),
        "生成引擎：" + str(result.get("engine", "auto")) + "  候选来源：" + str(best.get("engine", "")),
        "模型：" + str(best.get("model", ""))[-90:],
        "",
        "可信测试：生成候选 " + str(len(candidates)) + " 组，当前推荐样例为 " + str(best.get("id", "")) + " · " + str(best.get("title", "")) + "。",
        "执行验证：状态=" + status + "，pytest=" + passed + "，" + security + "。",
        "质量证据：质量分=" + str(row["score"]) + "，行覆盖=" + ("%.2f%%" % (float(metrics.get("line", 0)) * 100)) + "，分支覆盖=" + ("%.2f%%" % (float(metrics.get("branch", 0)) * 100)) + "，变异分数=" + mutationtext + "，断言数=" + str(metrics.get("asserts", 0)) + "。",
        "缺陷诊断：" + str(best.get("reason", ""))[:180],
        "修复推荐：" + str(best.get("fix", ""))[:180],
        "",
        "输入代码预览：",
    ]
    lines.extend(str(code).splitlines()[:10])
    lines += ["", "推荐测试预览："]
    lines.extend(str(best.get("code", "")).splitlines()[:14])
    lines += ["", "变异明细："]
    for item in mutation[:8]:
        killed = "已杀死" if item.get("killed") else "幸存"
        lines.append(str(item.get("operator", "")) + "  line " + str(item.get("line", "")) + "  " + killed)
    lines += ["", "执行日志预览："]
    lines.extend(str(result.get("logs", "")).splitlines()[-8:])
    return lines

def projectreportlines(token, ids=None):
    sess = session(token)
    conn = db()
    selected = parseids(ids)
    sql = "select id,user,project,target,mode,framework,status,score,passed,line,branch,mutation,mutationvalid,asserts,failures,created from evaluations where project=? and user=?"
    params = [sess["project"], sess["user"]]
    if selected:
        sql += " and id in (" + ",".join("?" for _ in selected) + ")"
        params.extend(selected)
    sql += " order by id desc limit 200"
    rows = [normalizedrow(row) for row in conn.execute(sql, params).fetchall()]
    total = len(rows)
    def avg(key):
        return sum(float(row[key] or 0) for row in rows) / max(1, total)
    passed = sum(1 for row in rows if row["passed"])
    validmutations = [row for row in rows if row.get("mutationvalid")]
    averagemutation = sum(float(row["mutation"] or 0) for row in validmutations) / max(1, len(validmutations))
    status_counts = {}
    target_counts = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        bucket = target_counts.setdefault(row["target"], {"count": 0, "score": 0.0, "mutation": 0.0, "mutationcount": 0})
        bucket["count"] += 1
        bucket["score"] += float(row["score"] or 0)
        if row.get("mutationvalid"):
            bucket["mutation"] += float(row["mutation"] or 0)
            bucket["mutationcount"] += 1
    top_targets = sorted(target_counts.items(), key=lambda item: (item[1]["count"], item[1]["score"] / max(1, item[1]["count"])), reverse=True)[:8]
    lines = [
        "STVR 项目整体评估报告",
        "报告范围：" + ("所选评估记录" if selected else "当前用户全部函数评估记录"),
        "项目：" + str(sess["project"]),
        "用户：" + str(sess["user"]) + " / " + str(sess["role"]),
        "生成时间：" + now(),
        "",
        "可信测试：累计评估 " + str(total) + " 次，覆盖目标文件/函数 " + str(len(target_counts)) + " 个。",
        "执行验证：通过 " + str(passed) + " 次，通过率 " + ("%.2f%%" % ((passed / max(1, total)) * 100)) + "。",
        "质量证据：平均质量分 " + ("%.1f" % avg("score")) + "，平均行覆盖 " + ("%.2f%%" % (avg("line") * 100)) + "，平均分支覆盖 " + ("%.2f%%" % (avg("branch") * 100)) + "，有效记录平均变异分数 " + ("%.2f%%" % (averagemutation * 100)) + "。",
        "缺陷诊断：采纳 " + str(status_counts.get("采纳", 0)) + " 次，复核 " + str(status_counts.get("复核", 0)) + " 次，丢弃 " + str(status_counts.get("丢弃", 0)) + " 次，平均失败数 " + ("%.2f" % avg("failures")) + "。",
        "修复推荐：优先处理复核和丢弃记录，补强低变异、低覆盖和失败日志集中的目标函数。",
        "",
        "重点函数概览：",
    ]
    if not top_targets:
        lines.append("暂无历史评估记录。完成一次评估后，整体报告会自动汇总当前用户的全部函数测试结果。")
    for target, item in top_targets:
        count = item["count"]
        lines.append(str(target) + "：评估 " + str(count) + " 次，平均质量分 " + ("%.1f" % (item["score"] / max(1, count))) + "，平均变异 " + (("%.2f%%" % ((item["mutation"] / item["mutationcount"]) * 100)) if item["mutationcount"] else "—") + "。")
    lines += ["", "最近评估记录："]
    for row in rows[:10]:
        lines.append("#" + str(row["id"]) + "  " + str(row["target"]) + "  " + str(row["status"]) + "  质量分=" + str(row["score"]) + "  变异=" + (("%.2f%%" % (float(row["mutation"] or 0) * 100)) if row.get("mutationvalid") else "—") + "  " + str(row["created"]))
    lines += ["", "报告说明：本报告用于项目级质量汇总；单次评估的输入代码、推荐测试、变异明细和执行日志请在历史记录详情中下载。"]
    return lines


def recordrow(token, ident):
    sess = session(token)
    conn = db()
    row = conn.execute("select * from evaluations where id=? and project=? and user=?", (ident, sess["project"], sess["user"])).fetchone()
    if not row:
        raise ValueError("记录不存在")
    return row


def testbytes(token, ident, candidate=0):
    row = recordrow(token, ident)
    result = json.loads(row["result"])
    candidates = result.get("candidates") or []
    index = max(0, min(int(candidate or 0), len(candidates) - 1)) if candidates else 0
    code = (candidates[index].get("code") if candidates else "") or "# no generated test\n"
    return code.encode("utf-8")



class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        if self.path.endswith((".html", ".js", ".css")) or self.path == "/":
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
        super().end_headers()

    def do_GET(self):
        try:
            path, parts = queryparts(self.path)
            token = parts.get("token") or self.headers.get("X-Token")
            if path == "/api/evaluate/status":
                reply(self, 200, jobstatus(token, parts.get("id", "")))
                return
            if path == "/api/model":
                reply(self, 200, {"ok": True, "config": modelconfig(), "sandbox": sandboxconfig()})
                return
            if path == "/api/dashboard":
                reply(self, 200, dashboard(token))
                return
            if path == "/api/assets":
                reply(self, 200, assets(token))
                return
            if path == "/api/users":
                reply(self, 200, users(token))
                return
            if path == "/api/references":
                reply(self, 200, referenceitems(token))
                return
            if path == "/api/history":
                reply(self, 200, history(token))
                return
            if path == "/api/record":
                reply(self, 200, record(token, int(parts.get("id", "0"))))
                return
            if path == "/api/test":
                replybytes(self, 200, testbytes(token, int(parts.get("id", "0")), parts.get("candidate", 0)), "text/x-python; charset=utf-8", "stvrtest.py")
                return
            if path == "/api/projectreportpdf":
                replybytes(self, 200, simplepdf(projectreportlines(token, parts.get("ids"))), "application/pdf", "report.pdf")
                return
            if path == "/api/reportpdf":
                row = recordrow(token, int(parts.get("id", "0")))
                replybytes(self, 200, simplepdf(reportlines(row)), "application/pdf", "report.pdf")
                return
            super().do_GET()
        except Exception as exc:
            reply(self, 200, {"ok": False, "error": str(exc)})

    def do_POST(self):
        try:
            path, _parts = queryparts(self.path)
            if path == "/api/upload":
                reply(self, 200, uploadzip(self))
                return
            payload = readjson(self)
            if path == "/api/login":
                reply(self, 200, login(payload))
                return
            if path == "/api/register":
                reply(self, 200, register(payload))
                return
            if path == "/api/warmup":
                session(payload.get("token") or self.headers.get("X-Token"))
                reply(self, 200, warmuplocal())
                return
            if path == "/api/unload":
                session(payload.get("token") or self.headers.get("X-Token"))
                reply(self, 200, unloadlocal())
                return
            if path == "/api/scan":
                token = payload.get("token") or self.headers.get("X-Token")
                session(token)
                files, _target = normalize(payload)
                reply(self, 200, {"ok": True, "functions": projectfunctions(files)})
                return
            if path == "/api/evaluate/start":
                token = payload.get("token") or self.headers.get("X-Token")
                sess = session(token)
                reply(self, 200, startjob(payload, sess, token))
                return
            if path == "/api/evaluate":
                sess = session(payload.get("token") or self.headers.get("X-Token"))
                reply(self, 200, evaluate(payload, sess))
                return
            if path == "/api/batch":
                sess = session(payload.get("token") or self.headers.get("X-Token"))
                reply(self, 200, batch(payload, sess))
                return
            if path == "/api/records":
                reply(self, 200, deleteevaluations(payload))
                return
            if path == "/api/users":
                reply(self, 200, useraction(payload))
                return
            reply(self, 404, {"ok": False, "error": "unknown endpoint"})
        except Exception as exc:
            reply(self, 200, {"ok": False, "error": str(exc)})

    def log_message(self, fmt, *args):
        sys.stderr.write("STVR " + (fmt % args) + "\n")


def selftest():
    auth = login({"user": "demo", "password": "demo123", "project": "check"})
    assert auth["ok"]
    one = evaluate({"token": auth["token"], "code": "def add(a, b):\n    return a + b\n", "count": 6, "level": 3, "framework": "pytest", "engine": "off"}, session(auth["token"]))
    assert one["ok"] and one["metrics"]["passed"] and one["candidates"][0]["asserts"] > 0
    project = evaluate({"token": auth["token"], "framework": "unittest", "engine": "off", "files": [
        {"path": "helper.py", "content": "import math\ndef root(x):\n    return math.sqrt(x)\n"},
        {"path": "target.py", "content": "from helper import root\ndef rounded(x):\n    return round(root(x), 2)\n"},
    ], "target": "target.py", "count": 5, "level": 3}, session(auth["token"]))
    collisioncode = "def echo(value):" + chr(10) + "    return value" + chr(10)
    collision = evaluate({"token": auth["token"], "framework": "pytest", "engine": "off", "files": [{"path": "types/functions.py", "content": collisioncode}], "target": "types/functions.py", "function": "echo", "count": 5, "level": 3}, session(auth["token"]))
    assert collision["ok"] and collision["metrics"]["passed"]
    featurecode = "import asyncio\nasync def _wait(x):\n    await asyncio.sleep(0)\n    return x\nclass Tool:\n    @staticmethod\n    def clean(x):\n        return x\n    def outer(self, x):\n        def inner(y):\n            return y\n        return inner(x)\n"
    discovered = projectfunctions([{"path": "feature.py", "content": featurecode}])
    names = {item["name"]: item["kind"] for item in discovered}
    assert names["_wait"] == "asyncinternal"
    assert names["Tool.clean"] == "method:static"
    assert names["Tool.outer.inner"] == "nested"
    hist = history(auth["token"])
    assert project["ok"] and project["mode"] == "project" and hist["items"]
    print(json.dumps({"ok": True, "functionscore": one["candidates"][0]["score"], "projectscore": project["candidates"][0]["score"], "records": len(hist["items"]), "framework": project["framework"], "targets": len(discovered)}, ensure_ascii=False))


def main():
    if "--check" in sys.argv:
        selftest()
        return
    db()
    port = 7860
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("STVR server running at http://127.0.0.1:" + str(port) + "/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
