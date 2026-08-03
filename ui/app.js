const state = {
  selected: 0,
  filter: "all",
  activeTarget: 0,
  lastRun: null,
  overrides: {},
  session: { user: "demo", project: "stvr", role: "开发者" },
  live: null,
  targetfunction: "",
  running: false,
  token: "",
  batchfiles: [],
  batchsource: "",
  history: [],
  target: "target.py",
  model: null,
  dashboard: null,
  users: [],
  record: null,
  assets: null,
  references: [],
  batchtargets: [],
  batch: null,
  authmode: "login",
  selectedRecords: new Set(),
};

const DATA = {
  funnel: [
    ["候选测试", 2952, 2952, "统一候选池"],
    ["可执行测试", 962, 2952, "执行筛选"],
    ["可评估测试", 947, 2952, "完整质量证据"],
    ["质量分层样例", 144, 144, "高、中、弱三层"],
    ["偏好配对", 996, 996, "程序化偏好构造"],
    ["筛选偏好配对", 284, 996, "表面特征控制"],
    ["外部函数", 89, 89, "6个公开项目"],
  ],
  flow: [
    ["可信测试", "生成可执行测试候选"],
    ["执行验证", "运行 pytest 或 unittest"],
    ["质量证据", "统计覆盖率和变异分数"],
    ["缺陷诊断", "定位断言、边界和异常路径风险"],
    ["修复推荐", "输出改进测试和评估报告"],
  ],
  evidence: [
    ["可信测试", "生成多组测试候选，避免直接相信单次模型输出"],
    ["执行验证", "运行 pytest 或 unittest，记录通过状态和失败日志"],
    ["质量证据", "汇总覆盖率、变异分数、断言和边界信号"],
    ["缺陷诊断", "识别弱断言、边界缺失、契约错误和异常路径缺口"],
    ["修复推荐", "给出可下载的改进测试和报告结论"],
  ],
  targets: [
    {
      name: "textmatch",
      label: "text match zero one",
      short: "字符串",
      code: "def text_match_zero_one(text):\n    \"\"\"Return True when text contains a followed by at least one b.\"\"\"\n    return isinstance(text, str) and \"ab\" in text",
      domain: "string",
    },
    {
      name: "hexagonal",
      label: "hexagonal number",
      short: "数值",
      code: "def hexagonal_num(n):\n    \"\"\"Return the nth hexagonal number.\"\"\"\n    return n * (2 * n - 1)",
      domain: "math",
    },
    {
      name: "listcount",
      label: "find lists",
      short: "列表",
      code: "def find_lists(items):\n    \"\"\"Count list objects inside a tuple-like input.\"\"\"\n    return sum(1 for item in items if isinstance(item, list))",
      domain: "collection",
    },
  ],
  candidates: [
    {
      id: "case379",
      title: "text_match_zero_one",
      tier: "highquality",
      status: "采纳",
      difficulty: "easy",
      score: 96,
      mutation: 1,
      line: 1,
      branch: 1,
      boundary: 1,
      asserts: 12,
      failures: 0,
      model: "domainranker",
      risk: "低风险",
      reason: "高质量正例：变异、行覆盖、分支覆盖和边界覆盖均为 1.000，失败日志为 0。",
      fix: "可直接进入推荐测试集，并作为相似字符串任务的 few shot 样例。",
      code: "import pytest\n\ndef test_text_match_zero_one():\n    assert text_match_zero_one(\"aab\") == True\n    assert text_match_zero_one(\"ab\") == True\n    assert text_match_zero_one(\"a\") == False\n    assert text_match_zero_one(\"cbb\") == False\n    assert text_match_zero_one(\"acbb\") == False\n    assert text_match_zero_one(\"\") == False\n\ndef test_text_match_zero_one_edge_cases():\n    assert text_match_zero_one(\"aa\") == False\n    assert text_match_zero_one(\"abb\") == True\n    assert text_match_zero_one(\"abbb\") == True\n    assert text_match_zero_one(\"b\") == False\n    assert text_match_zero_one(\"aaab\") == True",
    },
    {
      id: "case055",
      title: "same_chars",
      tier: "highquality",
      status: "采纳",
      difficulty: "medium",
      score: 94,
      mutation: 1,
      line: 1,
      branch: 1,
      boundary: 1,
      asserts: 13,
      failures: 0,
      model: "domainranker",
      risk: "低风险",
      reason: "高质量正例：包含对称性、空串、单字符和长度差异检查，失败日志为 0。",
      fix: "保留对称性与空输入检查，可迁移到集合等价类任务。",
      code: "import pytest\n\ndef test_same_chars_true():\n    assert same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc') == True\n    assert same_chars('abcd', 'dddddddabc') == True\n    assert same_chars('dddddddabc', 'abcd') == True\n\ndef test_same_chars_false():\n    assert same_chars('eabcd', 'dddddddabc') == False\n    assert same_chars('abcd', 'dddddddabce') == False\n\ndef test_empty_strings():\n    assert same_chars('', '') == True\n\ndef test_single_char_strings():\n    assert same_chars('a', 'a') == True\n    assert same_chars('a', 'b') == False",
    },
    {
      id: "case083",
      title: "hexagonal_num",
      tier: "highquality",
      status: "采纳",
      difficulty: "easy",
      score: 91,
      mutation: 1,
      line: 1,
      branch: 1,
      boundary: 1,
      asserts: 7,
      failures: 0,
      model: "domainranker",
      risk: "低风险",
      reason: "高质量正例：覆盖正常输入、零边界和类型异常，变异分数为 1.000。",
      fix: "适合保留为数学公式类函数的推荐模板。",
      code: "import pytest\n\ndef test_hexagonal_num_normal():\n    assert hexagonal_num(1) == 1\n    assert hexagonal_num(2) == 6\n    assert hexagonal_num(3) == 15\n    assert hexagonal_num(4) == 28\n    assert hexagonal_num(5) == 45\n\ndef test_hexagonal_num_edge_case():\n    assert hexagonal_num(0) == 0\n    assert hexagonal_num(10) == 190\n\ndef test_hexagonal_num_type():\n    with pytest.raises(TypeError):\n        hexagonal_num('a')",
    },
    {
      id: "case090",
      title: "find_lists",
      tier: "weakquality",
      status: "复核",
      difficulty: "easy",
      score: 47,
      mutation: 0,
      line: 0.714286,
      branch: 0.5,
      boundary: 1,
      asserts: 3,
      failures: 3,
      model: "basemodel",
      risk: "高风险",
      reason: "弱质量样例：能覆盖部分输入，但 mutation survival major 触发，说明测试对关键逻辑变化不敏感。",
      fix: "补充非 list 容器、嵌套 list、None 输入和 tuple/list 混合边界。",
      code: "def test_find_lists_single_list():\n    input_data = ([1, 2, 3],)\n    result = find_lists(input_data)\n    assert result == 1\n\ndef test_find_lists_multiple_lists():\n    input_data = ([1, 2, 3], [4, 5, 6], [7, 8])\n    result = find_lists(input_data)\n    assert result == 3\n\ndef test_find_lists_empty_tuple():\n    input_data = ()\n    result = find_lists(input_data)\n    assert result == 0",
    },
    {
      id: "case093",
      title: "any_int",
      tier: "weakquality",
      status: "复核",
      difficulty: "medium",
      score: 55,
      mutation: 0.384615,
      line: 0.818182,
      branch: 0.75,
      boundary: 1,
      asserts: 5,
      failures: 3,
      model: "basemodel",
      risk: "中风险",
      reason: "弱质量样例：断言数量可观，但变异杀伤不足，需要补充分支区分和反例。",
      fix: "补充 float、bool、负数和相加关系的互斥条件。",
      code: "import pytest\n\ndef test_normal_case():\n    assert any_int(5, 2, 7) == True\n    assert any_int(3, -2, 1) == True\n\ndef test_no_integer():\n    assert any_int(3.6, -2.2, 2) == False\n\ndef test_all_integers():\n    assert any_int(1, 2, 3) == True\n    assert any_int(-1, -2, -3) == True",
    },
    {
      id: "case214",
      title: "custom_contract",
      tier: "draft",
      status: "丢弃",
      difficulty: "unknown",
      score: 34,
      mutation: 0.18,
      line: 0.52,
      branch: 0.21,
      boundary: 0.25,
      asserts: 2,
      failures: 5,
      model: "rawprompt",
      risk: "高风险",
      reason: "调用契约不清晰，测试只检查 happy path，异常路径和边界路径缺失。",
      fix: "先确认函数输入类型和异常语义，再重新生成分支覆盖样例。",
      code: "def test_generated_contract():\n    result = target_function(sample_input)\n    assert result is not None\n    assert isinstance(result, object)",
    },
  ],
  factorial: [
    ["LL", 0.3727, 0.8637, 0.3072, 0.9261, 1.1174],
    ["LH", 0.3732, 0.8641, 0.3061, 0.9275, 1.1163],
    ["HL", 0.3675, 0.8623, 0.3018, 0.9315, 1.1245],
    ["HH", 0.3647, 0.8643, 0.3004, 0.9310, 1.1275],
  ],
  controls: [
    ["HH DPO", 0.3647, 0.8643, 0.3004, 0.9310, 1.1275],
    ["chosen only SFT", 0.4330, 0.8833, 0.3599, 0.9188, 1.0919],
    ["label shuffled DPO", 0.4330, 0.8829, 0.3622, 0.9144, 1.0920],
    ["matched random DPO", 0.4359, 0.8860, 0.3635, 0.9129, 1.0960],
  ],
  external: [
    ["base", 0.3596, 0.5535, 5.1461, 0.2957, 1.2093],
    ["LL", 0.3371, 0.5537, 4.8539, 0.2923, 1.1839],
    ["HL", 0.3371, 0.5070, 5.3258, 0.2916, 1.2118],
    ["HH", 0.3258, 0.5594, 5.1124, 0.2992, 1.1628],
  ],
  audit: [
    ["人工有效性", 0.9, 0.6786, 30, 3],
    ["失败类型", 0.7667, 0.7017, 30, 7],
    ["自动分类", 0.7667, 0.5579, 30, 7],
    ["严格行为", 0.9667, 0.0, 30, 1],
  ],
  failures: [
    ["变异幸存", "关键变异体幸存，说明测试没有杀死重要逻辑变化。"],
    ["断言薄弱", "测试可以执行，但断言不足或行为区分能力弱。"],
    ["契约错误", "测试调用方式和函数契约不一致，需要确认接口语义。"],
    ["异常缺口", "异常输入或错误路径没有被充分覆盖。"],
  ],
};

function pct(value) {
  return Math.round(value * 100) + "%";
}

function fixed(value) {
  return Number(value).toFixed(3);
}

function mutationText(item, value) {
  return item && (item.mutationvalid === false || Number(item.mutationvalid) === 0) ? "—" : pct(value);
}

function qs(selector) {
  return document.querySelector(selector);
}

function qsa(selector) {
  return Array.from(document.querySelectorAll(selector));
}

function getCandidates() {
  if (state.live && state.live.candidates && state.live.candidates.length) return state.live.candidates;
  const refs = state.references.length ? state.references : DATA.candidates;
  return refs.map((candidate) => ({ ...candidate, status: state.overrides[candidate.id] || candidate.status }));
}

function adjustCandidate(candidate, verify, index, fname) {
  const bonus = candidate.title === fname ? 4 : 0;
  const pressure = (verify - 3) * 3;
  const score = Math.max(20, Math.min(99, candidate.score + bonus - Math.max(0, index - 2) * 2 - pressure));
  const coverage = (Number(candidate.line || 0) + Number(candidate.branch || 0)) / 2;
  let status = "丢弃";
  if (candidate.failures > 0) {
    status = score >= 45 && candidate.failures <= 3 ? "复核" : "丢弃";
  } else if (candidate.mutationvalid !== false && Number(candidate.mutationvalid ?? 1) !== 0 && score >= 75 && candidate.mutation >= 0.6 && coverage >= 0.8) {
    status = "采纳";
  } else if (score >= 55) {
    status = "复核";
  }
  return { ...candidate, score, status: state.overrides[candidate.id] || status };
}

function detectFunction(code) {
  const matches = code.matchAll(/(?:async\s+)?def\s+([a-zA-Z][a-zA-Z0-9_]*)\s*\(/g);
  for (const match of matches) {
    if (!match[1].startsWith("test_")) return match[1];
  }
  return "";
}

function describeProfile() {
  const code = qs("#sourceCode").value;
  const fname = selectedFunctionName();
  const lines = code.split("\n").filter(Boolean).length;
  const signals = [];
  if (/str|text|string|match|char/.test(code)) signals.push("字符串语义");
  if (/list|tuple|dict|set|items/.test(code)) signals.push("集合结构");
  if (/int|num|return|math|n\)/.test(code)) signals.push("数值边界");
  if (/raise|except|TypeError|ValueError/.test(code)) signals.push("异常路径");
  if (!signals.length) signals.push("通用 Python 函数");
  return { fname, lines, signals };
}

function summarizeRun() {
  const candidates = state.live ? getCandidates() : [];
  const accept = candidates.filter((item) => item.status === "采纳").length;
  const review = candidates.filter((item) => item.status === "复核").length;
  const drop = candidates.filter((item) => item.status === "丢弃").length;
  const avg = candidates.reduce((sum, item) => sum + item.score, 0) / Math.max(1, candidates.length);
  const mutation = candidates.reduce((sum, item) => sum + item.mutation, 0) / Math.max(1, candidates.length);
  return { candidates, accept, review, drop, avg, mutation };
}


function makeRepair(candidate) {
  const fname = detectFunction(qs("#sourceCode").value);
  const safeName = fname || candidate.title || "target_function";
  const header = "import pytest\n\n";
  if (/hex|num|int|math/.test(safeName + " " + qs("#sourceCode").value)) {
    return header + "def test_" + safeName + "_validated_boundaries():\n" +
      "    assert " + safeName + "(1) is not None\n" +
      "    assert " + safeName + "(0) is not None\n\n" +
      "def test_" + safeName + "_rejects_bad_type():\n" +
      "    with pytest.raises((TypeError, ValueError)):\n" +
      "        " + safeName + "('invalid')\n";
  }
  if (/list|tuple|items|dict|set/.test(qs("#sourceCode").value)) {
    return header + "def test_" + safeName + "_collection_boundaries():\n" +
      "    assert " + safeName + "(()) == 0\n" +
      "    assert " + safeName + "(([1, 2],)) == 1\n" +
      "    assert " + safeName + "(([1], 'x', [], 3)) == 2\n\n" +
      "def test_" + safeName + "_non_list_values():\n" +
      "    assert " + safeName + "(('a', 1, None)) == 0\n";
  }
  return header + "def test_" + safeName + "_positive_and_negative_cases():\n" +
    "    assert " + safeName + "('ab') in (True, False)\n" +
    "    assert " + safeName + "('') in (True, False)\n\n" +
    "def test_" + safeName + "_boundary_inputs():\n" +
    "    assert " + safeName + "('a') in (True, False)\n" +
    "    assert " + safeName + "('bbb') in (True, False)\n";
}


async function api(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Token": state.token || "" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

async function apiget(path) {
  const join = path.includes("?") ? "&" : "?";
  const url = state.token ? path + join + "token=" + encodeURIComponent(state.token) : path;
  const response = await fetch(url, { headers: { "X-Token": state.token || "" } });
  return response.json();
}

function downloadBlob(name, content, type) {
  const blob = new Blob([content], { type });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

async function login(event) {
  event.preventDefault();
  const payload = {
    user: qs("#userName").value.trim() || "demo",
    password: qs("#password").value,
    confirm: qs("#confirmPassword").value,
    project: "stvr",
    role: qs("#userRole").value,
  };
  if (state.authmode === "register" && payload.password !== payload.confirm) {
    alert("两次密码不一致");
    return;
  }
  const result = await api(state.authmode === "register" ? "/api/register" : "/api/login", payload);
  if (!result.ok) {
    alert(result.error || (state.authmode === "register" ? "注册失败" : "登录失败"));
    return;
  }
  state.token = result.token;
  state.session = { user: result.user, project: result.project, role: result.role };
  qs("#login").classList.add("hidden");
  qs("#app").classList.remove("locked");
  qs("#userChip").textContent = state.session.user + " · " + state.session.role;
  qs("#railUser").textContent = state.session.user;
  qs("#railRole").textContent = state.session.role;
  applyRoleNav();
  await loadReferences();
  renderReport();
  await loadHistory();
  await loadDashboard();
  await loadModelStatus();
  await loadAssets();
  await loadUsers();
}

function setAuthMode(mode) {
  state.authmode = mode;
  qs("#loginTab").classList.toggle("active", mode === "login");
  qs("#registerTab").classList.toggle("active", mode === "register");
  qs("#authTitle").textContent = mode === "register" ? "用户注册" : "用户登录";
  qs("#authSubmit").textContent = mode === "register" ? "注册并登录" : "登录";
  qs(".confirm-row").classList.toggle("hidden", mode !== "register");
  qs(".role-row").classList.toggle("hidden", mode !== "register");
  qs("#password").autocomplete = mode === "register" ? "new-password" : "current-password";
  if (mode === "register") {
    qs("#confirmPassword").value = "";
    if (qs("#userName").value === "demo") {
      qs("#userName").value = "";
      qs("#password").value = "";
    }
  } else if (!qs("#userName").value) {
    qs("#userName").value = "demo";
    qs("#password").value = "demo123";
  }
}




function logout() {
  state.token = "";
  state.session = { user: "demo", project: "stvr", role: "开发者" };
  state.live = null;
  state.record = null;
  state.history = [];
  state.users = [];
  state.assets = null;
  state.references = [];
  state.batchfiles = [];
  state.batchsource = "";
  state.batchtargets = [];
  state.targetfunction = "";
  state.running = false;
  state.batch = null;
  state.dashboard = null;
  state.selected = 0;
  setAuthMode("login");
  qs("#login").classList.remove("hidden");
  qs("#app").classList.add("locked");
  qs("#userChip").textContent = "demo";
  qs("#railUser").textContent = "demo";
  qs("#railRole").textContent = "开发者";
  renderProjectDash();
  renderRecordDetail();
  renderHistory();
  renderUsers("请使用管理员账号查看和维护用户");
  applyRoleNav();
  clearJobProgress();
  clearBatchInput();
}

function targetKindLabel(target) {
  const kind = String(target.kind || "function");
  if (kind.includes("nested")) return kind.startsWith("async") ? "异步嵌套函数" : "嵌套函数";
  if (kind.includes("method")) {
    const prefix = kind.startsWith("async") ? "异步" : "";
    const style = kind.split(":")[1] || "instance";
    return prefix + ({ static: "静态方法", class: "类方法", instance: "实例方法" }[style] || "类方法");
  }
  if (kind.includes("internal")) return kind.startsWith("async") ? "异步内部函数" : "内部函数";
  return kind === "async" ? "异步函数" : "普通函数";
}

function scanFunctions(files) {
  const targets = [];
  files.forEach((file) => {
    const path = String(file.path || "target.py").replace(/\\/g, "/");
    const parts = path.split("/").filter(Boolean);
    const filename = (parts[parts.length - 1] || "").toLowerCase();
    if (filename.startsWith("test_") || filename.endsWith("_test.py") || parts.slice(0, -1).some((part) => ["test", "tests"].includes(part.toLowerCase()))) return;
    const stack = [];
    let decorators = [];
    const lines = String(file.content || "").split("\n");
    lines.forEach((raw, index) => {
      const expanded = raw.replace(/\t/g, "    ");
      const trimmed = expanded.trim();
      if (!trimmed || trimmed.startsWith("#")) return;
      const indent = expanded.length - expanded.trimStart().length;
      while (stack.length && indent <= stack[stack.length - 1].indent) stack.pop();
      const decorator = trimmed.match(/^@([A-Za-z_][A-Za-z0-9_.]*)/);
      if (decorator) {
        decorators.push(decorator[1].split(".").pop());
        return;
      }
      const classMatch = trimmed.match(/^class\s+([A-Za-z_][A-Za-z0-9_]*)/);
      if (classMatch) {
        stack.push({ type: "class", name: classMatch[1], indent });
        decorators = [];
        return;
      }
      const match = trimmed.match(/^(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
      if (!match) {
        decorators = [];
        return;
      }
      const leaf = match[2];
      if ((leaf.startsWith("__") && leaf.endsWith("__")) || leaf.startsWith("test_")) {
        decorators = [];
        return;
      }
      const scopes = stack.map((item) => item.name);
      const name = [...scopes, leaf].join(".");
      const callable = [...stack].reverse().find((item) => item.type === "function");
      const ownerClass = [...stack].reverse().find((item) => item.type === "class");
      const asynchronous = Boolean(match[1]);
      const internal = leaf.startsWith("_");
      let kind = asynchronous ? "async" : "function";
      if (callable) {
        kind = asynchronous ? "asyncnested" : "nested";
      } else if (ownerClass) {
        const style = decorators.includes("staticmethod") ? "static" : decorators.includes("classmethod") ? "class" : "instance";
        kind = (asynchronous ? "async" : "") + (internal ? "internalmethod" : "method") + ":" + style;
      } else if (internal) {
        kind = asynchronous ? "asyncinternal" : "internal";
      }
      const effective = callable ? callable.effective : name;
      targets.push({ path: file.path, name, leaf, line: index + 1, kind, async: asynchronous, internal, effective, direct: !callable });
      stack.push({ type: "function", name: leaf, indent, effective });
      decorators = [];
    });
  });
  return targets;
}

function selectedTargetPath() {
  return "target.py";
}

function selectedFunctionName() {
  const picker = qs("#functionSelect");
  if (picker && picker.value) return picker.value;
  if (state.targetfunction) return state.targetfunction;
  const current = scanFunctions([{ path: "target.py", content: qs("#sourceCode").value }]);
  return current.length ? current[0].name : detectFunction(qs("#sourceCode").value);
}

async function scanProjectTargets() {
  if (!state.token || !state.batchfiles.length || !/^https?:$/.test(window.location.protocol)) return;
  try {
    const result = await api("/api/scan", { token: state.token, files: state.batchfiles });
    if (result.ok && Array.isArray(result.functions)) state.batchtargets = result.functions;
  } catch (error) {
    console.warn("server scan unavailable", error);
  }
}

async function readProjectFiles(event) {
  const items = Array.from(event.target.files || []).filter((file) => file.name.endsWith(".py"));
  const files = [];
  for (const file of items.slice(0, 64)) {
    const content = await file.text();
    files.push({ path: file.webkitRelativePath || file.name, content });
  }
  state.batchfiles = files;
  state.batchtargets = scanFunctions(files);
  state.batch = null;
  const folder = items.find((file) => file.webkitRelativePath);
  state.batchsource = folder ? folder.webkitRelativePath.split("/")[0] : files.length + " 个 Python 文件";
  await scanProjectTargets();
  renderBatchImportState();
  renderBatchTargets();
  renderBatchResults();
}

async function readArchiveFile(event) {
  const file = (event.target.files || [])[0];
  if (!file) return;
  if (!state.token) { alert("请先登录"); return; }
  const status = qs("#batchImportState");
  if (status) status.innerHTML = "<b>正在解析</b><span>" + escapeHtml(file.name) + "</span>";
  const form = new FormData();
  form.append("archive", file);
  const response = await fetch("/api/upload", { method: "POST", headers: { "X-Token": state.token }, body: form });
  const result = await response.json();
  if (!result.ok) { alert(result.error || "压缩包上传失败"); renderBatchImportState(); return; }
  state.batchfiles = result.files || [];
  state.batchtargets = result.functions || scanFunctions(state.batchfiles);
  state.batchsource = file.name;
  state.batch = null;
  renderBatchImportState();
  renderBatchTargets();
  renderBatchResults();
}

function renderBatchImportState() {
  const node = qs("#batchImportState");
  const clear = qs("#clearBatch");
  const ready = state.batchfiles.length > 0;
  if (clear) clear.classList.toggle("hidden", !ready);
  if (!node) return;
  if (!ready) {
    node.innerHTML = "<b>选择待评估代码</b><span>单个 .py、完整项目或项目 ZIP</span>";
    return;
  }
  node.innerHTML = "<b>" + escapeHtml(state.batchsource || "项目已导入") + "</b><span>" + state.batchfiles.length + " 个 Python 文件已读取 · " + state.batchtargets.length + " 个业务函数可评估</span>";
}

function clearBatchInput() {
  state.batchfiles = [];
  state.batchtargets = [];
  state.batchsource = "";
  state.batch = null;
  ["#projectFiles", "#projectFolder", "#archiveFile"].forEach((selector) => {
    const input = qs(selector);
    if (input) input.value = "";
  });
  renderBatchImportState();
  renderBatchTargets();
  renderBatchResults();
}

async function loadReferences() {
  if (!state.token || !/^https?:$/.test(window.location.protocol)) return;
  const result = await apiget("/api/references");
  if (result.ok) {
    state.references = result.items || [];
    renderRuntime();
    renderSelected();
  }
}

async function loadHistory() {
  if (!state.token || !/^https?:$/.test(window.location.protocol)) return;
  const result = await apiget("/api/history");
  if (result.ok) {
    state.history = result.items || [];
    const available = new Set(state.history.map((item) => Number(item.id)));
    Array.from(state.selectedRecords).forEach((id) => {
      if (!available.has(Number(id))) state.selectedRecords.delete(Number(id));
    });
    renderHistory();
    renderBrief();
    renderReport();
  }
}

function renderHistory() {
  const node = qs("#historyList");
  if (!node) return;
  const selected = state.selectedRecords;
  const scope = qs("#historyScope");
  const remove = qs("#deleteHistory");
  if (scope) scope.textContent = selected.size ? "已选择 " + selected.size + " 条" : "未选择时分析全部";
  if (remove) remove.disabled = selected.size === 0;
  if (!state.history.length) {
    node.innerHTML = "<div class='empty'>暂无评估记录</div>";
    return;
  }
  const allSelected = state.history.every((item) => selected.has(Number(item.id)));
  const header = '<div class="history-row history-head"><label class="record-check"><input id="historyAll" type="checkbox" aria-label="选择全部记录" ' + (allSelected ? "checked" : "") + '></label><span>ID</span><span>目标</span><span>状态</span><span>质量分</span><span>覆盖</span><span>变异</span><span>时间</span></div>';
  const rows = state.history.map((item) => {
    const checked = selected.has(Number(item.id));
    return '<div class="history-row ' + (checked ? "selected" : "") + '" data-id="' + item.id + '"><label class="record-check"><input class="history-check" type="checkbox" data-check="' + item.id + '" aria-label="选择记录 ' + item.id + '" ' + (checked ? "checked" : "") + '></label><span>' + item.id + '</span><span title="' + escapeHtml(item.target) + '">' + escapeHtml(item.target) + '</span><span><i class="tag ' + tagClass(item.status) + '">' + item.status + '</i></span><span class="num">' + item.score + '</span><span>' + pct(item.line || 0) + '</span><span>' + mutationText(item, item.mutation || 0) + '</span><span>' + item.created + '</span></div>';
  }).join("");
  node.innerHTML = header + rows;
  const all = qs("#historyAll");
  if (all) all.addEventListener("change", (event) => {
    state.history.forEach((item) => {
      if (event.target.checked) selected.add(Number(item.id));
      else selected.delete(Number(item.id));
    });
    renderHistory();
    renderBrief();
    renderReport();
  });
  qsa(".history-check").forEach((input) => input.addEventListener("change", (event) => {
    const id = Number(event.target.dataset.check);
    if (event.target.checked) selected.add(id);
    else selected.delete(id);
    renderHistory();
    renderBrief();
    renderReport();
  }));
  qsa(".history-row[data-id]").forEach((row) => row.addEventListener("click", (event) => {
    if (event.target.closest("input")) return;
    loadRecord(row.dataset.id);
  }));
}

function analyzeHistory() {
  switchView("report");
  renderBrief();
  renderReport();
}

async function deleteHistory() {
  const ids = Array.from(state.selectedRecords);
  if (!ids.length) return;
  if (!confirm("确认删除选中的 " + ids.length + " 条评估记录？")) return;
  const result = await api("/api/records", { token: state.token, ids });
  if (!result.ok) {
    alert(result.error || "删除失败");
    return;
  }
  state.selectedRecords.clear();
  state.record = null;
  qs("#recordTitle").textContent = "评估详情";
  renderRecordDetail();
  await loadHistory();
  await loadDashboard();
}

async function loadDashboard() {
  if (!state.token || !/^https?:$/.test(window.location.protocol)) return;
  const result = await apiget("/api/dashboard");
  if (result.ok) {
    state.dashboard = result;
    renderProjectDash();
  }
}

function statusCount(name) {
  const rows = (state.dashboard && state.dashboard.statuses) || [];
  const found = rows.find((item) => item.status === name);
  return found ? found.count : 0;
}

function renderProjectDash() {
  const node = qs("#projectDash");
  if (!node) return;
  const data = state.dashboard || { total: 0, avg: { score: 0, line: 0, mutation: 0, passed: 0 }, recent: [] };
  node.innerHTML = [
    [data.total || 0, "评估记录", "当前项目"],
    [data.avg.score || 0, "平均质量分", "历史记录"],
    [pct(data.avg.line || 0), "平均行覆盖", "coverage"],
    [pct(data.avg.mutation || 0), "平均变异", "mutation"],
    [statusCount("采纳"), "采纳记录", "推荐用例"],
  ].map(([value, label, note]) => "<article><b>" + value + "</b><span>" + label + "</span><small>" + note + "</small></article>").join("");
}

function switchView(view) {
  const single = view === "run";
  [qs("#sampleSelect"), qs("#runButton")].forEach((item) => {
    if (item) item.classList.toggle("hidden", !single);
  });
  qsa(".nav-btn").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  qsa(".view").forEach((item) => item.classList.toggle("active", item.id === view));
  drawexternal();
  renderAssets();
  renderBatchImportState();
  renderBatchTargets();
  renderBatchResults();
  renderReport();
  renderBrief();
}

function applyRoleNav() {
  const internal = new Set(["data", "model", "audit", "users"]);
  const admin = state.session.role === "管理员";
  qsa(".nav-btn").forEach((button) => {
    if (internal.has(button.dataset.view)) button.classList.toggle("hidden", !admin);
  });
  const active = qs(".view.active");
  const current = active ? active.id : "run";
  if (!admin && internal.has(current)) switchView("run");
}

async function loadUsers() {
  const node = qs("#usersList");
  if (!node || !state.token || !/^https?:$/.test(window.location.protocol)) return;
  if (state.session.role !== "管理员") {
    state.users = [];
    renderUsers("请使用管理员账号查看和维护用户");
    return;
  }
  const result = await apiget("/api/users");
  if (result.ok) {
    state.users = result.items || [];
    renderUsers();
  } else {
    renderUsers(result.error || "无法读取用户列表");
  }
}

function renderUsers(message = "") {
  const node = qs("#usersList");
  if (!node) return;
  if (message) {
    node.innerHTML = "<div class='empty'>" + message + "</div>";
    return;
  }
  node.innerHTML = "<div class='user-row user-head'><span>账号</span><span>角色</span><span>创建时间</span></div>" +
    state.users.map((item) => "<div class='user-row' data-name='" + item.name + "' data-role='" + item.role + "'><b>" + item.name + "</b><span>" + item.role + "</span><span>" + item.created + "</span></div>").join("");
  qsa(".user-row[data-name]").forEach((row) => row.addEventListener("click", () => {
    qs("#editUser").value = row.dataset.name;
    qs("#editRole").value = row.dataset.role;
    qs("#editPassword").value = "";
  }));
}

async function saveUser(event) {
  event.preventDefault();
  if (state.session.role !== "管理员") { alert("需要管理员权限"); return; }
  const result = await api("/api/users", { token: state.token, action: "save", name: qs("#editUser").value.trim(), password: qs("#editPassword").value, role: qs("#editRole").value });
  if (!result.ok) { alert(result.error || "保存失败"); return; }
  qs("#editPassword").value = "";
  await loadUsers();
}

async function removeUser() {
  if (state.session.role !== "管理员") { alert("需要管理员权限"); return; }
  const name = qs("#editUser").value.trim();
  if (!name || !confirm("确认删除账号 " + name + "？")) return;
  const result = await api("/api/users", { token: state.token, action: "delete", name });
  if (!result.ok) { alert(result.error || "删除失败"); return; }
  qs("#editUser").value = "";
  await loadUsers();
}

async function loadRecord(id) {
  if (!id) return;
  const result = await apiget("/api/record?id=" + encodeURIComponent(id));
  if (result.ok) {
    state.record = result.record;
    renderRecordDetail();
  } else {
    alert(result.error || "记录不存在");
  }
}

function renderRecordDetail() {
  const node = qs("#recordDetail");
  if (!node) return;
  const row = state.record;
  if (!row) {
    node.innerHTML = "<div class='empty'>从左侧选择一条记录查看详情</div>";
    return;
  }
  const result = row.result || {};
  const metrics = result.metrics || {};
  const candidates = result.candidates || [];
  const candidate = candidates[0] || {};
  const sourceLabel = (item) => String(item.engine || "").startsWith("llmrepair") ? "模型修复" : String(item.engine || "").startsWith("llm") ? "质量偏好模型" : "执行观察";
  const sourceTrace = candidates.map((item) => sourceLabel(item) + " " + Number(item.score || 0) + "（" + (item.status || "待评估") + "）").join(" · ");
  const generation = candidates.some((item) => String(item.engine || "").startsWith("llm")) ? "GPU 已调用" : "沙箱生成";
  qs("#recordTitle").textContent = "#" + row.id + " · " + row.target;
  const mutationRows = (metrics.mutationdetails || candidate.mutationdetails || []).slice(0, 8).map((item) => "<div class='mutation-row'><span>" + item.operator + "</span><span>line " + item.line + "</span><b>" + (item.killed ? "已杀死" : "幸存") + "</b></div>").join("");
  const security = (result.security && result.security.summary) || (candidate.audit && candidate.audit.summary) || "日志正常";
  node.innerHTML = "<div class='record-metrics'>" +
    metricRow("质量分", row.score) + metricRow("状态", row.status) + metricRow("行覆盖", pct(metrics.line || row.line || 0)) + metricRow("变异", mutationText({ mutationvalid: metrics.mutationvalid ?? row.mutationvalid }, metrics.mutation || row.mutation || 0)) +
    "</div><div class='diagnosis'><b>生成链路：</b>" + generation + " · " + (sourceTrace || "暂无候选") + "<br><b>缺陷诊断：</b>" + (candidate.reason || "暂无诊断") + "<br><span>修复推荐：" + (candidate.fix || "暂无建议") + "</span><br><span>执行验证：" + security + "</span></div>" +
    "<div class='mutation-list'>" + (mutationRows || "<div class='empty'>暂无变异明细</div>") + "</div>" +
    "<pre><code>" + escapeHtml(candidate.code || "") + "</code></pre>";
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

function downloadCurrentTest() {
  const candidates = getCandidates();
  const candidate = candidates[state.selected] || candidates[0];
  if (!candidate) return;
  downloadBlob("stvrtest.py", candidate.code || "", "text/x-python;charset=utf-8");
}

function openRecordDownload(kind) {
  const id = state.record ? state.record.id : (state.live && state.live.recordid) || (state.history[0] && state.history[0].id);
  if (!id) { alert("请先完成一次评估或选择历史记录"); return; }
  const path = kind === "pdf" ? "/api/reportpdf" : "/api/test";
  window.location.href = path + "?id=" + encodeURIComponent(id) + "&token=" + encodeURIComponent(state.token);
}

function openProjectReportDownload() {
  if (!state.token) { alert("请先登录"); return; }
  const ids = Array.from(state.selectedRecords);
  const query = ids.length ? "&ids=" + encodeURIComponent(ids.join(",")) : "";
  window.location.href = "/api/projectreportpdf?token=" + encodeURIComponent(state.token) + query;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds || 0)));
  if (value < 60) return value + " 秒";
  const minutes = Math.floor(value / 60);
  const rest = value % 60;
  return minutes + " 分" + (rest ? " " + rest + " 秒" : "");
}

function renderJobProgress(info) {
  const node = qs("#runProgress");
  if (!node || !info) return;
  node.classList.remove("hidden");
  const value = Math.max(0, Math.min(100, Number(info.progress || 0)));
  qs("#progressStage").textContent = info.detail || info.label || "正在评估";
  qs("#progressEta").textContent = info.status === "completed" ? "已完成" : info.status === "failed" ? "已停止" : Number(info.eta || 0) > 0 ? "预计剩余 " + formatDuration(info.eta) : "正在估算";
  qs("#progressPercent").textContent = Math.round(value) + "%";
  qs("#progressElapsed").textContent = "已用 " + formatDuration(info.elapsed);
  qs("#progressFill").style.width = value + "%";
  qs("#progressTrack").setAttribute("aria-valuenow", String(Math.round(value)));
  const stages = { prepare: 0, generate: 0, execute: 1, evidence: 2, diagnose: 3, report: 4, done: 4 };
  const active = stages[info.stage] ?? 0;
  renderFlow(active, info.stage === "done" ? DATA.flow.length : active);
}

function clearJobProgress() {
  const node = qs("#runProgress");
  if (node) node.classList.add("hidden");
  if (qs("#progressFill")) qs("#progressFill").style.width = "0%";
}

async function requestSandbox(onProgress) {
  if (!/^https?:$/.test(window.location.protocol) || !state.token) return { ok: false, error: "backend unavailable" };
  const payload = {
    token: state.token,
    code: qs("#sourceCode").value,
    files: [],
    target: selectedTargetPath(),
    function: selectedFunctionName(),
    framework: qs("#frameworkSelect").value,
    engine: qs("#engineSelect").value,
    count: Number(qs("#candidateCount").value || 6),
    level: Number(qs("#verifyLevel").value || 3),
  };
  try {
    const started = await api("/api/evaluate/start", payload);
    if (!started.ok || !started.job) return { ok: false, error: started.error || "评估任务启动失败" };
    const deadline = Date.now() + 20 * 60 * 1000;
    while (Date.now() < deadline) {
      await wait(600);
      const status = await apiget("/api/evaluate/status?id=" + encodeURIComponent(started.job));
      if (onProgress) onProgress(status);
      if (status.status === "completed") return status.result;
      if (status.status === "failed") return { ok: false, error: status.error || "评估失败" };
    }
    return { ok: false, error: "评估等待超时，请检查模型与沙箱状态" };
  } catch (error) {
    return { ok: false, error: error.message || "评估请求失败" };
  }
}
function resetLive() {
  state.live = null;
  state.selected = 0;
  if (!state.running) clearJobProgress();
}

function renderSampleSelect() {
  const node = qs("#sampleSelect");
  if (!node) return;
  node.innerHTML = '<option value="">载入示例</option>' + DATA.targets
    .map((target, index) => '<option value="' + index + '">示例 · ' + target.short + '</option>')
    .join("");
}

function loadSample(index = 0) {
  const sample = DATA.targets[index] || DATA.targets[0];
  state.activeTarget = index;
  state.target = "target.py";
  state.targetfunction = "";
  qs("#sourceCode").value = sample.code;
  renderTargets();
  resetLive();
  renderAllDynamic();
}

function renderFunctionTargets() {
  const picker = qs("#functionSelect");
  const targets = scanFunctions([{ path: "target.py", content: qs("#sourceCode").value }]);
  if (!targets.length) {
    state.targetfunction = "";
    picker.innerHTML = "<option value=\"\">未识别到函数</option>";
    picker.disabled = true;
    return;
  }
  picker.disabled = false;
  if (!targets.some((item) => item.name === state.targetfunction)) state.targetfunction = targets[0].name;
  picker.innerHTML = targets.map((item) => {
    const routed = item.direct === false ? " · 经 " + item.effective + " 验证" : "";
    return "<option value=\"" + escapeHtml(item.name) + "\">" + escapeHtml(item.name) + " · " + targetKindLabel(item) + routed + "</option>";
  }).join("");
  picker.value = state.targetfunction;
}

function renderTargets() {
  state.target = "target.py";
  renderFunctionTargets();
}

function renderInspector() {
  const profile = describeProfile();
  const mode = state.live ? "函数沙箱" : "代码待评估";
  qs("#candidateValue").textContent = qs("#candidateCount").value;
  qs("#verifyValue").textContent = qs("#verifyLevel").value;
  qs("#inspector").innerHTML = "<span>函数 <b>" + profile.fname + "</b></span>" +
    "<span>代码行 <b>" + profile.lines + "</b></span>" +
    "<span>模式 <b>" + mode + "</b></span>" +
    "<span class='wide'>识别信号 <b>" + profile.signals.join(" / ") + "</b></span>";
}

function renderFlow(active = 0, done = 0) {
  qs("#flow").innerHTML = DATA.flow
    .map(([title, note], index) => {
      const cls = index < done ? "done" : index === active ? "active" : "";
      const stateText = index < done ? "已完成" : index === active ? "运行中" : "等待";
      return "<div class='flow-step " + cls + "'><span class='flow-index'>" + (index + 1) + "</span>" +
        "<div><b>" + title + "</b><small>" + note + "</small></div><span class='mini'>" + stateText + "</span></div>";
    })
    .join("");
}

function renderRuntime() {
  const node = qs("#runtime");
  if (!node) return;
  const summary = summarizeRun();
  if (!state.live || !summary.candidates.length) {
    node.innerHTML = '<div class="result-caption"><b>本次结果</b><span>完成评估后显示质量结论</span></div><div class="empty">暂无评估结果</div>';
    return;
  }
  const candidate = summary.candidates[state.selected] || summary.candidates[0];
  const coverage = (Number(candidate.line || 0) + Number(candidate.branch || 0)) / 2;
  const count = Number(qs("#candidateCount").value || candidate.asserts || 0);
  const level = Number(qs("#verifyLevel").value || 3);
  node.innerHTML = '<div class="result-caption"><b>本次结果</b><span>测试用例 ' + count + ' · 严格度 ' + level + '</span></div>' +
    '<div class="current-result result-head"><span>目标函数</span><span>结论</span><span>质量分</span><span>变异</span><span>覆盖</span><span>失败</span></div>' +
    '<article class="current-result result-row"><span class="case-name"><b>' + escapeHtml(candidate.title) + '</b></span>' +
    '<span><i class="tag ' + tagClass(candidate.status) + '">' + candidate.status + '</i></span>' +
    '<span class="num">' + candidate.score + '</span><span>' + mutationText(candidate, candidate.mutation) + '</span><span>' + pct(coverage) + '</span><span>' + candidate.failures + '</span></article>'; 
}

function renderMetrics() {
  const node = qs("#metrics");
  if (!node) return;
  const summary = summarizeRun();
  const live = state.live && state.live.metrics;
  const metrics = live ? [
    [live.passed ? "通过" : "失败", "pytest", "真实沙箱执行", live.passed ? 1 : 0.2],
    [pct(live.line), "行覆盖", "coverage run", live.line],
    [mutationText(live, live.mutation), "变异杀伤", live.mutationvalid ? live.killed + " / " + live.mutants + " mutants" : "不可评价", live.mutationvalid ? live.mutation : 0],
    [String(summary.accept), "推荐采纳", "当前输入", summary.accept / Math.max(1, summary.candidates.length)],
  ] : [
    ["—", "执行状态", "等待评估", 0],
    ["—", "行覆盖", "等待评估", 0],
    ["—", "变异杀伤", "等待评估", 0],
    ["0", "推荐采纳", "当前输入", 0],
  ];
  node.innerHTML = metrics
    .map(([value, label, note, width]) => "<article class='metric'><b>" + value + "</b><span>" + label + " · " + note + "</span><div class='bar'><i style='width:" + Math.max(7, width * 100) + "%'></i></div></article>")
    .join("");
}

function renderEvidence() {
  const node = qs("#evidence");
  if (!node) return;
  node.innerHTML = DATA.evidence
    .map(([title, note], index) => "<article><span>0" + (index + 1) + "</span><b>" + title + "</b><p>" + note + "</p></article>")
    .join("");
}

function tagClass(status) {
  if (status === "采纳") return "ok";
  if (status === "丢弃") return "bad";
  return "warn";
}

function metricRow(label, value) {
  return "<div class='metricline'><span>" + label + "</span><b>" + value + "</b></div>";
}

function renderSelected() {
  if (!state.live) {
    qs("#detailEyebrow").textContent = "缺陷诊断";
    qs("#selectedName").textContent = "暂无评估结果";
    qs("#selectedScore").textContent = "-";
    qs("#selectedCode").textContent = "";
    qs("#diagnosis").innerHTML = "<b>缺陷诊断：</b>暂无";
    qs("#repairCode").textContent = "";
    qs("#repairState").textContent = "待生成";
    return;
  }
  const candidates = getCandidates();
  const candidate = candidates[state.selected] || candidates[0];
  if (!candidate) return;
  qs("#detailEyebrow").textContent = "缺陷诊断";
  qs("#selectedName").textContent = candidate.title;
  qs("#selectedScore").textContent = candidate.score;
  qs("#selectedCode").textContent = candidate.code;
  const audit = candidate.audit && candidate.audit.summary ? candidate.audit.summary : "日志正常";
  qs("#diagnosis").innerHTML = "<b>诊断结论：</b>" + candidate.reason + "<br><span>执行验证：" + audit + "</span><div class='scoregrid'>" +
    metricRow("变异", candidate.mutationvalid === false ? "—" : fixed(candidate.mutation)) +
    metricRow("行覆盖", fixed(candidate.line)) +
    metricRow("分支", fixed(candidate.branch)) +
    metricRow("边界", fixed(candidate.boundary)) +
    "</div>";
  if (candidate.status === "采纳") {
    qs("#repairCode").textContent = "# 当前测试已通过质量门，无需修复。";
    qs("#repairState").textContent = "无需修复";
  } else {
    qs("#repairCode").textContent = makeRepair(candidate);
    qs("#repairState").textContent = "已增强";
  }
}

function renderFunnel() {
  const rows = state.assets && state.assets.funnel ? state.assets.funnel : DATA.funnel.map(([label, value, _max, note]) => ({ label, value, note }));
  const max = Math.max(1, ...rows.map((item) => Number(item.value || 0)));
  qs("#funnel").innerHTML = rows
    .map((item) => "<div class='funnel-row'><b>" + item.label + "</b><div class='funnel-track'><i style='width:" + Math.max(5, (Number(item.value || 0) / max) * 100) + "%'></i></div><span>" + Number(item.value || 0).toLocaleString() + "</span><small style='grid-column:2 / 4;color:var(--muted)'>" + item.note + "</small></div>")
    .join("");
}

async function loadAssets() {
  if (!state.token || !/^https?:$/.test(window.location.protocol)) return;
  const result = await apiget("/api/assets");
  if (result.ok) {
    state.assets = result;
    renderAssets();
  }
}

function renderAssets() {
  renderFunnel();
  const cards = qs("#assetCards");
  if (cards && state.assets) {
    const funnel = state.assets.funnel || [];
    const total = funnel.find((item) => item.label === "可信测试" || item.label === "候选测试") || {};
    const usable = funnel.find((item) => item.label === "可评估测试") || {};
    const pair = funnel.find((item) => item.label === "拒绝配对") || {};
    cards.innerHTML = [
      [Number(total.value || 0).toLocaleString(), "候选规模"],
      [Number(usable.value || 0).toLocaleString(), "可评估样例"],
      [Number(pair.value || 0).toLocaleString(), "偏好配对"],
    ].map(([value, label]) => "<article><b>" + value + "</b><span>" + label + "</span></article>").join("");
  }
  const project = qs("#projectBreakdown");
  if (project && state.assets) {
    const rows = (state.assets.projects || []).slice(0, 8);
    project.innerHTML = "<div class='table-row header'><span>项目</span><span>组别</span><span>通过</span><span>变异</span></div>" +
      rows.map((row) => "<div class='table-row'><b>" + (row.project || row.project_name || "project") + "</b><span>" + (row.group || "-") + "</span><span>" + pct(Number(row.execution_pass_rate || 0)) + "</span><span>" + pct(Number(row.mutation_score_mean || 0)) + "</span></div>").join("");
  }
}

function renderBatchTargets() {
  const node = qs("#batchTargets");
  if (!node) return;
  const files = state.batchfiles || [];
  const targets = state.batchtargets.length ? state.batchtargets : scanFunctions(files);
  state.batchtargets = targets;
  if (!files.length) {
    node.innerHTML = "<div class='empty'>导入 .py 文件、完整项目或项目 ZIP 后显示文件清单</div>";
    const stateNode = qs("#batchState");
    if (stateNode) stateNode.textContent = "尚未扫描";
    return;
  }
  const limit = Number((qs("#batchLimit") && qs("#batchLimit").value) || 6);
  const stateNode = qs("#batchState");
  if (stateNode) stateNode.textContent = targets.length ? "识别 " + targets.length + " 个目标 · 本次最多评估 " + Math.min(limit, targets.length) + " 个" : "未识别到业务函数";
  const rows = files.map((file) => {
    const path = String(file.path || "target.py").replace(/\\/g, "/");
    const own = targets.filter((item) => item.path === path || path.endsWith("/" + item.path));
    const parts = path.split("/").filter(Boolean);
    const filename = (parts[parts.length - 1] || "").toLowerCase();
    const testfile = filename.startsWith("test_") || filename.endsWith("_test.py") || parts.slice(0, -1).some((part) => ["test", "tests"].includes(part.toLowerCase()));
    const role = own.length ? "待评估" : testfile ? "测试依赖" : "项目依赖";
    const roleClass = own.length ? "ok" : testfile ? "warn" : "neutral";
    const names = own.length ? own.map((item) => item.name).join("、") : "不作为评估目标";
    return "<div class='table-row'><b>" + escapeHtml(path) + "</b><span>" + escapeHtml(names) + "</span><span><i class='tag " + roleClass + "'>" + role + "</i></span><span>" + own.length + "</span></div>";
  });
  node.innerHTML = "<div class='table-row header'><span>Python 文件</span><span>业务目标</span><span>文件角色</span><span>目标数</span></div>" + rows.join("");
}

async function runBatch() {
  if (!state.token) { alert("请先登录"); return; }
  if (!state.batchfiles.length) { alert("请先导入 .py 文件、完整项目或项目 ZIP"); return; }
  renderBatchTargets();
  qs("#batchSummary").textContent = "运行中";
  qs("#batchResults").innerHTML = "<div class='empty'>正在批量评估，请稍候</div>";
  const payload = { token: state.token, files: state.batchfiles, framework: qs("#batchFramework").value, engine: qs("#batchEngine").value, limit: Number(qs("#batchLimit").value || 6), count: 5, level: 3 };
  const result = await api("/api/batch", payload);
  if (!result.ok) { alert(result.error || "批量评估失败"); qs("#batchSummary").textContent = "失败"; return; }
  state.batch = result;
  await loadHistory();
  await loadDashboard();
  renderBatchResults();
}

function renderBatchResults() {
  const node = qs("#batchResults");
  if (!node) return;
  const result = state.batch;
  if (!result) {
    node.innerHTML = "<div class='empty'>批量运行后显示每个函数的质量分、覆盖率和变异分数</div>";
    if (qs("#batchSummary")) qs("#batchSummary").textContent = "未运行";
    return;
  }
  if (qs("#batchSummary")) qs("#batchSummary").textContent = result.passed + " / " + result.total + " 通过";
  node.innerHTML = "<div class='table-row header'><span>函数</span><span>状态</span><span>质量分</span><span>覆盖</span><span>变异</span><span>记录</span></div>" +
    (result.items || []).map((item) => item.ok ? "<div class='table-row'><b>" + item.function + "</b><span><i class='tag " + tagClass(item.status) + "'>" + item.status + "</i></span><span>" + item.score + "</span><span>" + pct(item.line || 0) + "</span><span>" + mutationText(item, item.mutation || 0) + "</span><span>#" + item.recordid + "</span></div>" : "<div class='table-row'><b>" + item.function + "</b><span class='tag bad'>失败</span><span>0</span><span>0%</span><span>0%</span><span>" + escapeHtml(item.error || "error") + "</span></div>").join("");
}


async function loadModelStatus() {
  if (!/^https?:$/.test(window.location.protocol)) return;
  const response = await fetch("/api/model");
  const result = await response.json();
  if (result.ok) {
    state.model = { ...(result.config || {}), sandbox: result.sandbox || {} };
    renderModelStatus();
  }
}

function modelDisplayName(cfg) {
  const raw = cfg.active === "local" ? (cfg.local || "") : (cfg.model || cfg.url || "");
  const name = String(raw).split("/").filter(Boolean).pop() || "未启用";
  if (/qwen2\.5-7b/i.test(name)) return "Qwen2.5-7B";
  if (/deepseek-coder-6\.7b/i.test(name)) return "DeepSeek 6.7B";
  if (/qwen2\.5-14b/i.test(name)) return "Qwen2.5-14B";
  return name.replace(/-/g, " ");
}

function renderModelStatus() {
  const node = qs("#modelStatus");
  if (!node) return;
  const cfg = state.model || { active: "none", available: false, model: "", local: "", sandbox: {}, runtime: {} };
  const rt = cfg.runtime || {};
  let status = cfg.active === "api" ? "接口模型" : cfg.active === "local" ? "本地模型" : "未启用";
  if (cfg.active === "local" && cfg.localready && !cfg.localrunnable) status = "待启用";
  if (cfg.active === "local" && cfg.localrunnable) status = "可运行";
  const device = rt.cuda ? "GPU 可用" : (rt.allowcpu ? "CPU 模式" : "待配置");
  const loaded = cfg.loaded ? "已加载" : "未加载";
  const sandbox = "轻量隔离";
  node.innerHTML = "<article><b>生成引擎</b><span>" + status + "</span></article>" +
    "<article><b>模型</b><span>" + modelDisplayName(cfg) + "</span></article>" +
    "<article><b>推理设备</b><span>" + device + "</span></article>" +
    "<article><b>加载状态</b><span>" + loaded + "</span></article>" +
    "<article><b>运行沙箱</b><span>" + sandbox + "</span></article>";
}

async function warmModel() {
  if (!state.token) { alert("请先登录"); return; }
  const button = qs("#warmModel");
  button.textContent = "预热中";
  button.disabled = true;
  const result = await api("/api/warmup", { token: state.token });
  button.disabled = false;
  button.textContent = "预热模型";
  if (!result.ok) { alert(result.error || "预热失败"); return; }
  state.model = { ...(result.status || {}), sandbox: (state.model && state.model.sandbox) || {} };
  renderModelStatus();
  alert("模型预热完成，用时 " + result.seconds + " 秒");
}

async function releaseModel() {
  if (!state.token) { alert("请先登录"); return; }
  const result = await api("/api/unload", { token: state.token });
  if (!result.ok) { alert(result.error || "释放失败"); return; }
  state.model = { ...(result.status || {}), sandbox: (state.model && state.model.sandbox) || {} };
  await loadModelStatus();
}

function renderTable(mode = "controls") {
  const rows = DATA[mode];
  qs("#modelTable").innerHTML = "<div class='table-row header'><span>组别</span><span>执行</span><span>变异</span><span>Attempt 变异</span><span>边界</span><span>失败</span></div>" +
    rows.map((row) => "<div class='table-row'><b>" + row[0] + "</b><span>" + pct(row[1]) + "</span><span>" + pct(row[2]) + "</span><span>" + pct(row[3]) + "</span><span>" + pct(row[4]) + "</span><span>" + fixed(row[5]) + "</span></div>").join("");
}

function renderAudit() {
  qs("#failureCards").innerHTML = DATA.failures
    .map(([name, text]) => "<article class='audit-item'><b>" + name + "</b><p>" + text + "</p></article>")
    .join("");
  qs("#auditTable").innerHTML = "<div class='table-row header'><span>字段</span><span>一致率</span><span>Kappa</span><span>分歧</span></div>" +
    DATA.audit.map((row) => "<div class='table-row'><b>" + row[0] + "</b><span>" + pct(row[1]) + "</span><span>" + fixed(row[2]) + "</span><span>" + row[4] + " / " + row[3] + "</span></div>").join("");
}

function heatColor(value) {
  if (value >= 85) return "#0f9488";
  if (value >= 70) return "#2563eb";
  if (value >= 55) return "#b7791f";
  return "#c2413a";
}

function analysisRows() {
  const own = (state.history || []).filter((item) => !item.user || item.user === state.session.user);
  if (!state.selectedRecords.size) return own;
  return own.filter((item) => state.selectedRecords.has(Number(item.id)));
}

function aggregateHistory(rows) {
  const targets = new Map();
  rows.forEach((item) => {
    const bucket = targets.get(item.target) || { target: item.target, count: 0, score: 0, line: 0, branch: 0, mutation: 0, mutationcount: 0 };
    bucket.count += 1;
    bucket.score += Number(item.score || 0);
    bucket.line += Number(item.line || 0);
    bucket.branch += Number(item.branch || 0);
    if (item.mutationvalid !== false && Number(item.mutationvalid ?? 1) !== 0) {
      bucket.mutation += Number(item.mutation || 0);
      bucket.mutationcount += 1;
    }
    targets.set(item.target, bucket);
  });
  return Array.from(targets.values()).map((item) => ({
    target: item.target,
    count: item.count,
    score: item.score / item.count,
    line: item.line / item.count,
    branch: item.branch / item.count,
    mutation: item.mutationcount ? item.mutation / item.mutationcount : null,
  })).sort((a, b) => b.score - a.score || b.count - a.count);
}

function chartBase(id) {
  const canvas = qs(id);
  if (!canvas) return null;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return { canvas, ctx };
}

function drawScoreChart(groups) {
  const base = chartBase("#scoreCanvas");
  if (!base) return;
  const { canvas, ctx } = base;
  if (!groups.length) {
    ctx.fillStyle = "#667085";
    ctx.font = "14px system-ui";
    ctx.fillText("暂无可分析记录", 28, 46);
    return;
  }
  const left = 48;
  const right = 20;
  const top = 24;
  const bottom = 54;
  const height = canvas.height - top - bottom;
  ctx.font = "12px system-ui";
  ctx.textAlign = "right";
  for (let value = 0; value <= 100; value += 25) {
    const y = top + height - (value / 100) * height;
    ctx.strokeStyle = "#e3e8ef";
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(canvas.width - right, y);
    ctx.stroke();
    ctx.fillStyle = "#667085";
    ctx.fillText(String(value), left - 8, y + 4);
  }
  const slot = (canvas.width - left - right) / groups.length;
  const width = Math.min(42, slot * 0.56);
  groups.forEach((item, index) => {
    const x = left + index * slot + (slot - width) / 2;
    const barHeight = Math.max(2, (item.score / 100) * height);
    ctx.fillStyle = item.score >= 75 ? "#0f9488" : item.score >= 55 ? "#b7791f" : "#c2413a";
    ctx.fillRect(x, top + height - barHeight, width, barHeight);
    ctx.fillStyle = "#152033";
    ctx.textAlign = "center";
    ctx.font = "700 12px system-ui";
    ctx.fillText(String(Math.round(item.score)), x + width / 2, top + height - barHeight - 7);
    ctx.fillStyle = "#667085";
    ctx.font = "11px system-ui";
    const name = item.target.length > 11 ? item.target.slice(0, 10) + "…" : item.target;
    ctx.fillText(name, x + width / 2, canvas.height - 24);
  });
}

function renderGate(rows) {
  const node = qs("#gateMetrics");
  const stateNode = qs("#gateState");
  if (!node || !stateNode) return;
  if (!rows.length) {
    stateNode.textContent = "待评估";
    stateNode.className = "pill";
    node.innerHTML = `<div class="empty">暂无可分析记录</div>`;
    return;
  }
  const average = (key, scale = 1) => rows.reduce((sum, item) => sum + Number(item[key] || 0), 0) / rows.length * scale;
  const mutationRows = rows.filter((item) => item.mutationvalid !== false && Number(item.mutationvalid ?? 1) !== 0);
  const averageMutation = mutationRows.length ? mutationRows.reduce((sum, item) => sum + Number(item.mutation || 0), 0) / mutationRows.length * 100 : null;
  const metrics = [
    ["质量分", average("score"), 75],
    ["行覆盖", average("line", 100), 80],
    ["分支覆盖", average("branch", 100), 70],
    ["变异分数", averageMutation, 60],
  ];
  const rejected = rows.some((item) => item.status === "丢弃");
  const review = rows.some((item) => item.status === "复核");
  const allMetricsPass = metrics.every((item) => item[1] !== null && item[1] >= item[2]);
  const status = rejected || !allMetricsPass ? "未通过" : review ? "需复核" : "通过";
  stateNode.textContent = status;
  stateNode.className = "pill gate-" + (status === "通过" ? "pass" : status === "需复核" ? "review" : "fail");
  node.innerHTML = metrics.map(([label, value, threshold]) => {
    const available = value !== null;
    const width = available ? value : 0;
    const display = available ? Math.round(value) : "—";
    return `<div class="gate-row"><div class="gate-label"><b>` + label + `</b><span>门槛 ` + threshold + `</span></div><div class="gate-track"><i style="width:` + Math.max(0, Math.min(100, width)) + `%;background:` + heatColor(width) + `"></i></div><strong>` + display + `</strong></div>`;
  }).join("");
}

function drawEvidenceChart(groups) {
  const canvas = qs("#evidenceCanvas");
  if (!canvas) return;
  canvas.height = Math.max(270, 82 + groups.length * 62);
  const base = chartBase("#evidenceCanvas");
  if (!base) return;
  const { ctx } = base;
  if (!groups.length) {
    ctx.fillStyle = "#667085";
    ctx.font = "14px system-ui";
    ctx.fillText("暂无可分析记录", 28, 46);
    return;
  }
  const left = 180;
  const right = 34;
  const top = 64;
  const plotWidth = canvas.width - left - right;
  const colors = ["#0f9488", "#2563eb", "#b7791f"];
  const labels = ["行覆盖", "分支覆盖", "变异分数"];
  labels.forEach((label, index) => {
    const x = left + index * 118;
    ctx.fillStyle = colors[index];
    ctx.fillRect(x, 18, 12, 12);
    ctx.fillStyle = "#536176";
    ctx.font = "12px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(label, x + 18, 29);
  });
  for (let value = 0; value <= 100; value += 25) {
    const x = left + plotWidth * value / 100;
    ctx.strokeStyle = "#e3e8ef";
    ctx.beginPath();
    ctx.moveTo(x, top - 8);
    ctx.lineTo(x, canvas.height - 18);
    ctx.stroke();
    ctx.fillStyle = "#7a8699";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(String(value), x, top - 18);
  }
  groups.forEach((item, index) => {
    const y = top + index * 62;
    const name = item.target.length > 21 ? item.target.slice(0, 20) + "…" : item.target;
    ctx.fillStyle = "#152033";
    ctx.font = "700 12px system-ui";
    ctx.textAlign = "right";
    ctx.fillText(name, left - 14, y + 25);
    [item.line, item.branch, item.mutation].forEach((raw, metricIndex) => {
      const available = raw !== null;
      const value = available ? Math.max(0, Math.min(100, raw * 100)) : 0;
      const barY = y + metricIndex * 12;
      ctx.fillStyle = "#edf1f6";
      ctx.fillRect(left, barY, plotWidth, 7);
      if (available) {
        ctx.fillStyle = colors[metricIndex];
        ctx.fillRect(left, barY, plotWidth * value / 100, 7);
      }
      ctx.fillStyle = "#536176";
      ctx.font = "10px system-ui";
      ctx.textAlign = "left";
      ctx.fillText(available ? Math.round(value) + "%" : "—", Math.min(canvas.width - 31, left + plotWidth * value / 100 + 7), barY + 7);
    });
  });
}

function renderBrief() {
  const rows = analysisRows();
  const groups = aggregateHistory(rows);
  const visibleGroups = groups.slice(0, 10);
  const scope = qs("#analysisScope");
  if (scope) {
    const range = state.selectedRecords.size ? "已选 " + rows.length + " 条" : "全部 " + rows.length + " 条";
    scope.textContent = range + (groups.length > 10 ? " · 前 10 个函数" : "");
  }
  drawScoreChart(visibleGroups);
  renderGate(rows);
  drawEvidenceChart(visibleGroups);
}

function buildReport() {
  const rows = analysisRows();
  const total = rows.length;
  const avg = (key) => rows.reduce((sum, item) => sum + Number(item[key] || 0), 0) / Math.max(1, total);
  const passed = rows.filter((item) => Number(item.passed || 0) === 1).length;
  const validMutationRows = rows.filter((item) => item.mutationvalid !== false && Number(item.mutationvalid ?? 1) !== 0);
  const averageMutation = validMutationRows.reduce((sum, item) => sum + Number(item.mutation || 0), 0) / Math.max(1, validMutationRows.length);
  const statusCountLocal = (status) => rows.filter((item) => item.status === status).length;
  const targets = new Map();
  rows.forEach((item) => {
    const current = targets.get(item.target) || { count: 0, score: 0, mutation: 0, mutationcount: 0 };
    current.count += 1;
    current.score += Number(item.score || 0);
    if (item.mutationvalid !== false && Number(item.mutationvalid ?? 1) !== 0) {
      current.mutation += Number(item.mutation || 0);
      current.mutationcount += 1;
    }
    targets.set(item.target, current);
  });
  const targetLines = Array.from(targets.entries())
    .sort((a, b) => b[1].count - a[1].count || (b[1].score / Math.max(1, b[1].count)) - (a[1].score / Math.max(1, a[1].count)))
    .slice(0, 8)
    .map(([target, item]) => target + "：评估 " + item.count + " 次，平均质量分 " + Math.round(item.score / Math.max(1, item.count)) + "，平均变异 " + (item.mutationcount ? pct(item.mutation / item.mutationcount) : "—") + "。");
  const recentLines = rows.slice(0, 10).map((item) => "#" + item.id + "  " + item.target + "  " + item.status + "  质量分=" + item.score + "  变异=" + mutationText(item, item.mutation || 0) + "  " + item.created);
  const fallback = summarizeRun();
  const lines = [
    "STVR 项目整体评估报告",
    "报告范围：" + (state.selectedRecords.size ? "所选评估记录" : "当前用户全部函数评估记录"),
    "用户：" + state.session.user + " / " + state.session.role,
    "项目：" + state.session.project,
    "生成时间：" + new Date().toLocaleString(),
    "",
    total ? "可信测试：累计评估 " + total + " 次，覆盖目标文件/函数 " + targets.size + " 个。" : "可信测试：当前还没有历史评估记录，页面暂按当前输入生成预览。",
    total ? "执行验证：通过 " + passed + " 次，通过率 " + pct(passed / Math.max(1, total)) + "。" : "执行验证：等待完成第一次评估后生成项目级通过率。",
    total ? "质量证据：平均质量分 " + Math.round(avg("score")) + "，平均行覆盖 " + pct(avg("line")) + "，平均分支覆盖 " + pct(avg("branch")) + "，有效记录平均变异分数 " + (validMutationRows.length ? pct(averageMutation) : "—") + "。" : "质量证据：当前候选平均质量分 " + Math.round(fallback.avg) + "，平均变异分数 " + fixed(fallback.mutation) + "。",
    total ? "缺陷诊断：采纳 " + statusCountLocal("采纳") + " 次，复核 " + statusCountLocal("复核") + " 次，丢弃 " + statusCountLocal("丢弃") + " 次，平均失败数 " + fixed(avg("failures")) + "。" : "缺陷诊断：完成评估后会汇总复核、丢弃和失败记录。",
    "修复推荐：优先处理复核和丢弃记录，补强低变异、低覆盖和失败日志集中的目标函数。",
    "",
    "重点函数概览：",
    ...(targetLines.length ? targetLines : ["暂无历史函数记录。"]),
    "",
    "最近评估记录：",
    ...(recentLines.length ? recentLines : ["暂无历史评估记录。"]),
    "",
    "报告说明：评估报告页用于项目级质量汇总；历史记录详情用于查看和下载单次函数评估报告。",
  ];
  return lines.join("\n");
}

function renderReport() {
  const node = qs("#reportText");
  if (node) node.value = buildReport();
}

function drawexternal() {
  const canvas = qs("#externalCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = 48 + i * 58;
    ctx.beginPath();
    ctx.moveTo(58, y);
    ctx.lineTo(w - 28, y);
    ctx.stroke();
  }
  ctx.fillStyle = "#667085";
  ctx.font = "13px system-ui";
  ctx.fillText("蓝：执行通过率    绿：变异分数    橙：失败日志反向分", 58, 28);
  DATA.external.forEach((row, index) => {
    const x = 92 + index * 145;
    const baseY = h - 54;
    const passH = row[1] * 520;
    const mutH = row[2] * 360;
    const failH = Math.max(0.05, 1.35 - row[5]) * 220;
    [[0, passH, "#2563eb"], [38, mutH, "#0f9488"], [76, failH, "#b7791f"]].forEach(([offset, height, color]) => {
      ctx.fillStyle = color;
      ctx.fillRect(x + offset, baseY - height, 24, height);
    });
    ctx.fillStyle = "#111827";
    ctx.font = "13px system-ui";
    ctx.fillText(row[0], x + 18, h - 22);
  });
}

async function runDemo() {
  if (state.running) return;
  if (!selectedFunctionName()) {
    alert("未识别到待测业务函数。单函数评估请输入函数实现；包含业务源码和测试文件的项目请在批量评估中导入。");
    return;
  }
  state.running = true;
  const button = qs("#runButton");
  button.disabled = true;
  button.textContent = "评估中";
  qs("#runState").textContent = "运行中";
  const coverageNode = qs("#coverageState");
  if (coverageNode) coverageNode.textContent = "分析中";
  renderFlow(0, 0);
  renderJobProgress({ status: "running", stage: "prepare", detail: "准备评估任务", progress: 1, elapsed: 0, eta: 0 });
  try {
    const result = await requestSandbox(renderJobProgress);
    if (result && result.ok) {
      state.live = result;
      qs("#runState").textContent = "评估完成";
      if (coverageNode) coverageNode.textContent = "真实指标";
      state.lastRun = new Date();
      renderAllDynamic();
      await loadHistory();
      await loadDashboard();
    } else {
      state.live = null;
      qs("#runState").textContent = "评估失败";
      if (coverageNode) coverageNode.textContent = "未生成指标";
      alert((result && result.error) || "评估失败");
    }
    state.lastRun = new Date();
    renderAllDynamic();
  } finally {
    state.running = false;
    button.disabled = false;
    button.textContent = "开始评估";
  }
}
function exportReport() {
  renderReport();
  downloadBlob("report.txt", qs("#reportText").value, "text/plain;charset=utf-8");
}

function renderAllDynamic() {
  renderInspector();
  renderMetrics();
  renderRuntime();
  renderSelected();
  renderReport();
  renderBrief();
}

function wire() {
  qs("#loginForm").addEventListener("submit", login);
  qs("#loginTab").addEventListener("click", () => setAuthMode("login"));
  qs("#registerTab").addEventListener("click", () => setAuthMode("register"));
  qsa(".nav-btn").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.classList.contains("hidden")) return;
      switchView(button.dataset.view);
    });
  });
  qs("#functionSelect").addEventListener("change", (event) => {
    state.targetfunction = event.target.value;
    resetLive();
    renderAllDynamic();
  });
  qs("#sourceCode").addEventListener("input", () => {
    renderFunctionTargets();
    resetLive();
    renderAllDynamic();
  });
  qs("#projectFiles").addEventListener("change", readProjectFiles);
  qs("#projectFolder").addEventListener("change", readProjectFiles);
  qs("#archiveFile").addEventListener("change", readArchiveFile);
  qs("#clearBatch").addEventListener("click", clearBatchInput);
  qs("#frameworkSelect").addEventListener("change", () => { resetLive(); renderAllDynamic(); });
  qs("#engineSelect").addEventListener("change", () => { resetLive(); renderAllDynamic(); });
  qs("#candidateCount").addEventListener("input", () => { resetLive(); renderAllDynamic(); });
  qs("#verifyLevel").addEventListener("input", () => { resetLive(); renderAllDynamic(); });
  qs("#sampleSelect").addEventListener("change", (event) => {
    if (event.target.value === "") return;
    loadSample(Number(event.target.value));
    event.target.value = "";
  });
  qs("#runButton").addEventListener("click", runDemo);
  qs("#logoutButton").addEventListener("click", logout);
  qs("#exportButton").addEventListener("click", exportReport);
  qs("#refreshReport").addEventListener("click", () => { renderBrief(); renderReport(); });
  qs("#downloadPdf").addEventListener("click", openProjectReportDownload);
  qs("#reloadHistory").addEventListener("click", loadHistory);
  qs("#analyzeHistory").addEventListener("click", analyzeHistory);
  qs("#deleteHistory").addEventListener("click", deleteHistory);
  qs("#downloadTest").addEventListener("click", downloadCurrentTest);
  qs("#downloadRecordTest").addEventListener("click", () => openRecordDownload("test"));
  qs("#downloadRecordPdf").addEventListener("click", () => openRecordDownload("pdf"));
  qs("#reloadUsers").addEventListener("click", loadUsers);
  qs("#warmModel").addEventListener("click", warmModel);
  qs("#releaseModel").addEventListener("click", releaseModel);
  qs("#runBatch").addEventListener("click", runBatch);
  qs("#batchLimit").addEventListener("change", renderBatchTargets);
  qs("#userForm").addEventListener("submit", saveUser);
  qs("#deleteUser").addEventListener("click", removeUser);
  qs("#tableMode").addEventListener("change", (event) => renderTable(event.target.value));
}

function safeInit(name, fn) {
  try {
    fn();
  } catch (error) {
    console.error("init failed:", name, error);
  }
}

safeInit("wire", wire);
safeInit("roleNav", applyRoleNav);
safeInit("sample", renderSampleSelect);
safeInit("source", () => { qs("#sourceCode").value = DATA.targets[0].code; });
safeInit("targets", renderTargets);
safeInit("flow", renderFlow);
safeInit("evidence", renderEvidence);
safeInit("funnel", renderFunnel);
safeInit("assets", renderAssets);
safeInit("table", renderTable);
safeInit("batchTargets", renderBatchTargets);
safeInit("batchResults", renderBatchResults);
safeInit("audit", renderAudit);
safeInit("brief", renderBrief);
safeInit("dynamic", renderAllDynamic);
safeInit("external", drawexternal);
safeInit("batchImport", renderBatchImportState);
safeInit("modelStatus", renderModelStatus);
safeInit("projectDash", renderProjectDash);
safeInit("recordDetail", renderRecordDetail);
safeInit("loadModelStatus", loadModelStatus);
safeInit("loadAssets", loadAssets);
