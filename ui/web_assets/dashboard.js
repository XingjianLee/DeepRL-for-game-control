"use strict";

const get = (id) => document.getElementById(id);
const selectors = {game: get("game"), algorithm: get("algorithm"), run: get("run"), seed: get("seed")};
const view = {
  error: get("error"), frame: get("frame"), empty: get("stage-empty"),
  play: get("play"), single: get("single"), speed: get("speed"), status: get("status"),
  score: get("score"), scoreLabel: get("score-label"), reward: get("reward"),
  steps: get("steps"), action: get("action"), hint: get("manual-hint"),
  setting: get("reward-setting"), body: get("result-body"),
};
const hints = {
  snake: "方向键转向", flap: "空格扇翅", lander: "↑ / 空格：主发动机；← / →：侧发动机",
  "lander-continuous": "↑ / 空格：主推力；← / →：侧推力。可同时按住。",
};
const movement = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"]);
const keys = new Set();
let catalog = null;
let mode = "ai";
let tab = "curve";
let result = null;
let evaluationIndex = 0;
let resultRequest = 0;
let session = null;
let state = null;
let inFlight = null;
let playing = false;
let speed = 1;
let tap = null;
let accumulated = 0;
let previous = 0;
let activeSeed = 20000;
let rankingRequest = 0;

const lander = () => selectors.game.value.startsWith("lander");
const model = () => catalog?.runs.find((run) => run.id === selectors.run.value);
const format = (number, digits = 0) => number.toLocaleString("zh-CN", {
  minimumFractionDigits: digits, maximumFractionDigits: digits,
});

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function options(select, entries, preferred) {
  select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
  if (entries.some(([value]) => value === preferred)) select.value = preferred;
  select.disabled = entries.length === 0;
}

async function request(path, body) {
  const options = body ? {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  } : {};
  const reply = await fetch(path, options);
  const answer = await reply.json();
  if (!reply.ok) throw new Error(answer.error || "读取失败，请刷新重试。");
  return answer;
}

function report(error) {
  view.error.hidden = !error;
  view.error.textContent = error || "";
}

async function refresh(initial = false) {
  try {
    const selectedGame = selectors.game.value;
    catalog = await request("/api/catalog");
    if (initial) {
      activeSeed = catalog.seed;
      selectors.seed.value = activeSeed;
      syncSlider();
    }
    options(selectors.game, Object.entries(catalog.games), selectedGame || catalog.game);
    changeGame(initial ? "" : selectors.algorithm.value, initial ? "" : selectors.run.value);
  } catch (error) {
    report(error.message);
    view.empty.textContent = "无法连接本地实验台";
  }
}

function changeGame(preferredAlgorithm = "", preferredModel = "") {
  const candidates = catalog.runs.filter((run) => run.game === selectors.game.value);
  const algorithms = [...new Set(candidates.map((run) => run.algorithm))];
  options(selectors.algorithm, algorithms.map((value) => [value, value.toUpperCase()]),
          preferredAlgorithm || (algorithms.includes("dqn") && lander() ? "dqn" : algorithms[0]));
  changeAlgorithm(preferredModel);
}

function changeAlgorithm(preferredModel = "") {
  const candidates = catalog.runs.filter((run) => run.game === selectors.game.value &&
                                                 run.algorithm === selectors.algorithm.value);
  options(selectors.run, candidates.map((run) => [
    run.id, format(run.steps) + " 步 · 种子 " + run.seed,
  ]), preferredModel);
  changeModel();
}

function syncSlider() {
  const slider = get("seed-slider");
  slider.max = Math.max(100000, Number(selectors.seed.value), activeSeed);
  slider.value = selectors.seed.value;
}

function changeModel() {
  view.setting.textContent = model() ? (model().shaped ? "训练使用塑形奖励" : "训练使用原环境奖励") : "";
  get("ai").disabled = !model();
  if (!model()) mode = "manual";
  updateMode();
  loadResults();
  resetEpisode();
}

function updateMode() {
  for (const name of ["ai", "manual"]) get(name).setAttribute("aria-pressed", String(mode === name));
  get("manual-controls").hidden = mode !== "manual";
  view.hint.textContent = hints[selectors.game.value] || "";
  view.speed.disabled = mode === "manual";
}

async function loadResults() {
  const token = ++resultRequest;
  result = null;
  evaluationIndex = 0;
  drawResults(model() ? "正在读取训练记录" : "暂无已保存模型");
  if (!model()) return;
  try {
    const data = await request("/api/result", {run: model().id});
    if (token !== resultRequest) return;
    result = data;
    drawResults();
  } catch (error) {
    if (token === resultRequest) drawResults(error.message);
  }
}

async function resetEpisode() {
  playing = false;
  inFlight = null;
  state = null;
  tap = null;
  keys.clear();
  accumulated = 0;
  session = crypto.randomUUID();
  const id = session;
  view.frame.hidden = true;
  view.empty.hidden = false;
  view.empty.textContent = "正在读取游戏画面";
  view.scoreLabel.textContent = lander() ? "着陆" : "分数";
  report("");
  updatePlayback();
  try {
    const data = await request("/api/reset", {
      session: id, game: selectors.game.value, run: model()?.id, seed: activeSeed, mode,
    });
    if (id !== session) return false;
    showState(data);
    return true;
  } catch (error) {
    if (id === session) {
      report(error.message);
      view.empty.textContent = "暂时无法显示游戏";
      updatePlayback();
    }
    return false;
  }
}

function updatePlayback() {
  view.play.disabled = !state;
  view.single.disabled = !state || state.done || playing || inFlight === session;
  view.play.textContent = playing ? "暂停" : state?.done ? "再玩一局" : state?.steps ? "继续" : "开始";
  view.status.textContent = state?.reason || (playing ? "进行中" : state?.steps ? "已暂停" : state ? "就绪" : "—");
  view.score.textContent = state ? (lander() ? state.score ? "成功" : "—" : format(state.score)) : "—";
  view.reward.textContent = state ? format(state.return, 1) : "—";
  view.steps.textContent = state ? format(state.steps) : "—";
  view.action.textContent = state?.action || "—";
}

function showState(data) {
  state = data;
  view.frame.dataset.game = selectors.game.value;
  view.frame.src = data.frame;
  view.frame.hidden = false;
  view.empty.hidden = true;
  if (state.done) playing = false;
  if (state.done && tab === "leaderboard") drawLeaderboard();
  updatePlayback();
}

async function advance(count) {
  if (!state || state.done || inFlight === session) return;
  const token = session;
  inFlight = token;
  const pressed = tap;
  tap = null;
  try {
    const data = await request("/api/step", {session: token, count, keys: [...keys], tap: pressed,
                                            player: get("player").value});
    if (session === token) showState(data);
  } catch (error) {
    if (session === token) {
      playing = false;
      report(error.message);
    }
  } finally {
    if (inFlight === token) {
      inFlight = null;
      updatePlayback();
    }
  }
}

async function toggle() {
  if (!state) return;
  if (state.done && !(await resetEpisode())) return;
  playing = !playing;
  previous = performance.now();
  accumulated = 0;
  updatePlayback();
}

function tick(now) {
  if (playing && state) {
    const factor = mode === "ai" ? speed : 1;
    accumulated = Math.min(8, accumulated + (now - previous) * state.fps * factor / 1000);
    const count = Math.floor(accumulated);
    if (count > 0 && !inFlight) {
      accumulated -= count;
      advance(count);
    }
  }
  previous = now;
  requestAnimationFrame(tick);
}

const svgNS = "http://www.w3.org/2000/svg";

function svgNode(tag, attrs, text) {
  const element = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function chart(points, steps) {
  const svg = svgNode("svg", {class: "chart", viewBox: "0 0 540 300", role: "img",
                              "aria-label": "训练过程折线图"});
  const x0 = 53, x1 = 524, y0 = 12, y1 = 242;
  const values = points.map((point) => point[1]);
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const rough = Math.max(1, (max - min) / 5);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const spacing = ([1, 2, 5, 10].find((n) => n >= rough / magnitude)) * magnitude;
  const low = Math.floor(min / spacing) * spacing;
  const high = Math.max(low + spacing, Math.ceil(max / spacing) * spacing);
  const x = (step) => x0 + step / steps * (x1 - x0);
  const y = (value) => y1 - (value - low) / (high - low) * (y1 - y0);
  for (let value = low; value <= high; value += spacing) {
    const py = y(value);
    svg.append(svgNode("line", {x1: x0, x2: x1, y1: py, y2: py, class: "grid"}),
               svgNode("text", {x: x0 - 9, y: py + 4, "text-anchor": "end"}, format(value)));
  }
  for (let i = 0; i <= 4; i++) {
    const value = steps * i / 4;
    const label = value >= 10000 ? format(value / 10000, value % 10000 === 0 ? 0 : 1) + "万" : format(value);
    svg.append(svgNode("text", {x: x(value), y: 264, "text-anchor": "middle"}, label));
  }
  svg.append(svgNode("text", {x: (x0 + x1) / 2, y: 291, "text-anchor": "middle"}, "环境步数"));
  svg.append(svgNode("polyline", {class: "trace", points: points.map(([step, value]) =>
    [x(step).toFixed(2), y(value).toFixed(2)].join(",")).join(" ")}));
  const [lastStep, lastValue] = points.at(-1);
  svg.append(svgNode("circle", {class: "point", cx: x(lastStep), cy: y(lastValue), r: 3.5}));
  return svg;
}

function evaluationHeader(data) {
  const row = node("div", undefined, "chart-heading");
  row.append(node("span", data.episodes + " 局 · 种子 " + data.seeds.join("–"), "note"));
  if (result.evaluations.length > 1) {
    const select = node("select", undefined, "dataset");
    select.setAttribute("aria-label", "选择评估记录");
    options(select, result.evaluations.map((entry, i) => [String(i), entry.label]), String(evaluationIndex));
    select.addEventListener("change", () => {
      evaluationIndex = Number(select.value);
      drawResults();
    });
    row.append(select);
  } else {
    row.append(node("span", data.label, "note"));
  }
  return row;
}

function drawResults(placeholder) {
  for (const button of document.querySelectorAll("[data-tab]")) {
    button.setAttribute("aria-pressed", String(button.dataset.tab === tab));
  }
  if (tab === "leaderboard") return drawLeaderboard();
  view.body.replaceChildren();
  const note = (text) => view.body.append(node("p", text, "note"));
  if (!result) return note(placeholder || "暂无训练记录");
  if (tab === "curve") {
    if (!result.curve.length) return note("暂无已完成回合。");
    const heading = node("div", undefined, "chart-heading");
    heading.append(node("strong", lander() ? "平均累计回报" : "平均游戏得分"),
                   node("span", format(result.trainingEpisodes) + " 个训练回合", "note"));
    const values = result.curve.map(([, value]) => value);
    const summary = node("dl", undefined, "curve-summary");
    for (const [label, value] of [["末段均值", values.at(-1)], ["最高段均值", Math.max(...values)]]) {
      const card = node("div");
      card.append(node("dt", label), node("dd", format(value, 2)));
      summary.append(card);
    }
    view.body.append(heading, summary, chart(result.curve, model().steps),
                     node("p", "曲线按环境步数分段，每个点为该段已结束回合的均值。", "note"));
    return;
  }
  const data = result.evaluations[evaluationIndex];
  if (!data) return note("当前模型没有已保存的评估。");
  view.body.append(evaluationHeader(data));
  if (tab === "comparison") {
    if (data.comparison.length < 2) return note("暂无满足相同训练步数、奖励和评估种子的对照记录。");
    const table = node("table");
    const head = table.createTHead().insertRow();
    for (const title of ["算法", "训练种子数", lander() ? "着陆率" : "平均分数", "平均回报"]) {
      head.append(node("th", title));
    }
    const body = table.createTBody();
    for (const entry of data.comparison) {
      const row = body.insertRow();
      for (const field of [entry.algorithm.toUpperCase(), entry.seeds,
        lander() ? format(entry.score * 100, 1) + "%" : format(entry.score, 2), format(entry.return, 1)]) {
        row.append(node("td", field));
      }
    }
    view.body.append(node("p", "相同训练步数、奖励与评估种子。", "note comparison-note"), table);
    return;
  }
  const metrics = node("dl", undefined, "metrics");
  const entries = lander() ? [["着陆率", format(data.meanScore * 100, 1) + "%"],
    ["平均回报", format(data.meanReturn, 1)], ["最高回报", format(data.bestReturn, 1)]] :
    [["平均分数", format(data.meanScore, 2)], ["最高分数", format(data.bestScore)],
     ["平均回报", format(data.meanReturn, 1)]];
  for (const [label, value] of entries) {
    const row = node("div");
    row.append(node("dt", label), node("dd", value));
    metrics.append(row);
  }
  view.body.append(metrics);
}

async function drawLeaderboard() {
  const token = ++rankingRequest;
  const game = selectors.game.value;
  const seed = activeSeed;
  const heading = node("p", "关卡种子 " + format(seed) + " · 单局成绩", "note comparison-note");
  view.body.replaceChildren(heading);
  try {
    const rows = await request("/api/leaderboard", {game, seed});
    if (tab !== "leaderboard" || token !== rankingRequest) return;
    if (!rows.length) {
      view.body.append(node("p", "还没有成绩。完成一局人工试玩或模型回放即可上榜。", "note"));
      return;
    }
    const table = node("table", undefined, "leaderboard");
    const head = table.createTHead().insertRow();
    for (const title of ["名次", "玩家 / 模型", lander() ? "着陆" : "分数", "回报", "操作"]) {
      head.append(node("th", title));
    }
    const body = table.createTBody();
    rows.forEach((entry, index) => {
      const row = body.insertRow();
      const identity = node("td", entry.player);
      const kind = node("small", entry.model ? "模型" : "人工", "note");
      identity.append(kind);
      const actions = node("td");
      const remove = node("button", "删除", "delete-score");
      remove.type = "button";
      remove.setAttribute("aria-label", `删除 ${entry.player} 的这一局成绩`);
      remove.addEventListener("click", async () => {
        if (!window.confirm(`删除「${entry.player}」的这一局成绩？`)) return;
        remove.disabled = true;
        try {
          await request("/api/leaderboard/delete", {id: entry.id});
          if (tab === "leaderboard" && token === rankingRequest) drawLeaderboard();
        } catch (error) {
          remove.disabled = false;
          report(error.message);
        }
      });
      actions.append(remove);
      row.append(node("td", index + 1), identity,
                 node("td", lander() ? entry.score ? "成功" : "—" : format(entry.score)),
                 node("td", format(entry.reward, 1)), actions);
    });
    view.body.append(table);
  } catch (error) {
    if (tab === "leaderboard" && token === rankingRequest) view.body.append(node("p", error.message, "note"));
  }
}

selectors.game.addEventListener("change", () => changeGame());
selectors.algorithm.addEventListener("change", () => changeAlgorithm());
selectors.run.addEventListener("change", changeModel);
selectors.seed.addEventListener("input", syncSlider);
get("seed-slider").addEventListener("input", (event) => { selectors.seed.value = event.target.value; });
get("seed-form").addEventListener("submit", (event) => {
  event.preventDefault();
  activeSeed = Number(selectors.seed.value);
  syncSlider();
  resetEpisode();
  if (tab === "leaderboard") drawLeaderboard();
});
get("refresh").addEventListener("click", () => refresh());
for (const name of ["ai", "manual"]) {
  get(name).addEventListener("click", () => {
    if (mode !== name) {
      mode = name;
      updateMode();
      resetEpisode();
    }
  });
}
view.play.addEventListener("click", toggle);
view.single.addEventListener("click", () => advance(1));
get("reset").addEventListener("click", resetEpisode);
view.speed.addEventListener("click", () => {
  speed = speed === 4 ? 1 : speed * 2;
  view.speed.textContent = speed + "×";
});
for (const button of document.querySelectorAll("[data-tab]")) {
  button.addEventListener("click", () => {
    tab = button.dataset.tab;
    drawResults();
  });
}
window.addEventListener("keydown", (event) => {
  if (mode !== "manual" || ["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  if (!movement.has(event.code)) return;
  event.preventDefault();
  keys.add(event.code);
  if (!event.repeat) tap = event.code;
  if (!playing && state?.steps === 0) toggle();
});
window.addEventListener("keyup", (event) => keys.delete(event.code));
window.addEventListener("blur", () => { keys.clear(); tap = null; });
refresh(true);
requestAnimationFrame(tick);
