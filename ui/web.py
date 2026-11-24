"""Serve the dashboard and current experiment data on the loopback interface."""

from base64 import b64encode
from collections import OrderedDict
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
import sqlite3
from statistics import fmean
from threading import Lock

import gymnasium as gym
import pygame

from experiments.evaluate import ALGORITHMS, ENVIRONMENTS, load_agent, make_env
from games.lander import LANDER_GAMES
from ui.controls import lander_thrust, snake_turn
from ui.history import comparable_runs, load_runs

ASSETS = Path(__file__).with_name("web_assets")
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}
GAME_NAMES = {
    "snake": "贪吃蛇", "flap": "Flappy Bird", "lander": "月球着陆（离散）",
    "lander-continuous": "月球着陆（连续）",
}
KEYS = {"ArrowUp": pygame.K_UP, "ArrowRight": pygame.K_RIGHT, "ArrowDown": pygame.K_DOWN,
        "ArrowLeft": pygame.K_LEFT, "Space": pygame.K_SPACE}
REASONS = {
    "wall": "撞墙", "body": "撞到蛇身", "board_full": "填满棋盘", "collision": "碰撞",
    "landed": "成功着陆", "crash": "坠毁", "out_of_bounds": "飞出边界", "time_limit": "达到步数上限",
}
ACTIONS = {
    "snake": ("直行", "相对左转", "相对右转"),
    "flap": ("保持", "扇翅"),
    "lander": ("关闭推力", "左侧发动机", "主发动机", "右侧发动机"),
}


class Replay:
    def __init__(self, game, run, seed):
        self.game = game
        self.steps, self.total_reward = 0, 0.0
        self.reason = self.action = None
        self.done = False
        self.env = make_env(game, render_mode="rgb_array")
        try:
            self.agent = load_agent(run.path, self.env) if run else None
            self.obs, info = self.env.reset(seed=seed)
        except BaseException:
            self.env.close()
            raise
        self.score = info.get("score", 0)
        self.fps = 6 if game == "snake" and run is None else self.env.metadata.get("render_fps", 30)

    def advance(self, count, keys, tap):
        for index in range(count):
            if self.done:
                break
            pressed_once = tap if index == 0 else None
            if self.agent is not None:
                action = self.agent.predict(self.obs)
            elif self.game == "snake":
                action = (snake_turn(KEYS[pressed_once], self.env.unwrapped.heading)
                          if pressed_once in KEYS and pressed_once != "Space" else 0)
                action = action or 0
            elif self.game == "flap":
                action = int(pressed_once == "Space")
            else:
                pressed = {key_code: code in keys for code, key_code in KEYS.items()}
                action = lander_thrust(pressed, self.game == "lander-continuous")

            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.steps += 1
            self.total_reward += float(reward)
            self.score = info["score"]
            self.done = terminated or truncated
            if self.done:
                reason = info.get("reason") or ("collision" if terminated else "time_limit")
                self.reason = REASONS.get(reason, reason)
            if self.game == "lander-continuous":
                main, side = action
                self.action = f"主推 {main:+.2f} / 侧推 {side:+.2f}"
            else:
                self.action = ACTIONS[self.game][action]

    def frame(self):
        rgb = self.env.render()
        image = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        file = BytesIO()
        pygame.image.save(image, file, "frame.png")
        return {
            "frame": "data:image/png;base64," + b64encode(file.getvalue()).decode("ascii"),
            "steps": self.steps, "score": self.score, "return": self.total_reward,
            "reason": self.reason, "done": self.done, "action": self.action, "fps": self.fps,
        }


class Lab:
    def __init__(self, root, game, seed):
        self.root, self.game, self.seed = root, game, seed
        self.scores = root / "leaderboard.sqlite3"
        self.replays = OrderedDict()
        self.lock = Lock()
        self.refresh()

    def refresh(self):
        self.runs = {
            run.path.relative_to(self.root).as_posix(): run
            for run in load_runs(self.root, ENVIRONMENTS, ALGORITHMS)
        }

    def catalog(self):
        self.refresh()
        return {
            "game": self.game, "seed": self.seed, "games": GAME_NAMES,
            "runs": [{"id": key, "game": run.game, "algorithm": run.algorithm,
                      "seed": run.seed, "steps": run.steps, "label": run.completed_at,
                      "shaped": run.shaped} for key, run in self.runs.items()],
        }

    def result(self, model):
        run = self.runs.get(model)
        if run is None:
            raise ValueError("找不到所选模型。")
        datasets = []
        for data in run.evaluations:
            groups = comparable_runs(self.runs.values(), run, data)
            datasets.append({
                "label": data.label, "file": data.filename, "episodes": len(data.rows),
                "seeds": [min(data.seeds), max(data.seeds)], "meanScore": data.score,
                "meanReturn": data.reward, "bestScore": data.best_score,
                "bestReturn": data.best_return,
                "comparison": [{"algorithm": algo, "seeds": len(results),
                                "score": fmean(result.score for result in results.values()),
                                "return": fmean(result.reward for result in results.values())}
                               for algo, results in sorted(groups.items())],
            })
        return {"curve": run.curve, "trainingEpisodes": len(run.training), "evaluations": datasets}

    def reset(self, data):
        session = data.get("session")
        game, seed, mode = data.get("game"), data.get("seed"), data.get("mode")
        if not isinstance(session, str) or not re.fullmatch(r"[0-9a-f-]{36}", session):
            raise ValueError("会话无效，请刷新页面。")
        if game not in ENVIRONMENTS or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("游戏或关卡种子无效。")
        if mode not in ("ai", "manual"):
            raise ValueError("控制方式无效。")
        model = self.runs.get(data.get("run")) if mode == "ai" else None
        if mode == "ai" and (model is None or model.game != game):
            raise ValueError("请选择当前游戏的模型，或切换到人工试玩。")
        old = self.replays.pop(session, None)
        if old is not None:
            old.env.close()
        if len(self.replays) >= 4:
            self.replays.popitem(last=False)[1].env.close()
        replay = Replay(game, model, seed)
        replay.player = (f"{model.algorithm.upper()} · {model.steps:,} 步 · 种子 {model.seed}"
                         if model else "玩家")
        replay.game_seed = seed
        replay.recorded = False
        self.replays[session] = replay
        return replay.frame()

    def advance(self, data):
        session, count = data.get("session"), data.get("count")
        replay = self.replays.get(session)
        if replay is None:
            raise ValueError("请重开当前回合。")
        if type(count) is not int or not 1 <= count <= 8:
            raise ValueError("步数无效。")
        self.replays.move_to_end(session)
        player = data.get("player")
        if replay.agent is None and replay.steps == 0 and isinstance(player, str):
            replay.player = player.strip()[:20] or "玩家"
        replay.advance(count, data.get("keys", ()), data.get("tap"))
        if replay.done and not replay.recorded:
            self.root.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.scores)) as db, db:
                db.execute("""CREATE TABLE IF NOT EXISTS scores (
                    session TEXT PRIMARY KEY, game TEXT, seed INTEGER, player TEXT,
                    model INTEGER, score INTEGER, reward REAL, steps INTEGER)""")
                db.execute("INSERT OR IGNORE INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (session, replay.game, replay.game_seed, replay.player, replay.agent is not None,
                            int(replay.score), replay.total_reward, replay.steps))
            replay.recorded = True
        return replay.frame()

    def leaderboard(self, data):
        game, seed = data.get("game"), data.get("seed")
        if game not in ENVIRONMENTS or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("游戏或关卡种子无效。")
        if not self.scores.is_file():
            return []
        with closing(sqlite3.connect(self.scores)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(
                """SELECT session AS id, player, model, score, reward, steps FROM scores
                   WHERE game = ? AND seed = ?
                   ORDER BY score DESC, reward DESC, rowid ASC LIMIT 10""",
                (game, seed),
            )]

    def delete_score(self, data):
        record = data.get("id")
        if not isinstance(record, str) or not re.fullmatch(r"[0-9a-f-]{36}", record):
            raise ValueError("排行榜记录无效。")
        if not self.scores.is_file():
            return {"deleted": False}
        with closing(sqlite3.connect(self.scores)) as db, db:
            deleted = db.execute("DELETE FROM scores WHERE session = ?", (record,)).rowcount
        return {"deleted": bool(deleted)}

    def close(self):
        for replay in self.replays.values():
            replay.env.close()


def serve(root, game, seed, port):
    lab = Lab(root, game, seed)

    class Handler(BaseHTTPRequestHandler):
        def send(self, payload, kind, status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; "
                             "base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, data, status=200):
            self.send(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                      "application/json; charset=utf-8", status)

        def log_message(self, *args):
            pass

        def allowed_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}",
                                                 f"localhost:{self.server.server_port}")

        def do_GET(self):
            if not self.allowed_host():
                self.send_json({"error": "请通过本机地址访问。"}, 403)
            elif self.path in STATIC:
                filename, kind = STATIC[self.path]
                self.send((ASSETS / filename).read_bytes(), kind)
            elif self.path == "/api/catalog":
                with lab.lock:
                    self.send_json(lab.catalog())
            else:
                self.send_json({"error": "页面不存在。"}, 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.allowed_host() or (origin and origin != f"http://{self.headers.get('Host')}"):
                self.send_json({"error": "请通过本机地址访问。"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.headers.get_content_type() != "application/json" or not 0 < size <= 4096:
                    raise ValueError("请求无效。")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("请求无效。")
                with lab.lock:
                    if self.path == "/api/result":
                        response = lab.result(data.get("run"))
                    elif self.path == "/api/reset":
                        response = lab.reset(data)
                    elif self.path == "/api/step":
                        response = lab.advance(data)
                    elif self.path == "/api/leaderboard":
                        response = lab.leaderboard(data)
                    elif self.path == "/api/leaderboard/delete":
                        response = lab.delete_score(data)
                    else:
                        self.send_json({"error": "页面不存在。"}, 404)
                        return
            except (TypeError, ValueError, KeyError):
                self.send_json({"error": "请求无效，或所选回合已失效。请重开。"}, 400)
            except sqlite3.Error:
                self.send_json({"error": "排行榜暂时无法读写。"}, 422)
            except (OSError, ImportError, RuntimeError, pygame.error, gym.error.Error, EOFError):
                self.send_json({"error": "无法载入游戏或模型，请检查 DRL 环境和模型文件。"}, 422)
            else:
                self.send_json(response)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"强化学习实验台  http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            with lab.lock:
                lab.close()
