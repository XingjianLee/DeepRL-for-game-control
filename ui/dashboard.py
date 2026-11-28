"""Desktop dashboard for live gameplay and saved training experiments."""

from functools import lru_cache
from math import ceil
import pickle
from statistics import fmean

import gymnasium as gym
import numpy as np
import pygame

from experiments.evaluate import ALGORITHMS, ENVIRONMENTS, load_agent, make_env
from games.lander import LANDER_GAMES
from ui.controls import HEADINGS, THRUSTERS, lander_thrust, snake_turn
from ui.history import comparable_runs, load_runs

SIZE = (1264, 800)
BACKGROUND = "#0c131a"
PANEL = "#131e28"
FIELD = "#1b2935"
WELL = "#090f15"
BORDER = "#2a3b4a"
TEXT = "#e9f0f4"
MUTED = "#a2b4c2"
ACCENT = "#67e3b7"
ACCENT_INK = "#09271e"
AMBER = "#f6bc76"

GAMES = {
    "snake": ("贪吃蛇", "方向键转向"),
    "flap": ("Flappy Bird", "空格扇翅"),
    "lander": ("月球着陆 · 离散", "↑ / 空格启动主推力，← / → 启动侧推力"),
    "lander-continuous": ("月球着陆 · 连续", "↑ / 空格启动主推力，← / → 启动侧推力"),
}
REASONS = {
    "wall": "撞墙", "body": "撞到蛇身", "board_full": "填满棋盘", "collision": "发生碰撞",
    "landed": "成功着陆", "crash": "坠毁", "out_of_bounds": "飞出边界", "time_limit": "达到步数上限",
}
ACTIONS = {
    "snake": ("直行", "相对左转", "相对右转"),
    "flap": ("保持", "扇翅"),
    "lander": ("关闭推力", "左侧发动机", "主发动机", "右侧发动机"),
}


@lru_cache(maxsize=24)
def font(size, bold=False):
    family = pygame.font.match_font(("hiraginosansgb", "notosanscjk", "microsoftyahei", "arialunicode"))
    face = pygame.font.Font(family, size)
    face.set_bold(bold)
    return face


def compact(value):
    return f"{value / 1_000:g}k" if value >= 1_000 else f"{value:g}"


class Dashboard:
    def __init__(self, root, game, seed):
        self.canvas = pygame.Surface(SIZE)
        self.runs = load_runs(root, ENVIRONMENTS, ALGORITHMS)
        self.envs = {}
        self.game = game
        self.seed = seed
        self.manual = False
        self.speed = 1
        self.tab = "training"
        self.algorithm = ""
        self.model_index = self.dataset_index = 0
        self.buttons = []
        self.pointer = (-1, -1)
        self.choose_game(game)

    def choose_game(self, game):
        self.game = game
        self.options = tuple(run for run in self.runs if run.game == game)
        self.algorithm = self.options[0].algorithm if self.options else ""
        self.select_model(0)

    def choose_algorithm(self, algorithm):
        self.algorithm = algorithm
        self.select_model(0)

    def select_model(self, index):
        self.models = tuple(run for run in self.options if run.algorithm == self.algorithm)
        self.model_index = index
        self.run = self.models[index] if self.models else None
        self.select_dataset(0)
        self.restart()

    def select_dataset(self, index):
        self.dataset_index = index
        datasets = self.run.evaluations if self.run else ()
        self.dataset = datasets[index] if datasets else None
        self.groups = comparable_runs(self.options, self.run, self.dataset) if self.dataset else {}

    def set_manual(self, manual):
        self.manual = manual
        self.restart()

    def restart(self):
        self.playing = self.finished = False
        self.steps = self.score = self.pending_action = 0
        self.total_reward = 0.0
        self.last_action = None
        self.agent = self.frame = None
        self.status = "准备就绪"
        self.issue = ""
        try:
            spatial = bool(self.run and not self.manual and self.run.snake_spatial)
            env_key = (self.game, spatial)
            if env_key not in self.envs:
                options = {"spatial_features": True} if spatial else {}
                self.envs[env_key] = make_env(self.game, render_mode="rgb_array", **options)
            self.env = self.envs[env_key]
            self.obs, info = self.env.reset(seed=self.seed)
            self.score = info.get("score", 0)
            self.render_frame()
            if not self.manual:
                if self.run:
                    self.agent = load_agent(self.run.path, self.env)
                else:
                    self.status = "当前游戏没有已保存模型"
        except (OSError, RuntimeError, ValueError, ImportError, EOFError,
                pygame.error, gym.error.Error, pickle.UnpicklingError) as error:
            self.status = "暂时无法启动"
            self.issue = str(error).splitlines()[0] if str(error) else type(error).__name__

    def render_frame(self):
        frame = self.env.render()
        self.frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    def change_seed(self, delta):
        self.seed = max(0, self.seed + delta)
        self.restart()

    @property
    def ready(self):
        return self.frame is not None and (self.manual or self.agent is not None) and not self.issue

    def toggle(self):
        if self.finished:
            self.restart()
        if self.ready:
            self.playing = not self.playing
            self.next_frame = pygame.time.get_ticks()
            self.status = "进行中" if self.playing else "已暂停"

    def advance(self, *, render=True):
        if not self.ready or self.finished:
            return
        if self.agent is not None:
            action = self.agent.predict(self.obs)
        elif self.game in LANDER_GAMES:
            action = lander_thrust(pygame.key.get_pressed(), self.game == "lander-continuous")
        else:
            action = self.pending_action
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = action
        self.pending_action = 0
        self.steps += 1
        self.total_reward += float(reward)
        self.score = info.get("score", 0)
        if render:
            self.render_frame()
        self.finished = terminated or truncated
        if self.finished:
            self.playing = False
            reason = info.get("reason") or ("collision" if terminated else "time_limit")
            self.status = REASONS.get(reason, reason)
        else:
            self.status = "进行中" if self.playing else "已暂停"

    def tick(self, now):
        if self.playing and now >= self.next_frame:
            fps = 6 if self.manual and self.game == "snake" else self.env.metadata.get("render_fps", 30)
            factor = 1 if self.manual else self.speed
            interval = 1000 / (fps * factor)
            for _ in range(4):
                if not self.playing or now < self.next_frame:
                    break
                self.advance(render=False)
                self.next_frame += interval
            self.next_frame = max(self.next_frame, now)
            self.render_frame()

    def handle_key(self, key):
        if key == pygame.K_p or (key == pygame.K_SPACE and not self.manual):
            self.toggle()
        elif key == pygame.K_r:
            self.restart()
        elif key == pygame.K_n and not self.playing:
            self.advance()
        elif self.manual and self.ready and not self.finished:
            if self.game == "snake" and key in HEADINGS:
                action = snake_turn(key, self.env.unwrapped.heading)
                if action is not None:
                    self.pending_action = action
            elif self.game == "flap" and key == pygame.K_SPACE:
                self.pending_action = 1
            elif self.game not in LANDER_GAMES or key not in THRUSTERS:
                return
            if self.steps == 0 and not self.playing:
                self.toggle()

    def close(self):
        # Flappy Bird shuts down Pygame even for RGB rendering. Keep its environment
        # open while switching tabs, and close every environment only at exit.
        for env in self.envs.values():
            env.close()
        font.cache_clear()

    def click(self, pos):
        for rect, callback in self.buttons:
            if rect.collidepoint(pos):
                callback()
                return

    def text(self, value, x, y, size=17, color=TEXT, *, bold=False, align="left", width=None):
        face = font(size, bold)
        value = str(value)
        if width is not None and face.size(value)[0] > width:
            while value and face.size(value + "…")[0] > width:
                value = value[:-1]
            value += "…"
        label = face.render(value, True, color)
        if align == "right":
            x -= label.get_width()
        elif align == "center":
            x -= label.get_width() // 2
        self.canvas.blit(label, (x, y))

    def box(self, rect, fill=PANEL, edge=None, radius=12):
        pygame.draw.rect(self.canvas, fill, rect, border_radius=radius)
        if edge:
            pygame.draw.rect(self.canvas, edge, rect, width=1, border_radius=radius)

    def button(self, rect, label, action, *, selected=False, primary=False, enabled=True):
        rect = pygame.Rect(rect)
        hover = enabled and rect.collidepoint(self.pointer)
        fill = ACCENT if primary and enabled else ("#214436" if selected else FIELD)
        if hover and not primary and not selected:
            fill = "#283c4b"
        color = (ACCENT_INK if primary else ACCENT if selected else TEXT) if enabled else "#6c8191"
        self.box(rect, fill, ACCENT if selected or hover else BORDER, radius=8)
        self.text(label, rect.centerx, rect.centery - 10, 16, color,
                  bold=primary or selected, align="center", width=rect.width - 12)
        if enabled:
            self.buttons.append((rect, action))

    def draw(self, pointer=(-1, -1)):
        self.buttons.clear()
        self.pointer = pointer
        self.canvas.fill(BACKGROUND)
        self.text("强化学习实验台", 32, 29, 34, bold=True)
        self.text("游戏回放与训练结果", 33, 77, 17, MUTED)
        self.text(ENVIRONMENTS[self.game], 1232, 49, 19, ACCENT, align="right")
        for i, (game, (title, _)) in enumerate(GAMES.items()):
            self.button((32 + 180 * i, 116, 166, 40), title,
                        lambda game=game: self.choose_game(game), selected=self.game == game)

        self.draw_game()
        self.draw_results()
        self.text("P 暂停 / 继续   N 单步   R 重开   Esc 退出", 34, 761, 15, MUTED)
        if self.run:
            self.text(f"运行记录  {self.run.completed_at}", 1230, 761, 15, MUTED, align="right")
        return self.canvas

    def draw_game(self):
        self.box((32, 176, 700, 557), PANEL, BORDER)
        self.text("游戏画面", 52, 194, 21, bold=True)
        self.text(self.status, 712, 198, 16, AMBER if self.finished or self.issue else ACCENT,
                  align="right", width=290)
        viewport = pygame.Rect(52, 233, 660, 340)
        self.box(viewport, WELL, radius=8)
        if self.frame is not None:
            target = self.frame.get_rect().fit(viewport.inflate(-20, -16))
            target.center = viewport.center
            frame = pygame.transform.smoothscale(self.frame, target.size)
            self.canvas.blit(frame, target)
        if self.issue:
            self.text(self.issue, viewport.centerx, viewport.bottom - 29, 14, AMBER,
                      align="center", width=viewport.width - 30)
        elif self.frame is None:
            self.text("暂无游戏画面", viewport.centerx, viewport.centery - 12, 19, MUTED, align="center")

        self.button((52, 587, 88, 38), "暂停" if self.playing else "再玩一局" if self.finished
                    else "继续" if self.steps else "开始", self.toggle, primary=True, enabled=self.ready)
        self.button((150, 587, 64, 38), "重开", self.restart, enabled=self.frame is not None)
        self.button((224, 587, 64, 38), "单步", self.advance,
                    enabled=self.ready and not self.playing and not self.finished)
        self.button((300, 587, 62, 38), f"{self.speed}×", self.cycle_speed, enabled=not self.manual)
        self.button((502, 587, 44, 38), "-", lambda: self.change_seed(-1), enabled=self.seed > 0)
        self.text(f"种子 {self.seed}", 606, 596, 16, MUTED, align="center", width=110)
        self.button((668, 587, 44, 38), "+", lambda: self.change_seed(1))

        labels = (("着陆" if self.game in LANDER_GAMES else "分数",
                   ("成功" if self.score else "—") if self.game in LANDER_GAMES else str(self.score)),
                  ("累计回报", f"{self.total_reward:.1f}"), ("运行步数", str(self.steps)))
        for i, (label, value) in enumerate(labels):
            x = 52 + 224 * i
            self.box((x, 640, 212, 54), FIELD, radius=8)
            self.text(label, x + 12, 658, 15, MUTED)
            self.text(value, x + 199, 653, 22, bold=True, align="right")
        action = "等待开始"
        if self.last_action is not None:
            if self.game == "lander-continuous":
                main, side = self.last_action
                action = f"主推指令 {main:+.2f}  /  侧推指令 {side:+.2f}"
            else:
                action = ACTIONS[self.game][self.last_action]
        self.text(f"动作  {action}", 53, 705, 15, MUTED)
        if self.manual:
            self.text(GAMES[self.game][1], 712, 705, 15, MUTED, align="right", width=355)

    def cycle_speed(self):
        self.speed = {1: 2, 2: 4, 4: 1}[self.speed]

    def draw_results(self):
        self.box((752, 176, 480, 557), PANEL, BORDER)
        self.text("控制与训练结果", 772, 194, 21, bold=True)
        self.button((1080, 190, 59, 34), "AI", lambda: self.set_manual(False), selected=not self.manual)
        self.button((1145, 190, 67, 34), "人工", lambda: self.set_manual(True), selected=self.manual)

        available = {run.algorithm for run in self.options}
        button_span = 444 // len(ALGORITHMS)
        for i, algorithm in enumerate(ALGORITHMS):
            self.button((772 + i * button_span, 240, button_span - 5, 35), algorithm.upper(),
                        lambda algorithm=algorithm: self.choose_algorithm(algorithm),
                        selected=algorithm == self.algorithm, enabled=algorithm in available)

        self.box((772, 287, 440, 62), FIELD, BORDER, radius=8)
        self.button((782, 298, 33, 39), "<", lambda: self.select_model(self.model_index - 1),
                    enabled=self.model_index > 0)
        self.button((1169, 298, 33, 39), ">", lambda: self.select_model(self.model_index + 1),
                    enabled=self.model_index + 1 < len(self.models))
        if self.run:
            label = f"{self.run.steps:,} 步  ·  训练种子 {self.run.seed}"
            detail = f"模型 {self.model_index + 1}/{len(self.models)}"
            detail += "  ·  训练含奖励塑形" if self.run.shaped else "  ·  原环境奖励"
            self.text(label, 992, 294, 17, bold=True, align="center", width=335)
            self.text(detail, 992, 321, 14, MUTED, align="center")
        else:
            self.text("没有可选模型", 992, 307, 17, MUTED, align="center")

        for i, (tab, label) in enumerate((("training", "训练曲线"), ("evaluation", "评估结果"),
                                        ("comparison", "算法对比"))):
            self.button((772 + 148 * i, 365, 144, 35), label,
                        lambda tab=tab: setattr(self, "tab", tab), selected=self.tab == tab)
        if self.tab == "training":
            self.draw_training()
        else:
            self.draw_evaluation(self.tab == "comparison")

    def draw_training(self):
        if self.run is None or not self.run.curve:
            self.text("没有可显示的训练日志", 992, 515, 17, MUTED, align="center")
            return
        metric = "累计回报" if self.game in LANDER_GAMES else "游戏分数"
        self.text(f"训练回合平均{metric}", 774, 420, 18, bold=True)
        self.text(f"{len(self.run.training):,} 个已完成回合", 1211, 423, 14, MUTED, align="right")
        plot = pygame.Rect(828, 469, 370, 166)
        values = [value for _, value in self.run.curve]
        low, high = min(0, min(values)), max(0, max(values))
        bound = max(1.0, high - low)
        if low < 0:
            low -= bound * 0.06
        high += bound * 0.10
        for i in range(5):
            y = plot.bottom - i * plot.height / 4
            value = low + i * (high - low) / 4
            pygame.draw.line(self.canvas, BORDER, (plot.left, y), (plot.right, y))
            self.text(f"{value:.0f}" if bound >= 8 else f"{value:.1f}", plot.left - 9,
                      round(y) - 8, 13, MUTED, align="right")
        points = [(plot.left + step / self.run.steps * plot.width,
                   plot.bottom - (value - low) / (high - low) * plot.height)
                  for step, value in self.run.curve]
        if len(points) > 1:
            pygame.draw.lines(self.canvas, ACCENT, False, points, width=3)
        pygame.draw.circle(self.canvas, ACCENT, points[-1], 5)
        for i in range(5):
            self.text(compact(self.run.steps * i / 4), plot.left + i * plot.width // 4,
                      plot.bottom + 11, 13, MUTED, align="center")
        self.text("训练环境步数", plot.centerx, 675, 15, MUTED, align="center")
        self.text("曲线按训练步数分段，统计各段结束的回合。", 774, 703, 13, MUTED)

    def draw_evaluation(self, compare):
        if self.dataset is None:
            self.text("该模型没有已保存的评估", 992, 509, 17, MUTED, align="center")
            self.text("运行 evaluate 并保存 CSV 后可查看结果", 992, 542, 14, MUTED, align="center")
            return
        self.text(self.dataset.label, 775, 423, 18, bold=True)
        self.text(f"{len(self.dataset.rows)} 局 · 种子 {min(self.dataset.seeds)}–{max(self.dataset.seeds)}",
                  969, 425, 13, MUTED, align="center", width=200)
        if len(self.run.evaluations) > 1:
            self.button((1100, 417, 110, 30), "切换数据",
                        lambda: self.select_dataset((self.dataset_index + 1) % len(self.run.evaluations)))
        if compare:
            self.draw_comparison()
            return

        landing = self.game in LANDER_GAMES
        labels = (("平均回报", f"{self.dataset.reward:.1f}"),
                  ("着陆率", f"{self.dataset.score:.1%}"),
                  ("最高回报", f"{self.dataset.best_return:.1f}")) if landing else (
                  ("平均分数", f"{self.dataset.score:.2f}"),
                  ("最高分数", f"{self.dataset.best_score:g}"),
                  ("平均回报", f"{self.dataset.reward:.1f}"))
        for i, (label, value) in enumerate(labels):
            x = 774 + i * 147
            self.box((x, 467, 139, 74), FIELD, BORDER, radius=8)
            self.text(label, x + 12, 475, 14, MUTED)
            self.text(value, x + 12, 499, 22, ACCENT if i == 0 else TEXT, bold=True)
        values = [reward if landing else score for _, score, reward in self.dataset.rows]
        low, high = min(values), max(values)
        counts, _ = np.histogram(values, bins=min(14, max(1, ceil(high - low))))
        self.text("评估回报分布" if landing else "评估分数分布", 775, 561, 16, bold=True)
        area = pygame.Rect(775, 591, 433, 79)
        gap = area.width / len(counts)
        for i, n in enumerate(counts):
            if n:
                height = round(area.height * n / max(counts))
                pygame.draw.rect(self.canvas, ACCENT,
                                 (round(area.x + i * gap), area.bottom - height,
                                  max(1, round(gap) - 4), height), border_radius=3)
        self.text(f"{low:g}" if not landing else f"{low:.0f}", area.left, 684, 14, MUTED)
        self.text(f"{high:g}" if not landing else f"{high:.0f}", area.right, 684, 14, MUTED, align="right")

    def draw_comparison(self):
        self.text("匹配训练步数、奖励设置与评估种子", 774, 463, 15, MUTED)
        if len(self.groups) < 2:
            self.text("当前记录未找到匹配的其他算法", 992, 541, 17, MUTED, align="center")
            return
        lander = self.game in LANDER_GAMES
        self.text("算法", 785, 498, 14, MUTED)
        self.text("训练种子", 1001, 498, 14, MUTED, align="right")
        self.text("回报 / 着陆" if lander else "平均分数", 1198, 498, 14, MUTED, align="right")
        for i, (algorithm, runs) in enumerate(sorted(self.groups.items())):
            y = 523 + i * 40
            self.box((775, y, 433, 34), FIELD, radius=6)
            self.text(algorithm.upper(), 785, y + 6, 17, ACCENT if algorithm == self.algorithm else TEXT,
                      bold=True)
            self.text(len(runs), 1001, y + 7, 16, align="right")
            reward = fmean(result.reward for result in runs.values())
            score = fmean(result.score for result in runs.values())
            value = f"{reward:.1f} / {score:.1%}" if lander else f"{score:.2f}"
            self.text(value, 1198, y + 7, 17, align="right")
        self.text("每个训练种子等权汇总。", 774, 703, 13, MUTED)


def launch(root, game, seed):
    pygame.display.init()
    pygame.font.init()
    window = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("强化学习实验台")
    clock = pygame.time.Clock()
    dashboard = Dashboard(root, game, seed)
    try:
        while True:
            viewport = dashboard.canvas.get_rect().fit(window.get_rect())

            def logical(position):
                return (round((position[0] - viewport.x) * SIZE[0] / viewport.width),
                        round((position[1] - viewport.y) * SIZE[1] / viewport.height))

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return
                if event.type == pygame.KEYDOWN:
                    dashboard.handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    dashboard.click(logical(event.pos))
            dashboard.tick(pygame.time.get_ticks())
            surface = dashboard.draw(logical(pygame.mouse.get_pos()))
            window.fill(BACKGROUND)
            window.blit(surface if viewport.size == SIZE else
                        pygame.transform.smoothscale(surface, viewport.size), viewport)
            pygame.display.flip()
            clock.tick(60)
    finally:
        dashboard.close()
        pygame.quit()
