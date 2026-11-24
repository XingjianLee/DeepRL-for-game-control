# 深度强化学习游戏控制

项目包含贪吃蛇、Flappy Bird 和 Lunar Lander，支持 PyTorch DQN、PPO、A2C、离散 SAC、随机基线和模型回放。

## 运行

使用现有的 `DRL` conda 环境。已验证 Python 3.11.16、PyTorch 2.14.0、
Gymnasium 1.3.0、NumPy 2.4.6 和 Pygame 2.6.1。训练在 CPU 上运行。

### 实验台

```bash
conda activate DRL
python run.py web
```

在浏览器打开 `http://127.0.0.1:8000`。上方选择游戏、算法、训练记录和关卡种子；
左侧可回放模型、暂停、单步、重开或选择「人工」试玩。模型和数据直接读取 `runs/`，
也可传入 `--runs` 指向其他目录。网页服务只监听本机地址。用 `Ctrl+C` 停止。
关卡种子可拖动滑杆或输入整数，点「设定」后生效。
完整玩完一局后，人工和模型成绩自动进入同游戏、同关卡种子的排行榜。
人工可填写玩家名，成绩保存在 `runs/leaderboard.sqlite3`。每个榜单显示前十局，点击某行的「删除」并确认即可移除该局。
需要原桌面界面时可以运行 `python run.py dashboard`。

- **训练曲线**按训练环境步数分段汇总已结束回合。街机游戏展示得分，登月展示累计回报。
- **评估结果**读取模型目录中的 `heldout.csv`、`evaluation.csv` 或 `validation.csv`，
  展示均值、最高分或最高回报、着陆率和分布。多个评估文件可以切换。
- **算法对比**按相同游戏、训练步数、奖励设置和评估文件类型及关卡种子匹配，
  每个训练种子等权汇总。选「月球着陆 · 离散」的 500,000 步模型可查看三种算法的对照。

### 命令行训练

```bash
conda activate DRL
python run.py train --game snake --algorithm dqn --steps 50000 --seed 42
python run.py train --game snake --algorithm ppo --steps 50000 --seed 42
python run.py train --game snake --algorithm a2c --steps 50000 --seed 42
```

批量训练时，游戏、算法和步数可以各列多个值。下面会依次训练 8 个组合，
每个组合都从头训练，并使用同一个训练种子。

```bash
python run.py train --game snake flap --algorithm dqn ppo --steps 50000 500000 --seed 42
```

连续版登月只支持 SAC。批次中有其他可训练组合时，程序会提示并跳过不兼容的组合。
同批传入 `--a2c-profile` 时，该设置只作用于 A2C。

结果保存到 `runs/<游戏>/<算法>/step_<步数>/seed_<种子>/`。
每个组合只保留一份结果，批次遇到已有完整结果会跳过。传入 `--overwrite` 可重训，
新模型成功保存后会完整替换旧目录，其中的旧评估文件也会一起移除。
同个组合的不同训练设置可用不同的 `--output-root` 保存。
其中 `config.json` 记录版本、训练种子和步数，`train.csv` 记录完整回合，
`model.pt` 保存策略权重。DQN 的前 1,000 步主要用于收集经验。

训练种子控制环境随机序列、网络初始化，以及各算法使用的探索、采样或经验回放。
固定种子便于复现。正式对比应为每种算法使用同一组训练种子，再用另一组固定种子独立评估。

使用两次训练结束时输出的模型目录，在相同关卡种子上评估。
训练会采样动作，评估 DQN 选取最大 Q 值的动作，评估 PPO、A2C 和 SAC 选取概率最高的动作。
命令输出的标准差描述本组测试回合间的分数差异。

```bash
DQN_RUN=runs/snake/dqn/step_50000/seed_42
PPO_RUN=runs/snake/ppo/step_50000/seed_42
python run.py evaluate --run "$DQN_RUN" "$PPO_RUN" --episodes 100 --seed 10000
python run.py evaluate --episodes 100 --seed 10000
python run.py evaluate --run "$DQN_RUN" --output dqn_evaluation.csv
```

通过桌面终端运行下列命令打开 Pygame 窗口。人工模式会等你按方向键或空格再开始。
点击窗口后，用上下左右方向键控制蛇，松开按键后沿当前方向前进。
按 R 用相同种子重开，按 Esc 退出。

```bash
python run.py play
python run.py play --algorithm dqn --seed 20000
```

`--algorithm dqn` 从默认的 `runs/` 中选择最近保存的 DQN 模型。也可以用
`--algorithm ppo`、`--algorithm a2c` 或 `--algorithm sac`，也可以用 `--run` 指定某一次训练的目录。

### Flappy Bird

在已具备 Gymnasium、NumPy、Pygame 和 Matplotlib 的 `DRL` 环境中，
使用 `python -m pip install --no-deps flappy-bird-gymnasium==0.4.0` 安装游戏环境。
项目使用 12 维数值观测，动作是保持和扇翅，每局最多 3,600 步。
游戏得分为通过的管道数，训练奖励沿用第三方环境定义。

```bash
python run.py play --game flap
python run.py train --game flap --algorithm ppo --steps 500000 --seed 42
python run.py play --game flap --algorithm ppo --seed 20000
```

人工试玩时，点击窗口后按空格开始，之后每次按空格扇翅。按 R 重开，按 Esc 退出。

### Lunar Lander

离散动作版 `LunarLander-v3` 提供 8 维数值观测和 4 个动作，训练使用官方原始奖励。
此环境需要在 `DRL` 中添加 Box2D。Python 3.11 的 Apple Silicon Mac 可使用预编译包。

```bash
python -m pip install --only-binary=:all: --no-deps Box2D==2.3.10
python run.py play --game lander
python run.py train --game lander --algorithm ppo --steps 50000 --seed 42
python run.py play --game lander --algorithm ppo
```

点击窗口后，按住 ↑ 或空格开主发动机，按住 ← 或 → 开对应的侧向发动机。
按 R 用相同种子重开，按 Esc 退出。

`evaluate --game lander` 输出累计回报及着陆、坠毁、飞出边界和超时的比例。
逐局 CSV 的 `score` 为官方环境判定着陆时的 1，其余为 0。`return` 为官方奖励的累计值。
官方判定着陆包含停在平台外的情形。正式跨算法对比使用各游戏的原环境奖励。

### 连续版 Lunar Lander

`lander-continuous` 对应官方 `LunarLanderContinuous-v3`，与离散版共用 Box2D 和评估指标。
两维连续动作分别控制主推力和侧推力。键盘可以同时开启两个方向的推力，按住时使用满推力。
SAC 自动使用连续策略和以状态、动作为输入的双 Q 网络，训练沿用官方奖励。

```bash
python run.py play --game lander-continuous
python run.py train --game lander-continuous --algorithm sac --steps 100000 --seed 42
python run.py play --game lander-continuous --algorithm sac
```

## 游戏与算法

- 棋盘为 10×10。状态包括棋盘上的食物和蛇身节序、朝向、三个候选动作的碰撞提示和食物相对位移。
- 动作依次为直行、向左转、向右转。吃到食物奖励 +1，碰撞奖励 −1，其他步奖励 0。得分为吃到的食物数。
- 撞墙、撞蛇身或填满棋盘会终止。每局 300 步的外部上限会截断。各算法都会从截断前的最后观测进行价值自举。PPO 和 A2C 的 GAE 在回合边界停止递推。
- A2C 和 PPO 共用 Actor-Critic 网络和 GAE。A2C 每收集 32 步做一次优势加权策略更新，末尾不足 32 步也会更新。PPO 每批收集至多 512 步，通过裁剪目标做多轮更新。
- `algorithm/` 管学习逻辑，`games/` 管环境和画面，`experiments/` 管训练记录与独立评估，`ui/` 管试玩和模型回放。

### A2C 改进配置

`--a2c-profile improved` 启用按游戏选定的训练设置。省略此参数或使用
`--a2c-profile baseline` 时使用原版 A2C。运行配置会记录所选设置。

- 贪吃蛇在训练奖励中加入食物距离势函数。测试和回放沿用原游戏，比较每局吃到的食物数。
- Flappy Bird 保留游戏奖励，标准化每批轨迹的优势，学习率从 `5e-4` 线性降至
  `1e-4`，熵系数从 `0.03` 线性降至 `0.005`。

```bash
python run.py train --game snake --algorithm a2c --a2c-profile improved --steps 500000 --seed 42
python run.py train --game flap --algorithm a2c --a2c-profile improved --steps 500000 --seed 42
```

### 离散 SAC

离散 SAC 使用分类策略、两个 Q 网络、经验回放和软更新的目标网络。策略与目标价值对所有离散动作求期望。
熵温度自动调节，目标熵为 `0.3 × log(动作数)`。前 1,000 步随机探索，从第 1,000 步起每隔 4 步更新。
三款游戏均使用原始环境奖励。模型文件保存回放所需的策略权重。

```bash
python run.py train --game lander --algorithm sac --steps 50000 --seed 42
python run.py play --game lander --algorithm sac
```

## 验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖 Gymnasium 接口、可复现重置、离屏渲染、碰撞、吃食物、尾部移动、
时间截断、A2C 的更新次数、SAC 的软价值，以及四种算法的短训练及存档加载。完整跑分建议使用多个训练种子。
