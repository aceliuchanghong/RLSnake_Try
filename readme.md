## 基于强化学习训练贪吃蛇

强化学习是 agent 在与 env 互动当中,为了达成 goal 而进行学习的过程

**第一层**: agent(玩家) env(环境) goal(目标)

**第二层**: state(状态) action(行动) reward(奖励)

**第三层**: policy(策略) value(价值)

```
REINFORCE (1992) → Actor-Critic → A2C (2016) → A3C (2016) → PPO (2017) → (GRPO)
                ↘ DQN (2013/2015) [基于价值的分支]
```

---

### 基于DQN的思路

1.  **初始化：** 创建策略网络 `policy_net` 和目标网络 `target_net`（两者初始权重相同），初始化经验回放缓冲区，设置`epsilon`为较高值。
2.  **预填充：** 让Agent随机玩游戏，收集经验，填充经验回放缓冲区，直到达到最小训练批次大小。
3.  **循环训练：**
      * **动作选择：** Agent根据当前的`epsilon`执行`ε-贪婪策略`，选择一个动作。
      * **环境交互：** Agent执行选择的动作，与环境交互，获得新的状态、奖励和是否结束的信息。
      * **经验存储：** 将 (当前状态, 动作, 奖励, 下一个状态, 是否结束) 存储到经验回放缓冲区。
      * **模型更新 (如果缓冲区足够)：**
          * 从经验回放缓冲区中随机采样一个批次的经验。
          * 使用 `policy_net` 计算当前Q值
            >`policy_net` 对在当前状态 `states` 下，**实际执行的动作 `actions`** 所能获得的未来累积奖励的**估计**
          * 使用 `target_net` 计算目标Q值。
            >`target_q_values` = (当前步的即时奖励 `rewards`) + ($\gamma$ \* 由 `target_net` 估计的，从下一个状态 `next_states` 开始，采取最优动作所能获得的未来所有折扣奖励的总和)
          * 计算当前Q值和目标Q值之间的MSE损失。
          * 反向传播，更新 `policy_net` 的权重。
          * 衰减 `epsilon`。
      * **目标网络更新：** 定期将 `policy_net` 的权重复制到 `target_net`。
4.  **重复步骤3：** 直到训练达到预设的迭代次数或收敛条件。

* **Q值 $Q(s, a)$** 是在状态 $s$ 采取动作 $a$ 后，未来累积奖励的**期望值**。
* **$\max_a Q(s, a)$** 是在状态 $s$ 下，通过选择**最优动作**所能获得的未来累积奖励的**最大期望值**。
* **贝尔曼期望方程**（Bellman Expectation Equation）定义了 $Q$ 值与下一个状态的 $Q$ 值之间的关系：
$$
Q(s, a) = E[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
$$
其中：
  * $R_{t+1}$ 是在状态 $s$ 采取动作 $a$ 后获得的即时奖励。
  * $\gamma$ 是折扣因子，衡量未来奖励的重要性。
  * $Q(S_{t+1}, A_{t+1})$ 是在下一个状态 $S_{t+1}$ 采取下一个动作 $A_{t+1}$ 的 $Q$ 值。
  * $E[\dots]$ 表示期望值，因为奖励和下一个状态可能存在随机性。
* **DQN**网络就是设计的对于某个状态输出离散的动作,所以策略也是针对离散动作输出Q值

### 未开始训练时的游戏

`uv run snake/run.py`

![](z_using_files/describe_imgs/before.gif)

#### 第一次模型

似乎不太想吃食物,不断远离食物,靠近食物

```python
奖励函数的设计如下：

游戏结束：奖励为 - 5 - 10 / (1 + 0.1 * (len(self.snake) - 1))，惩罚与蛇长度反比。
吃到食物：奖励为 5 + 0.3 * (len(self.snake) - 1)，奖励与蛇长度正比。
未吃到食物：
接近食物：+0.3
远离食物：-0.3
距离不变：-0.1
远离蛇尾：奖励 +0.01 * tail_distance。
```

![](z_using_files/describe_imgs/02.png)

#### 第二次模型

很容易死+偶尔转圈圈,但是好歹跑起来了

```python
if self.game_over:
    # 简化游戏结束的惩罚，设置为固定值
    reward = -10
else:
    if self.snake[0] == self.food:
        # 显著增加吃到食物的奖励
        reward = 20
        self.prev_distance = None  # 重置距离，因为食物位置会改变
    else:
        # 调整接近或远离食物的奖励权重
        current_distance = self._calculate_distance(self.snake[0], self.food)
        if self.prev_distance is not None:
            if current_distance < self.prev_distance:
                reward += 1.0  # 增加接近食物的奖励
            elif current_distance > self.prev_distance:
                reward -= 0.8  # 增加远离食物的惩罚
            else:
                reward -= 0.2  # 距离不变时的惩罚
                self.prev_distance = current_distance
        # 防止无限循环
if self.current_steps >= self.max_steps:
    self.game_over = True
    reward = -20
```

![](z_using_files/describe_imgs/01.gif)

#### 第三次模型

随着训练次数,可以看见,平均奖励在增加,20000次一个小时

似乎还是哪儿有问题,难不成一直增加训练时间,我觉得还是收敛太慢了,需要修改`DQN`这个类

还是没搞懂,到底是哪儿影响了收敛速度

![](z_using_files/describe_imgs/reward01.png)

```
. . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . 
. . . . . . S S S S S S S . . . 
S S S S S S . . . . . . S S S . 
S S . . . S . . . . . . . . S . 
. S S S S H . . . . F . . S S . 
. . . . S S S S S S S S S S . . 
. . . . . . . . . . . . . . . . 
Steps: 531, Score: 36, Direction: DOWN, Reward: 1094.04

. . . . . . . . . . . . . . . . 
. . F . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . 
. . . . . . S S S S S S . . . . 
. . . . . . . . . . . S . . . . 
. . . . . . . . . . . S S . . . 
. . . . . . . . . . . . S . . . 
. S S . . . . . . . . . S . . . 
. S S S . . . . . . . . S . . . 
. S . S S . . . . . . . S . . . 
. S . . S . . . . . . . S . . . 
. S . . S . . . . . . . S . . . 
. S . . S . . . . . . . S . . . 
. S . . S . . . . . . . S . . . 
. S S H S S S S S S S S S . . . 
Steps: 911, Score: 44, Direction: RIGHT, Reward: 1411.78
```

#### 第四次模型

训练200000次 效果仍然一般,估计是DQN网络有问题,我觉得
```
. . . . . . . . . . . . . . . . 
. . . S S S S S S S . . . . . . 
. . . S . . H . . . . . . . . . 
. . S S . . S S . F . . . . . . 
. S S . . . . S . . . . . . . . 
. S . . . . . S . . . . . . . . 
. S . . . S S S . . . . . . . . 
. S . . S S . . . . . . . . . . 
. S . . S . . . . . . . . . . . 
. S . . S S . . . . . . . . . . 
. S . . . S . . . . . . . . . . 
. S . . . S S . . . . . . . . . 
. S S . . S S . . . . . . . . . 
. . S . . S . . . . . . . . . . 
. . S . S S . . . . . . . . . . 
. . S S S . . . . . . . . . . . 
Steps: 839, Score: 46, Direction: UP, Reward: 1510.400000000015
```

---

### 基于PPO的思路

1.  **初始化：**
      * 创建一个**Actor网络** (策略网络 $\pi_\theta$)，用于输出动作的概率分布。
      * 创建一个**Critic网络** (价值网络 $V_\phi$)，用于评估某个状态的价值。
      * 初始化一个用于临时存储本轮交互数据的缓冲区。
2.  **数据收集：**
      * 让Agent使用当前的Actor网络 $\pi_\theta$ 与环境交互固定的步数（例如2048步），或者直到一个或多个回合结束。
      * 在每一步，Actor网络接收当前状态 `s`，输出一个动作的概率分布。我们从这个分布中**采样**一个动作 `a`。
      * **存储所有交互数据：** 将 (状态 `s`, 动作 `a`, 动作的对数概率 `log_prob(a)`, 奖励 `r`, 下一个状态 `s'`, 是否结束 `done`) 完整地存储到临时缓冲区中。
3.  **循环训练：**
      * **计算优势 (Advantage) 和价值目标 (Value Targets)：**
          * 当数据收集完毕后，我们从后往前遍历收集到的所有数据。
          * 使用 **Critic网络** $V_\phi$ 估计每个状态的价值。
          * 计算 **优势函数 $\hat{A}_t$**。优势函数告诉我们，在状态 $s_t$ 采取动作 $a_t$ 相较于该状态的平均价值 $V(s_t)$ 有多好。一个常用的计算方法是 **GAE (Generalized Advantage Estimation)**。
          * 计算 **价值目标 $V_{target}$** (也叫`returns`)，作为Critic网络学习的目标。它等于当前奖励加上未来折扣后的价值。
4.  **模型更新：**
      * 将收集到的所有数据分成多个小批次 (mini-batches)。
      * 对每个小批次进行多次（例如10次）迭代更新：
          * **计算Actor损失 (策略损失):**
              * 用**当前的Actor网络**对于批次中的状态 `s`，计算采取旧动作 `a` 的**新对数概率**。
              * 计算新旧策略概率的比率：$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。
              * 根据PPO的核心**Clipped Surrogate Objective Function**计算Actor损失。这个公式通过“裁剪”概率比率来防止策略更新步子太大，从而保证训练的稳定性。
          * **计算Critic损失 (价值损失):**
              * 使用**当前的Critic网络**计算批次中状态 `s` 的价值估计 $V_\phi(s)$。
              * 计算它和之前算好的**价值目标 $V_{target}$** 之间的均方误差(MSE)损失。
          * **计算总损失：** 总损失 = Actor损失 + Critic损失 (+ 可选的熵奖励)。
          * **反向传播：** 更新Actor和Critic网络的权重。
5.  **清空数据并重复：**
      * 完成所有小批次的更新后，**清空临时缓冲区**。
      * 回到步骤2，使用**更新后的Actor网络**去收集新一轮的数据。

### install

```shell
uv init
uv venv
source .venv/bin/activate
uv pip install .

# win
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv/Scripts/activate
```

### structure
```text
RLSnake_Try/
|
├── infer.py
├── train.py
├── rl_snake/
│   ├── DQN.py
│   ├── ReplayBuffer.py
│   └── SnakeEnv.py
└── snake/
    ├── SnakeGameMain.py
    └── run.py
```

### TODO List

- [ ] 游戏通关
- [ ] 适配多种长宽的屏幕
- [ ] 提出新的的奖励算法
- [ ] 给出文章和视频教学
- [x] 看看PPO和GRPO
- [ ] 复刻五子棋与XXOO
- [ ] 复刻象棋
