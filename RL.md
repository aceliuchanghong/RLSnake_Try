### PPO

**Proximal Policy Optimization**（近端策略优化）

PPO的核心思想是在更新策略时，既要最大化期望回报，又不能让新策略偏离旧策略太远。


| 特征 | DQN (Deep Q-Network) | PPO (Proximal Policy Optimization) |
| :--- | :--- | :--- |
| **算法类别** | **基于价值 (Value-Based)** | **基于策略 (Policy-Based)** |
| **学习目标** | 学习一个**Q函数** $Q(s, a)$，来估计在状态`s`下采取动作`a`的好坏。策略是隐式的（通常是ε-贪婪） | 直接学习一个**策略** $\pi(a\|s)$ 该策略会直接输出在状态`s`下应该采取每个动作的概率。 |
| **数据使用** | **离策略 (Off-Policy)** 使用一个大的经验回放缓冲区 (`Replay Buffer`)，从中随机采样，打破数据相关性，提高样本效率。 | **在策略 (On-Policy)** 每次更新只使用最新一轮与环境交互收集到的数据，然后丢弃。它不能使用旧数据。 |
| **网络结构** | 通常只有一个网络（或两个：`policy_net`, `target_net`）来预测所有动作的Q值。 | 通常有两个独立或共享部分网络：**Actor**（策略网络）和 **Critic**（价值网络）。 |
| **动作空间** | 主要用于**离散**动作空间。 | 可以轻松处理**离散**和**连续**动作空间。 |
| **核心优势** | 样本效率高（因为重复利用旧数据）。 | 训练过程更稳定、更可靠，对超参数不那么敏感。 |

PPO 的核心是使用**剪切概率比**（Clipped Surrogate Objective）来更新策略。它的目标函数可以表示为：
$$
L(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$
其中：
- $\theta$：策略网络的参数。
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$：新旧策略的概率比。
- $\hat{A}_t$：优势函数（Advantage Function），衡量动作的相对好坏。
- $\epsilon$：剪切参数（通常为 0.1 或 0.2），限制策略更新的幅度。
- $\text{clip}(x, a, b)$：将 $x$ 限制在 $[a, b]$ 范围内。

---

#### 相关定义

##### **回报 (Return)**
   
**回报 (Return, $G_t$)**: 从时间步 $t$ 开始，未来所有奖励的折扣总和。它代表了从当前时刻开始，在一个具体的轨迹（episode）中，我们**实际**能获得的总收益。
    $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
    其中 $R_{t+1}$ 是在 $t+1$ 时刻获得的即时奖励，$\gamma$ 是折扣因子。

##### **V值 (State-Value Function, $V(s)$)**

**直观理解：** 状态 $s$ 有多好？
**定义：** V值，即状态价值函数 $V_\pi(s)$，衡量的是当Agent从状态 $s$ 出发，并**遵循策略 $\pi$** 进行决策时，它所能获得的**期望回报**。
$$V_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$


这个公式回答的问题是：“如果我现在处于状态 $s$，并且我接下来的所有行为都遵循我的策略 $\pi$，那么平均下来我能得到多少总回报？”
在PPO算法中，**Critic网络** 的职责就是学习并估计这个 $V(s)$。给定一个状态，Critic会输出一个数值，这个数值就是对该状态价值的估计。

##### **Q值 (Action-Value Function, $Q(s, a)$)**

**直观理解：** 在状态 $s$ 下，执行动作 $a$ 有多好?
**定义：** Q值，即动作价值函数 $Q_\pi(s, a)$，衡量的是当Agent在状态 $s$ **选择了特定的动作 $a$**，然后**在后续所有步骤中遵循策略 $\pi$** 时，它所能获得的**期望回报**。
$$Q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

这个公式回答的问题是：“如果我现在处于状态 $s$，并且我**强制**执行动作 $a$，之后再让我的策略 $\pi$ 接管，那么平均下来我能得到多少总回报？”

##### **时序差分(Temporal-Difference, TD)**

利用当前对未来的估计，来更新更早的估计

* **时序（Temporal）：** 指的是学习过程是按时间顺序（$t, t+1, t+2, \dots$）发生的。
* **差分（Difference）：** 指的是学习的信号来自于两个估计值之间的**差异**。

TD误差是在时间步 $t$ 产生的一个信号，它衡量了我们**当前的价值估计**与**基于下一步实际情况的、更优的价值估计**之间的差距。

1.  **我们已有的估计：** 在 $t$ 时刻，我们处于状态 $S_t$。我们的`Critic`网络给出了对这个状态的价值估计：$V(S_t)$。这是我们的“旧知识”。

2.  **我们新的、更可靠的估计 (TD目标)：** 接着，我们执行动作 $A_t$，环境给了我们两个新的信息：
    * 一个即时奖励 $R_{t+1}$。
    * 一个新的状态 $S_{t+1}$。

    利用这些新信息，我们可以构建一个对 $V(S_t)$ 的**新目标值**，称为 **TD目标（TD Target）**：
    $$
    \text{TD Target} = R_{t+1} + \gamma V(S_{t+1})
    $$
    * 这个目标值的含义是：状态 $S_t$ 的价值，应该约等于我们马上得到的**真实奖励**，再加上我们对**下一个状态价值的估计**（折扣后）
    * 相比于单纯的 $V(S_t)$，TD目标包含了**一步的真实世界信息**（$R_{t+1}$），因此它通常是对 $V(S_t)$ 更准确、更可靠的一个估计。

3.  **计算误差：** TD误差就是“新估计”与“旧估计”之间的差值。
    $$
    \delta_t = \underbrace{(R_{t+1} + \gamma V(S_{t+1}))}_{\text{TD Target (新估计)}} - \underbrace{V(S_t)}_{\text{旧估计}}
    $$

---

#### 1. 优势函数 (Advantage Function)

优势函数衡量在状态 $s$ 采取动作 $a$ 比平均水平好多少。
$$A(s, a) = Q(s, a) - V(s)$$在实践中，我们无法得到真实的Q值和V值，所以我们用估计值。一个简单的方法是：$$\hat{A}_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
这里 $R_{t+1} + \gamma V(S_{t+1})$ 是对 $Q(S_t, A_t)$ 的估计。

更常用的是 **GAE (Generalized Advantage Estimation)**，它通过引入参数 $\lambda$ 来权衡偏差和方差，效果更好：
$$\hat{A}_t^{GAE}(\gamma, \lambda) = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$
其中 $\delta_{t+l} = r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})$ 是时序差分误差 (TD Error)。

#### 2. Actor 损失 (Clipped Surrogate Objective)

这是PPO的灵魂。我们定义策略概率比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，其中 $\pi_{\theta_{old}}$ 是收集数据时的旧策略。

Actor的目标函数是：
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

  * $\hat{\mathbb{E}}_t$ 表示在所有时间步上取平均。
  * $\hat{A}_t$ 是我们计算好的优势函数。
  * **第一部分 $r_t(\theta) \hat{A}_t$：** 这是标准的策略梯度目标。如果优势 $\hat{A}_t$ 是正的，我们想增大 $r_t(\theta)$（即增大采取该动作的概率）；如果优势是负的，我们想减小它。
  * **第二部分 $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$：** 这是PPO的创新。`clip` 函数将比率 $r_t(\theta)$ 强制限制在 $[1-\epsilon, 1+\epsilon]$ 的范围内（通常 $\epsilon=0.2$）。
      * 这个限制意味着，即使某个动作的优势特别大，我们也不会让策略的更新幅度过大（被 $1+\epsilon$ 卡住）。
      * 同理，即使优势特别小，我们也不会让策略的惩罚过猛（被 $1-\epsilon$ 卡住）。
  * **$\min(\dots)$：** 我们取这两项中较小的一个作为最终的目标。这形成了一个“悲观”的下界，确保了更新的保守性和稳定性。

#### 3. Critic 损失

这个很简单，就是价值网络预测值 $V_\phi(s_t)$ 和我们计算出的价值目标（也叫`Returns`）$V_{target}$ 之间的均方误差。
$$L^{VF}(\phi) = \left( V_\phi(s_t) - V_{target} \right)^2$$
其中 $V_{target}$ 通常是 $R_{t} + \gamma V_{target}(s_{t+1})$。


- net.py

```python
import torch
import torch.nn as nn

# Actor和Critic可以共用一个网络主体，或者完全分开
# 这里使用一个共享主干的网络
class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        # 共享的卷积和全连接层
        self.shared_base = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * input_shape[0] * input_shape[1], 256),
            nn.ReLU()
        )
        
        # Actor head: 输出动作的logits
        self.actor_head = nn.Linear(256, num_actions)
        
        # Critic head: 输出状态的价值
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        base_output = self.shared_base(x)
        action_logits = self.actor_head(base_output)
        state_value = self.critic_head(base_output)
        return action_logits, state_value
```

- PPOAgent.py

```python
import torch
import torch.nn as nn
# 用于从策略分布中采样
from torch.distributions import Categorical

class PPOAgent:
    def __init__(
        self,
        state_shape,
        num_actions,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95, # GAE参数
        clip_epsilon=0.2, # PPO裁剪参数
        epochs=10,        # 每次数据更新的迭代次数
        batch_size=64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        self.network = ActorCritic(state_shape, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO是on-policy，需要一个临时存储
        self.memory = [] # 一个简单的列表来存储轨迹

    def store_transition(self, state, action, log_prob, reward, done, value):
        # 存储一次交互的完整信息
        self.memory.append((state, action, log_prob, reward, done, value))
        
    def clear_memory(self):
        self.memory = []

    def select_action(self, state):
        # 与DQN不同，这里总是从策略网络中采样
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, state_value = self.network(state)
        
        # 从logits创建概率分布
        action_dist = Categorical(logits=action_logits)
        # 从分布中采样一个动作
        action = action_dist.sample()
        # 计算该动作的对数概率，用于后续更新
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()

    def update(self):
        # 1. 从内存中解包所有数据
        states, actions, old_log_probs, rewards, dones, values = zip(*self.memory)
        
        # 2. 计算优势(Advantage)和回报(Returns)
        advantages = torch.zeros(len(rewards), dtype=torch.float32).to(self.device)
        last_advantage = 0
        
        # 从后往前计算GAE
        for t in reversed(range(len(rewards) - 1)):
            if dones[t]:
                td_error = rewards[t] - values[t]
                last_advantage = td_error
            else:
                td_error = rewards[t] + self.gamma * values[t+1] - values[t]
                last_advantage = td_error + self.gamma * self.gae_lambda * last_advantage
            advantages[t] = last_advantage
            
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32).to(self.device) # 计算价值目标
        
        # 数据转换
        states = torch.FloatTensor(np.array(states[:-1])).unsqueeze(1).to(self.device)
        actions = torch.tensor(actions[:-1], dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(old_log_probs[:-1], dtype=torch.float32).to(self.device)

        # 3. 进行多轮(epochs)优化
        for _ in range(self.epochs):
            # 将数据分批
            for index in range(0, len(states), self.batch_size):
                # 获取当前批次的数据
                batch_states = states[index : index + self.batch_size]
                batch_actions = actions[index : index + self.batch_size]
                batch_old_log_probs = old_log_probs[index : index + self.batch_size]
                batch_advantages = advantages[index : index + self.batch_size]
                batch_returns = returns[index : index + self.batch_size]

                # 4. 计算当前策略的输出
                new_logits, new_values = self.network(batch_states)
                new_values = new_values.squeeze()
                
                new_dist = Categorical(logits=new_logits)
                new_log_probs = new_dist.log_prob(batch_actions)
                
                # 5. 计算PPO损失
                # Actor Loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic Loss
                critic_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy Bonus (可选，鼓励探索)
                entropy = new_dist.entropy().mean()
                
                # 总损失
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                # 6. 梯度更新
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
                
        # 7. 清空内存，为下一轮数据收集做准备
        self.clear_memory()
```

---

#### GAE的直观解释：一个聪明的加权平均

GAE的核心公式是：
$$\hat{A}_t^{GAE}(\gamma, \lambda) = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

让我们把这个求和展开来看：
$$\hat{A}_t^{GAE} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + (\gamma\lambda)^3\delta_{t+3} + \dots$$
其中 $\delta_{t+l} = R_{t+l+1} + \gamma V(S_{t+l+1}) - V(S_{t+l})$ 是在未来第 $l$ 步的TD误差。

这个公式实际上是在说：
在 $t$ 时刻的优势，是**当前TD误差 $\delta_t$**，加上**折扣后的未来第一步的TD误差 $(\gamma\lambda)\delta_{t+1}$**，再加上**更深度折扣后的未来第二步的TD误差 $(\gamma\lambda)^2\delta_{t+2}$**，以此类推。

它是一个对未来所有TD误差的**指数加权移动平均**。

##### 参数 $\lambda$ 的魔力：偏差-方差的“调谐旋钮”

参数 $\lambda$（取值范围[0, 1]）控制了这个加权平均的权重衰减速度，从而让我们能够在偏差和方差之间自由调节。

* **当 $\lambda=0$ 时：**
    公式变为 $\hat{A}_t^{GAE} = \delta_t + 0 + 0 + \dots = \delta_t$。
    这**完全等同于单步TD估计**。我们得到了一个**高偏差、低方差**的估计。我们只相信一步之内的信息。

* **当 $\lambda=1$ 时：**
    公式变为 $\hat{A}_t^{GAE} = \sum_{l=0}^{\infty}(\gamma)^l \delta_{t+l}$。
    经过数学推导可以证明，这种情况**完全等同于蒙特卡洛估计** $G_t - V(S_t)$。
    我们得到了一个**低偏差、高方差**的估计。我们相信直到回合结束的整个轨迹。

* **当 $0 < \lambda < 1$ 时（例如，常用的 $\lambda=0.95$）：**
    这就是GAE的精髓所在。我们得到了一个**折衷方案**。
    * 我们给最近的TD误差 $\delta_t$ 最高的权重。
    * 我们也会考虑未来的TD误差 $\delta_{t+1}, \delta_{t+2}, \dots$，但它们的权重会随着 $(\gamma\lambda)$ 的幂次快速衰减。
    * 这相当于说：“我主要相信我的一步估计，但为了减少它的偏差，我也会参考一下未来几步的情况，不过未来的信息越远，我就越不信任它。”

**总结：** GAE通过引入参数 $\lambda$ ，巧妙地将高偏差低方差的TD估计和低偏差高方差的蒙特卡洛估计融合在了一起。它允许我们通过调整 $\lambda$ 来控制我们愿意接受多少偏差和多少方差，从而在实践中获得更稳定、更高效的训练效果。这就是为什么它比简单的单步TD估计“效果更好”。
