# Reinforcement Learning in Autonomous Driving

the State

the Action

the Policy

> using the policy, given the state, agent will take some actions. and this is the reinforcement learning!

Gen-drive

传统方法：传统的预测性和确定性规划通常将预测和规划过程分开，这会将egovehicle和environment分离开，这种割裂会导致不符合社会驾驶规范的行为。

现有approach的局限性：综合预测规划框架：仍然无法摆脱确定性规划。

进一步瓶颈：行为不确定性，多模态，多目标的相互交互

![alt text](image-1.png)

论文创新：to overcome these challenges, we propose the adoption of generation-evaluation methods for the planning task.

关键点：将ego vehicle（agent）整合到一个social interaction context，生成一系列可能得结果，使用一个case evaluator来指导评估决策。

![alt text](image-2.png)

> 这个case evaluator怎么搞

另一个创新点：关于扩散模型：扩散模型当前在决策中的应用仍然有限。主要是两点：

1. 评估哪个case更符合人类价值观和期望是一件很复杂的事情
2. 生成模型用来做规划的数据少，样本少，不像仿真场景。

文章解决这个问题是这样的：引入了一个用来case evaluate的一个评估模型，这是一个vlm base的模型。用于反馈偏好数据进行训练。
另外就是样本少的问题，文章用RL finetuning framework，根据获得的奖励模型来提高扩散生成的质量。用来微调。

> 重点要搞懂两个东西：一个是vlm奖励模型，一个是如何finetune。
> 另外，finetuning的作用是为了更好利用数据？还是为了做数据生成？这是需要明确的。

## gendrive的三大模块

LOOP:
1. 行为扩散模型 - 场景生成器（generate case）
2. 场景评估模型 - VLM based reward model（看case）
3. RL finetuning framework - DDPO算法来优化场景生成器

简述：扩散生成式驾驶策略和训练框架。

一个向量空间中中心查询场景上下文编码器
一个基于扩散场景生成器
一个场景评估其来评估生成的场景质量进行规划

训练过程：
1. 使用大量真实世界的驾驶数据来训练基本的扩散模型（场景编码器+生成器）
2. 生成一个成对的场景偏好数据集，这些场景（case）由扩散模型生成，由VLM混合标注管道来进行打标。
3. 在这个场景偏好数据集的基础上，训练一个场景打分专家。
4. 使用RLAIF来finetune扩散模型，优化生成的轨迹。