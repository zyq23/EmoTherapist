EmoTherapist: A Mental Health Dialogue System with Real-Time Emotion Recognition Integration
<p align="center">
<img src="https://img.shields.io/badge/Paper-Under_Review-blue.svg" alt="Paper Status">
<img src="https://img.shields.io/badge/Python-3.8%2B-green.svg" alt="Python Version">
<img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" alt="PyTorch">
<img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg" alt="License">
</p>

📌 项目概述 (Overview)
EmoTherapist 是一个面向心理健康干预领域的大型对话系统框架。现有的心理健康对话系统大多仅依赖语义内容生成回复，缺乏对用户隐含情感状态的显式建模。这种“情感盲区”导致模型生成的回复往往缺乏针对性与共情力（Emotional Attunement），且无法根据用户在多轮对话中细微的情感变化动态调整心理咨询策略。

为解决这一挑战，本项目提出了一种整合**实时情感识别（Real-Time Emotion Recognition）与大语言模型（LLM）**的双轨架构。通过引入显式的情感约束条件，EmoTherapist 能够动态感知用户的情绪波动，并自适应地生成具备专业深度与人文关怀的干预回复。

🚀 核心创新点 (Key Contributions)
实时情感感知 (Real-Time Emotion Recognition): 构建独立的情感识别流，从用户的多模态或文本输入中实时提取细粒度情感状态，为后续生成模块提供关键的条件特征。

情感约束生成 (Emotion-Conditioned Response Generation): 将动态提取的情感标签作为强制约束条件（Hard/Soft Constraints）输入 LLM，引导模型在生成阶段实现策略与情感的双重对齐。

🏗️ 系统架构 (Architecture)
EmoTherapist 的处理流水线（Pipeline）由三个核心模块构成：

外部情感感知模块 (Emotion Perception Module): * 基于 [填入具体模型，如 RoBERTa 或自研分类器] 实时捕捉对话中的离散/连续情感分布。

策略对齐与生成基座 (Generative LLM Backbone): * 采用 [填入具体模型，如 LLaMA-3 8B / Qwen2] 作为基座，通过指令微调（SFT）注入专业心理咨询策略（如 CBT、DBT 技巧）。

长程上下文与记忆机制 (Long-Context & Memory Mechanism): * 针对心理干预长周期的特性，引入 [填入具体技术，如 Longformer 注意力机制 / 检索增强 RAG]，确保模型在长程对话中保持背景连贯性与情感记忆。

🩺 临床干预场景 (Clinical Scenarios & Case Studies)
本项目的数据集与干预策略覆盖了 21 种深度的心理咨询场景。系统并非提供通用安慰，而是针对特定心理问题采取循证心理学支持的专业策略。

<details>
<summary><b>点击展开查看：典型干预场景与专业策略表现</b></summary>


创伤性哀伤 (Traumatic Grief)

常规模型: “你要学会放下，时间会冲淡一切。” (可能引发二次伤害)

EmoTherapist: 避免使用“放下”等切断性表述，引入**“持续联结 (Continuing Bonds)”**的专业概念。通过如“解冻手表”等时间感知隐喻，在安全范围内激活正性记忆，强调哀伤与成长可以并行。

强迫症 (Obsessive-Compulsive Disorder, OCD)

EmoTherapist: 向用户解释 OCD 的认知扭曲特点，通过“温度计”量化焦虑程度。在对话中引导区分“可能性”与“必然性”，并巧妙融入渐进式暴露干预（ERP）的理念，强调小步骤行动的价值。

慢性疼痛 (Chronic Pain)

EmoTherapist: 运用“蛮横房客”等隐喻肯定疼痛的真实性，揭示用户作为“情绪忍者”的防御模式，引导身心对话技术，重构疼痛的警示意义。

性别认同探索 (Gender Identity Exploration)

EmoTherapist: 正常化来访者的矛盾心理，识别并处理潜在的代际创伤影响。使用“蝴蝶破茧”隐喻尊重其成长与探索的非线性节奏。

文化适应 (Cultural Adaptation)

EmoTherapist: 利用视觉隐喻具象化困境，引导用户发现过往的“成功例外”。通过降低语言表达的完美主义，将适应过程重构为一场“探索游戏”。

网络成瘾干预 (Internet Addiction)

EmoTherapist: 放弃说教，转而肯定用户在游戏中的积极体验。发掘可迁移的核心能力（如团队协作、战略规划），通过职业愿景引导内部动机的转变。

</details>
