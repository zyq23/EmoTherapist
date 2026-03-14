import torch
from transformers import AutoModel, AutoTokenizer
from emotion.emotion_classifier import EmotionClassifier
from typing import List, Dict
import json
import os
import random

class MentalHealthChatbot:
    def __init__(self, model_path: str = "./MeChat", strategy_mapping_path: str = "./strategy_mapping.json"):
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载对话模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True
        ).half().to(self.device).eval()

        # 加载情绪分类器
        self.emotion_classifier = EmotionClassifier()

        # 对话历史记录
        self.dialogue_history = []

        # 情绪状态跟踪
        self.current_emotion = "neutral"
        self.emotion_history = []

        # 加载策略映射
        self.strategy_mapping = self.load_strategy_mapping(strategy_mapping_path)

        # 初始化情绪-策略映射
        self.emotion_strategy_map = {
            "angry": self.get_emotion_specific_strategies("angry"),
            "sad": self.get_emotion_specific_strategies("sad"),
            "fear": self.get_emotion_specific_strategies("fear"),
            "surprise": self.get_emotion_specific_strategies("surprise"),
            "neutral": self.get_emotion_specific_strategies("neutral"),
            "happy": self.get_emotion_specific_strategies("happy")
        }

    def load_strategy_mapping(self, strategy_mapping_path):
        """加载策略映射文件"""
        if not os.path.exists(strategy_mapping_path):
            raise FileNotFoundError(f"策略映射文件未找到：{strategy_mapping_path}")

        with open(strategy_mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_emotion_specific_strategies(self, emotion):
        """获取特定情绪的策略"""
        specific_strategies = []
        for strategy in self.strategy_mapping["strategies"]:
            if strategy["emotion"] == emotion:
                specific_strategies.append(strategy)
        return specific_strategies

    def get_dialogue_context(self) -> str:
        """构建对话上下文"""
        dialogue_lines = []
        for item in self.dialogue_history[-20:]:  # 保留最近20轮对话
            if item["role"] == "counselor":
                dialogue_lines.append("咨询师：" + item["content"])
            else:
                dialogue_lines.append("来访者：" + item["content"])
        return "\n".join(dialogue_lines) + "\n咨询师："

    def generate_prompt(self) -> str:
        """生成专业prompt，添加情绪条件化生成"""
        dialogue_context = self.get_dialogue_context()

        emotion_condition = f"""情绪条件化提示：
情绪类别：{self.current_emotion}
情绪强度：{self.emotion_classifier.current_emotion_intensity}
情绪变化趋势：{self.get_emotion_trend()}
适合的咨询策略：
{self.get_suitable_strategies()}"""

        return f"""现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

{emotion_condition}

[对话历史]
{dialogue_context}"""

    def get_emotion_trend(self) -> str:
        """分析情绪变化趋势"""
        if len(self.emotion_history) < 2:
            return "暂无足够数据"

        last_two = self.emotion_history[-2:]
        if last_two[0] == last_two[1]:
            return f"持续{last_two[0]}状态"
        else:
            return f"从{last_two[0]}转为{last_two[1]}"

    def get_suitable_strategies(self) -> str:
        """根据当前情绪强度获取适合的咨询策略"""
        emotion_intensity = self.emotion_classifier.current_emotion_intensity
        emotion = self.current_emotion
        suitable_strategies = []

        emotion_specific_strategies = self.emotion_strategy_map.get(emotion, [])
        for strategy in emotion_specific_strategies:
            min_intensity = strategy["intensity_range"][0]
            max_intensity = strategy["intensity_range"][1]
            if min_intensity <= emotion_intensity <= max_intensity:
                suitable_strategies.append(strategy)

        if not suitable_strategies:
            for strategy in self.strategy_mapping["strategies"]:
                if strategy["emotion"] == "universal":
                    suitable_strategies.append(strategy)

        if not suitable_strategies:
            for strategy in self.strategy_mapping["strategies"]:
                if strategy["strategy_id"] == "STRAT-BASE-001":
                    return self.format_strategy_response(strategy)

        suitable_strategies.sort(key=lambda x: x["priority_level"], reverse=True)
        top_strategy = suitable_strategies[0]
        return self.format_strategy_response(top_strategy)

    def format_strategy_response(self, strategy):
        """格式化策略响应模板"""
        if "response_template" not in strategy:
            return "目前暂时没有适合的策略建议，请继续描述你的感受。"

        template = strategy["response_template"]
        if isinstance(template, dict):
            steps = list(template.values())
            if steps:
                return random.choice(steps)
            else:
                return "请继续分享你的感受，我会尽力提供支持。"
        else:
            return template

    def update_emotion(self, text: str) -> None:
        """更新情绪状态"""
        emotion, confidence = self.emotion_classifier.predict(text)
        self.current_emotion = emotion
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 20:  # 保留最近20次记录
            self.emotion_history.pop(0)

    def generate_response(self, user_input: str) -> str:
        """生成响应"""
        # 更新情绪状态
        self.update_emotion(user_input)

        # 更新对话历史
        self.dialogue_history.append({"role": "client", "content": user_input})

        # 生成prompt
        prompt = self.generate_prompt()

        # 获取模型响应
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            temperature=0.8,
            top_p=0.8,
            repetition_penalty=1.1
        )

        # 更新对话历史
        self.dialogue_history.append({"role": "counselor", "content": response})

        return response


if __name__ == "__main__":
    chatbot = MentalHealthChatbot()
    print("心理咨询对话开始（输入0退出）...")

    while True:
        try:
            user_input = input("来访者：").strip()
            if user_input == "0":
                break

            response = chatbot.generate_response(user_input)
            print(f"咨询师：[{chatbot.current_emotion}] {response}")

        except KeyboardInterrupt:
            print("\n对话结束")
            break


