# import torch
# import torch.nn as nn
# from transformers import BertTokenizer, BertModel
# from typing import List, Dict, Tuple, Optional
# import os
# import json
# import re
#
#
# class EmotionClassifier(nn.Module):
#     def __init__(self, model_dir: str = "/data/zyq/smile/emotion_model"):
#         """
#         情绪分类器 - 修正维度错误后的完整实现
#
#         参数:
#             model_dir: 包含以下文件的目录:
#                 - pytorch_model.bin (BERT模型权重)
#                 - config.json (BERT模型配置)
#                 - classifier_head.bin (自定义分类头权重)
#                 - vocab.txt (tokenizer词汇表)
#                 - label_map.json (标签映射)
#         """
#         super(EmotionClassifier, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_dir = model_dir
#
#         # 验证模型文件是否存在
#         self._check_model_files()
#
#         # 初始化组件
#         self._init_label_map()  # 需要先初始化label_map来确定分类头维度
#         self._init_tokenizer()
#         self._init_model()
#         self._init_strong_emotion_words()
#
#         # 预测缓存
#         self.last_prediction = {}
#         self.max_length = 128
#
#     def _check_model_files(self):
#         """验证必要的模型文件是否存在"""
#         required_files = [
#             'pytorch_model.bin',
#             'config.json',
#             'classifier_head.bin',
#             'vocab.txt',
#             'label_map.json'
#         ]
#
#         missing_files = []
#         for file in required_files:
#             if not os.path.exists(os.path.join(self.model_dir, file)):
#                 missing_files.append(file)
#
#         if missing_files:
#             raise FileNotFoundError(
#                 f"模型目录缺少必要文件: {missing_files}\n"
#                 f"请确保目录 {self.model_dir} 包含完整的模型文件"
#             )
#
#     def _init_tokenizer(self):
#         """初始化tokenizer"""
#         self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
#
#     def _init_label_map(self):
#         """初始化标签映射"""
#         with open(os.path.join(self.model_dir, 'label_map.json'), 'r', encoding='utf-8') as f:
#             self.label_map = json.load(f)
#             self.reverse_label_map = {v: k for k, v in self.label_map.items()}
#         self.num_labels = len(self.label_map)
#
#     def _init_model(self):
#         """初始化模型结构"""
#         # 加载BERT模型
#         self.bert = BertModel.from_pretrained(self.model_dir)
#
#         # 构建分类头结构 - 注意输出维度设置为num_labels
#         self.drop1 = nn.Dropout(p=0.3)
#         self.linear1 = nn.Linear(self.bert.config.hidden_size, 256)
#         self.drop2 = nn.Dropout(p=0.2)
#         self.linear2 = nn.Linear(256, self.num_labels)  # 关键修正：使用num_labels而不是vocab_size
#         self.relu = nn.ReLU()
#
#         # 加载分类头权重
#         classifier_state = torch.load(
#             os.path.join(self.model_dir, 'classifier_head.bin'),
#             map_location=self.device
#         )
#
#         self.drop1.load_state_dict(classifier_state['drop1.state_dict'])
#         self.linear1.load_state_dict(classifier_state['linear1.state_dict'])
#         self.drop2.load_state_dict(classifier_state['drop2.state_dict'])
#         self.linear2.load_state_dict(classifier_state['linear2.state_dict'])
#
#         self.to(self.device)
#         self.eval()
#
#     def _init_strong_emotion_words(self):
#         """初始化强情绪词库"""
#         self.strong_emotion_words = {
#             "angry": ["气死", "烦死了", "混蛋", "垃圾", "恶心", "有病", "去死", "妈的"],
#             "happy": ["开心极了", "太高兴了", "好幸福", "笑死", "棒极了", "完美"],
#             "sad": ["想自杀", "心碎了", "绝望", "崩溃", "泪流满面", "想哭"],
#             "fear": ["吓死了", "好可怕", "恐怖", "吓尿", "胆战心惊", "毛骨悚然"],
#             "surprise": ["惊呆了", "震惊", "难以置信", "天啊", "我的天"]
#         }
#
#     def preprocess_text(self, text: str) -> str:
#         """文本预处理"""
#         if not isinstance(text, str):
#             return ""
#
#         # 去除特殊字符但保留情感符号
#         text = re.sub(
#             r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f\U00031350-\U000323af😀-🙏]',
#             '', text)
#         # 合并重复标点
#         text = re.sub(r'([!?。，])\1+', r'\1', text)
#         return text.strip()
#
#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
#         """
#         前向传播
#
#         参数:
#             input_ids: 输入token IDs
#             attention_mask: 注意力mask
#             **kwargs: 接收其他可能参数（如token_type_ids）
#
#         返回:
#             情绪分类logits
#         """
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=kwargs.get('token_type_ids', None)  # 处理可能的token_type_ids
#         )
#         pooled_output = outputs.last_hidden_state[:, 0, :]
#         output = self.drop1(pooled_output)
#         output = self.linear1(output)
#         output = self.relu(output)
#         output = self.drop2(output)
#         return self.linear2(output)
#
#     def _detect_strong_emotion(self, text: str) -> Optional[str]:
#         """强情绪词检测"""
#         if not text:
#             return None
#
#         text_lower = text.lower()
#         for emotion, words in self.strong_emotion_words.items():
#             for word in words:
#                 if word in text_lower:
#                     return emotion
#         return None
#
#     def predict(self, text: str, context: Optional[List[str]] = None) -> Tuple[str, float]:
#         """
#         情绪预测
#
#         参数:
#             text: 待分析文本
#             context: 上下文对话列表(可选)
#
#         返回:
#             tuple: (情绪标签, 置信度)
#         """
#         # 预处理文本
#         text = self.preprocess_text(text)
#         if not text:
#             return "neutral", 0.0
#
#         # 缓存检查
#         cache_key = hash(text + (context[-1] if context else ""))
#         if cache_key in self.last_prediction:
#             return self.last_prediction[cache_key]
#
#         # 强情绪词覆盖检测
#         strong_emotion = self._detect_strong_emotion(text)
#         if strong_emotion:
#             self.last_prediction[cache_key] = (strong_emotion, 0.95)
#             return strong_emotion, 0.95
#
#         # 模型预测
#         inputs = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#             return_token_type_ids=False  # 关键修改：不返回token_type_ids
#         ).to(self.device)
#
#         with torch.no_grad():
#             outputs = self(**inputs)
#             probs = torch.softmax(outputs, dim=1)[0]
#             pred_idx = torch.argmax(probs).item()
#             emotion = self.reverse_label_map[pred_idx]
#             confidence = probs[pred_idx].item()
#
#         # 缓存结果
#         self.last_prediction[cache_key] = (emotion, confidence)
#         return emotion, confidence
#
#     def predict_with_details(self, text: str, context: Optional[List[str]] = None) -> Dict:
#         """
#         带详细结果的预测
#
#         返回:
#             dict: {
#                 'text': 原始文本,
#                 'label': 情绪标签,
#                 'score': 置信度,
#                 'probabilities': 各类别概率分布,
#                 'is_strong': 是否强情绪词触发
#             }
#         """
#         text = self.preprocess_text(text)
#         strong_emotion = self._detect_strong_emotion(text)
#
#         if strong_emotion:
#             probs = {label: 0.01 for label in self.label_map}
#             probs[strong_emotion] = 0.95
#             return {
#                 'text': text,
#                 'label': strong_emotion,
#                 'score': 0.95,
#                 'probabilities': probs,
#                 'is_strong': True
#             }
#
#         inputs = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#             return_token_type_ids=False  # 关键修改：不返回token_type_ids
#         ).to(self.device)
#
#         with torch.no_grad():
#             outputs = self(**inputs)
#             probs = torch.softmax(outputs, dim=1)[0]
#             pred_idx = torch.argmax(probs).item()
#
#             return {
#                 'text': text,
#                 'label': self.reverse_label_map[pred_idx],
#                 'score': probs[pred_idx].item(),
#                 'probabilities': {
#                     label: probs[i].item()
#                     for label, i in self.label_map.items()
#                 },
#                 'is_strong': False
#             }
#
#     def batch_predict(self, texts: List[str]) -> List[Dict]:
#         """批量预测"""
#         return [self.predict_with_details(text) for text in texts]
#
#     def get_available_emotions(self) -> List[str]:
#         """获取支持的情绪标签"""
#         return list(self.label_map.keys())
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 初始化分类器
#     classifier = EmotionClassifier(model_dir="/data/zyq/smile/emotion_model")
#
#     # 测试文本
#     test_texts = [
#         "老板今天又骂我了，真的很生气！",
#         "😊 今天拿到了offer好开心，太幸福了！",
#         "听到这个消息我很难过，想哭...",
#         "明天要考试了，感觉非常焦虑，吓死我了",
#         "你猜怎么着？我中奖了！完全没想到！",
#         "今天天气不错",
#         "这个产品太垃圾了，气死我了！",
#         "我...我不知道该怎么办...感觉很绝望"
#     ]
#
#     print("=== 情绪分类测试 ===")
#     print(f"支持的情绪类别: {classifier.get_available_emotions()}\n")
#
#     # 单条文本预测
#     print("\n=== 单条文本预测 ===")
#     sample_text = test_texts[0]
#     result = classifier.predict_with_details(sample_text)
#     print(f"文本: {result['text']}")
#     print(f"预测情绪: {result['label']} (置信度: {result['score']:.2%})")
#     print("概率分布:")
#     for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
#         print(f"  {emotion}: {prob:.2%}")
#
#     # 批量预测
#     print("\n=== 批量预测 ===")
#     batch_results = classifier.batch_predict(test_texts)
#     for res in batch_results:
#         text_display = res['text'][:20] + "..." if len(res['text']) > 20 else res['text']
#         print(f"{text_display.ljust(25)} → {res['label'].ljust(8)} ({res['score']:.0%})" +
#               (" [强情绪]" if res['is_strong'] else ""))


#情绪-策略映射体系
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Tuple, Optional
import os
import json
import re

class EmotionClassifier(nn.Module):
    def __init__(self, model_dir: str = "/data/zyq/smile/emotion_model"):
        """
        情绪分类器 - 修正维度错误后的完整实现

        参数:
            model_dir: 包含以下文件的目录:
                - pytorch_model.bin (BERT模型权重)
                - config.json (BERT模型配置)
                - classifier_head.bin (自定义分类头权重)
                - vocab.txt (tokenizer词汇表)
                - label_map.json (标签映射)
        """
        super(EmotionClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        # 验证模型文件是否存在
        self._check_model_files()

        # 初始化组件
        self._init_label_map()  # 需要先初始化label_map来确定分类头维度
        self._init_tokenizer()
        self._init_model()
        self._init_strong_emotion_words()

        # 预测缓存
        self.last_prediction = {}
        self.max_length = 128

        # 情绪强度估计
        self.current_emotion_intensity = 0.0

    def _check_model_files(self):
        """验证必要的模型文件是否存在"""
        required_files = [
            'pytorch_model.bin',
            'config.json',
            'classifier_head.bin',
            'vocab.txt',
            'label_map.json'
        ]

        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(self.model_dir, file)):
                missing_files.append(file)

        if missing_files:
            raise FileNotFoundError(
                f"模型目录缺少必要文件: {missing_files}\n"
                f"请确保目录 {self.model_dir} 包含完整的模型文件"
            )

    def _init_tokenizer(self):
        """初始化tokenizer"""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)

    def _init_label_map(self):
        """初始化标签映射"""
        with open(os.path.join(self.model_dir, 'label_map.json'), 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.num_labels = len(self.label_map)

    def _init_model(self):
        """初始化模型结构"""
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(self.model_dir)

        # 构建分类头结构 - 注意输出维度设置为num_labels
        self.drop1 = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.drop2 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(256, self.num_labels)  # 关键修正：使用num_labels而不是vocab_size
        self.relu = nn.ReLU()

        # 加载分类头权重
        classifier_state = torch.load(
            os.path.join(self.model_dir, 'classifier_head.bin'),
            map_location=self.device
        )

        self.drop1.load_state_dict(classifier_state['drop1.state_dict'])
        self.linear1.load_state_dict(classifier_state['linear1.state_dict'])
        self.drop2.load_state_dict(classifier_state['drop2.state_dict'])
        self.linear2.load_state_dict(classifier_state['linear2.state_dict'])

        self.to(self.device)
        self.eval()

    def _init_strong_emotion_words(self):
        """初始化强情绪词库"""
        self.strong_emotion_words = {
            "angry": ["气死", "烦死了", "混蛋", "垃圾", "恶心", "有病", "去死", "妈的"],
            "happy": ["开心极了", "太高兴了", "好幸福", "笑死", "棒极了", "完美"],
            "sad": ["想自杀", "心碎了", "绝望", "崩溃", "泪流满面", "想哭"],
            "fear": ["吓死了", "好可怕", "恐怖", "吓尿", "胆战心惊", "毛骨悚然"],
            "surprise": ["惊呆了", "震惊", "难以置信", "天啊", "我的天"]
        }

    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not isinstance(text, str):
            return ""

        # 去除特殊字符但保留情感符号
        text = re.sub(
            r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f\U00031350-\U000323af😀-🙏]',
            '', text)
        # 合并重复标点
        text = re.sub(r'([!?。，])\1+', r'\1', text)
        return text.strip()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力mask
            **kwargs: 接收其他可能参数（如token_type_ids）

        返回:
            情绪分类logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=kwargs.get('token_type_ids', None)  # 处理可能的token_type_ids
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop1(pooled_output)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.drop2(output)
        return self.linear2(output)

    def _detect_strong_emotion(self, text: str) -> Optional[str]:
        """强情绪词检测"""
        if not text:
            return None

        text_lower = text.lower()
        for emotion, words in self.strong_emotion_words.items():
            for word in words:
                if word in text_lower:
                    return emotion
        return None

    def predict(self, text: str, context: Optional[List[str]] = None) -> Tuple[str, float]:
        """
        情绪预测

        参数:
            text: 待分析文本
            context: 上下文对话列表(可选)

        返回:
            tuple: (情绪标签, 置信度)
        """
        # 预处理文本
        text = self.preprocess_text(text)
        if not text:
            return "neutral", 0.0

        # 缓存检查
        cache_key = hash(text + (context[-1] if context else ""))
        if cache_key in self.last_prediction:
            return self.last_prediction[cache_key]

        # 强情绪词覆盖检测
        strong_emotion = self._detect_strong_emotion(text)
        if strong_emotion:
            self.last_prediction[cache_key] = (strong_emotion, 0.95)
            self.current_emotion_intensity = 0.95
            return strong_emotion, 0.95

        # 模型预测
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False  # 关键修改：不返回token_type_ids
        ).to(self.device)

        with torch.no_grad():
            outputs = self(**inputs)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            emotion = self.reverse_label_map[pred_idx]
            confidence = probs[pred_idx].item()

        # 缓存结果
        self.last_prediction[cache_key] = (emotion, confidence)

        # 记录情绪强度
        self.current_emotion_intensity = round(confidence, 2)

        return emotion, confidence

    def predict_with_details(self, text: str, context: Optional[List[str]] = None) -> Dict:
        """
        带详细结果的预测

        返回:
            dict: {
                'text': 原始文本,
                'label': 情绪标签,
                'score': 置信度,
                'probabilities': 各类别概率分布,
                'is_strong': 是否强情绪词触发
            }
        """
        text = self.preprocess_text(text)
        strong_emotion = self._detect_strong_emotion(text)

        if strong_emotion:
            probs = {label: 0.01 for label in self.label_map}
            probs[strong_emotion] = 0.95
            return {
                'text': text,
                'label': strong_emotion,
                'score': 0.95,
                'probabilities': probs,
                'is_strong': True
            }

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False  # 关键修改：不返回token_type_ids
        ).to(self.device)

        with torch.no_grad():
            outputs = self(**inputs)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

            # 记录情绪强度
            self.current_emotion_intensity = round(probs[pred_idx].item(), 2)

            return {
                'text': text,
                'label': self.reverse_label_map[pred_idx],
                'score': probs[pred_idx].item(),
                'probabilities': {
                    label: probs[i].item()
                    for label, i in self.label_map.items()
                },
                'is_strong': False
            }

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """批量预测"""
        return [self.predict_with_details(text) for text in texts]

    def get_available_emotions(self) -> List[str]:
        """获取支持的情绪标签"""
        return list(self.label_map.keys())
