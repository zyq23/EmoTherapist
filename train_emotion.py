import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import random
import os

# 设置随机种子保证可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义情绪标签顺序（固定顺序确保一致性）
EMOTION_LABELS = ["neutral", "angry", "happy", "sad", "fear", "surprise"]
LABEL_MAP = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# 创建模型保存目录
os.makedirs("emotion_model", exist_ok=True)


# 数据加载类
class EmotionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 加载数据文件
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)[0]  # 假设数据在列表的第一个元素中

        # 验证所有标签都在预定义的标签集合中
        for item in self.data:
            if item['label'] not in LABEL_MAP:
                raise ValueError(f"发现未知标签: {item['label']}。请确保所有标签都在预定义集合中: {EMOTION_LABELS}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item]['content'])
        label = LABEL_MAP[self.data[item]['label']]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 增强的情绪分类模型
class EnhancedEmotionClassifier(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-chinese'):
        super(EnhancedEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop1 = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.drop2 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop1(pooled_output)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.drop2(output)
        return self.linear2(output)


# 保存完整模型
def save_full_model(model, tokenizer, output_dir, label_map):
    """保存完整模型文件到指定目录"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型权重
    model.bert.save_pretrained(output_dir)

    # 保存分类头
    classifier_state = {
        'drop1.state_dict': model.drop1.state_dict(),
        'linear1.state_dict': model.linear1.state_dict(),
        'drop2.state_dict': model.drop2.state_dict(),
        'linear2.state_dict': model.linear2.state_dict()
    }
    torch.save(classifier_state, os.path.join(output_dir, 'classifier_head.bin'))

    # 保存tokenizer
    tokenizer.save_pretrained(output_dir)

    # 保存标签映射
    with open(os.path.join(output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"✅ 完整模型已保存到 {output_dir}")


# 带权重平衡的训练函数
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples, class_weights=None):
    model = model.train()
    losses = []
    correct_predictions = 0

    if class_weights is not None:
        class_weights = class_weights.to(device)

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)

        if class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, labels)
        else:
            loss = criterion(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# 增强的评估函数
def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    return (correct_predictions.double() / n_examples,
            np.mean(losses),
            all_preds,
            all_labels,
            all_probabilities)


# 计算类别权重
def calculate_class_weights(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)[0]

    label_counts = {label: 0 for label in EMOTION_LABELS}
    for item in data:
        label_counts[item['label']] += 1

    weights = [1.0 / label_counts[label] for label in EMOTION_LABELS]
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() * len(EMOTION_LABELS)  # 归一化

    print("\n类别分布:")
    for label in EMOTION_LABELS:
        print(f"{label}: {label_counts[label]} samples")

    print("\n类别权重:", {label: f"{weights[i]:.2f}" for i, label in enumerate(EMOTION_LABELS)})

    return weights


# 主函数
def main():
    # 检查数据文件是否存在
    train_path = "./train_emotion_data/usual_train.json"
    eval_path = "./train_emotion_data/usual_eval.json"
    test_path = "./train_emotion_data/usual_test.json"

    if not all(os.path.exists(path) for path in [train_path, eval_path, test_path]):
        raise FileNotFoundError(
            "请确保训练集(usual_train.json)、验证集(usual_eval.json)和测试集(usual_test.json)文件存在")

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    MAX_LEN = 128
    BATCH_SIZE = 16

    # 创建数据集
    train_dataset = EmotionDataset(train_path, tokenizer, MAX_LEN)
    eval_dataset = EmotionDataset(eval_path, tokenizer, MAX_LEN)
    test_dataset = EmotionDataset(test_path, tokenizer, MAX_LEN)

    # 计算类别权重（处理不平衡数据）
    class_weights = calculate_class_weights(train_path)

    # 创建数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    model = EnhancedEmotionClassifier(n_classes=len(EMOTION_LABELS))
    model = model.to(device)

    # 训练参数
    EPOCHS = 15  # 增加epoch数量
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10%的warmup步数
        num_training_steps=total_steps
    )
    global criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # 训练循环
    history = defaultdict(list)
    best_f1 = 0
    best_model = None

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 30)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler,
            len(train_dataset),
            class_weights=class_weights)

        print(f'Train loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')

        eval_acc, eval_loss, eval_preds, eval_labels, _ = eval_model(
            model,
            eval_data_loader,
            device,
            len(eval_dataset))

        # 计算验证集F1分数
        eval_f1 = f1_score(eval_labels, eval_preds, average='weighted')
        print(f'Validation loss: {eval_loss:.4f} | Accuracy: {eval_acc:.4f} | F1: {eval_f1:.4f}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['eval_acc'].append(eval_acc)
        history['eval_loss'].append(eval_loss)
        history['eval_f1'].append(eval_f1)

        # 保存最佳模型（基于F1分数）
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_model = model.state_dict()
            print(f"New best model found with F1: {eval_f1:.4f}")

    # 保存完整的最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
        save_full_model(model, tokenizer, "emotion_model", LABEL_MAP)

    # 在测试集上评估最佳模型
    print("\nEvaluating on test set...")
    model.load_state_dict(best_model)
    model = model.to(device)

    test_acc, test_loss, test_preds, test_labels, test_probs = eval_model(
        model,
        test_data_loader,
        device,
        len(test_dataset))

    # 将数字标签转换回原始标签
    test_labels = [REVERSE_LABEL_MAP[label] for label in test_labels]
    test_preds = [REVERSE_LABEL_MAP[pred] for pred in test_preds]

    # 计算测试集指标
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')

    print('\n' + '=' * 50)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Weighted F1: {test_f1:.4f}')
    print(f'Test Macro F1: {test_f1_macro:.4f}')
    print('\nClassification Report:')
    print(classification_report(test_labels, test_preds, target_names=EMOTION_LABELS))
    print('=' * 50)

    # 保存预测结果示例
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)[0]

    results = []
    for i, item in enumerate(test_data[:10]):  # 保存前10个示例的预测结果
        results.append({
            'text': item['content'],
            'true_label': item['label'],
            'predicted_label': test_preds[i],
            'probabilities': {REVERSE_LABEL_MAP[j]: f"{test_probs[i][j]:.4f}" for j in range(len(EMOTION_LABELS))}
        })

    with open('emotion_model/prediction_examples.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n训练完成！完整模型已保存到 'emotion_model' 目录")
    print("预测示例已保存为 'emotion_model/prediction_examples.json'")


if __name__ == '__main__':
    main()