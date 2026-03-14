import json

# 输入.txt文件路径
input_file = "usual_eval_labeled.txt"
# 输出.json文件路径
output_file = "usual_eval.json"

with open(input_file, 'r', encoding='utf-8') as f_in, \
        open(output_file, 'w', encoding='utf-8') as f_out:
    # 读取所有行并解析为JSON对象
    data = [json.loads(line.strip()) for line in f_in if line.strip()]

    # 写入为标准JSON数组
    json.dump(data, f_out, ensure_ascii=False, indent=2)

print(f"转换完成！输出文件: {output_file}")