import torch
import json
import os
import shutil
import argparse

def hierarchical_positional_encoding(ori_embed, new_pos, hp_alpha):
    position_ids = torch.arange(new_pos)  # [0, 1, 2, ..., 1023]
    i = position_ids // ori_embed.size(0)
    j = position_ids % ori_embed.size(0)
    base_embedding = (ori_embed - ori_embed[0:1] * hp_alpha) / (1 - hp_alpha)
    position_embeddings = hp_alpha * base_embedding[i] + (1 - hp_alpha) * base_embedding[j]
    return position_embeddings

def main(read_dir, save_dir, new_pos, hp_alpha):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 拷贝无关文件
    for name in os.listdir(read_dir):
        if name not in ["pytorch_model.bin", "config.json", "tokenizer_config.json"]:
            src_path = os.path.join(read_dir, name)
            dst_path = os.path.join(save_dir, name)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)

    # 更新 config.json
    with open(os.path.join(read_dir, "config.json"), "r", encoding="utf8") as fr:
        config_data = json.load(fr)
    ori_pos = config_data["max_position_embeddings"]
    config_data["max_position_embeddings"] = new_pos
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf8") as fw:
        json.dump(config_data, fw, ensure_ascii=False, indent=1)

    # 更新 tokenizer_config.json
    if os.path.exists(os.path.join(read_dir, "tokenizer_config.json")):
        with open(os.path.join(read_dir, "tokenizer_config.json"), "r", encoding="utf8") as fr:
            tokenizer_data = json.load(fr)
        tokenizer_data["model_max_length"] = new_pos
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf8") as fw:
            json.dump(tokenizer_data, fw, ensure_ascii=False, indent=1)

    # 加载原始模型权重
    ori_dict = torch.load(os.path.join(read_dir, "pytorch_model.bin"))

    # 获取原始位置嵌入
    ori_embed = None
    ori_embed_name = None
    for name, param in ori_dict.items():
        if "embeddings.position_embeddings.weight" in name:
            ori_embed = ori_dict[name]
            ori_embed_name = name
            break
    if ori_embed is None:
        raise ValueError("没有找到原始位置嵌入")
    position_embeddings = hierarchical_positional_encoding(ori_embed, new_pos, hp_alpha)

    # 更新位置嵌入
    ori_dict[ori_embed_name] = position_embeddings

    # 保存更新后的模型
    torch.save(ori_dict, os.path.join(save_dir, "pytorch_model.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="扩展模型位置嵌入长度（最大 Token 序列长度），并使用层次分解的位置编码进行初始化")
    parser.add_argument("--input", type=str, required=True, help="读取模型文件的目录")
    parser.add_argument("--output", type=str, required=True, help="保存新模型文件的目录")
    parser.add_argument("--new_pos", type=int, required=True, help="新的位置嵌入长度")
    parser.add_argument("--hp_alpha", type=float, default=0.4, help="插值参数 hp_alpha")

    args = parser.parse_args()

    main(args.input, args.output, args.new_pos, args.hp_alpha)
