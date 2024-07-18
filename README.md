# hierarchical-positional

使用层次分解位置编码技术，扩展模型最大 Token 序列长度（位置嵌入长度）的工具. 

A tool extending model position embedding length (maximum Token sequence length)

参考：苏剑林. (Dec. 04, 2020). 《层次分解位置编码，让BERT可以处理超长文本 》[Blog post]. Retrieved from https://kexue.fm/archives/7947

## 使用
```shell
python3 hierarchical_position.py --input bert-base-uncased/ --output bert-base-uncased-1024 --new_pos 1024
```

其中 new_pos 参数为新的位置嵌入长度
