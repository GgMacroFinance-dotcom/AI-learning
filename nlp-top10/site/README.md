# NLP Algorithm Explorer 网站说明

这是一个无需构建工具的静态网站，用于可视化理解 NLP 十大核心算法。

## 本地打开

方式一：直接用浏览器打开：

```text
nlp-top10/site/index.html
```

方式二：启动本地静态服务器：

```bash
cd nlp-top10/site
python -m http.server 8000
```

然后访问本地页面。

## 功能

- 算法演进时间线：统计方法、序列模型、神经网络、注意力机制、Transformer。
- 算法卡片：展示通俗解释、输入输出、核心思想和 Codex 项目方向。
- 任务筛选：按表示、检索、分类、序列标注、生成、注意力、预训练筛选。
- 学习进度：点击“标记掌握”，使用浏览器 localStorage 保存进度。
- 算法对比：选择两个算法，对比核心思路、优势、局限和适用任务。

## 文件结构

```text
site/
|-- index.html   # 页面结构
|-- styles.css   # 响应式样式
|-- app.js       # 算法数据和交互逻辑
`-- README.md    # 使用说明
```

## 后续扩展建议

- 接入每个算法项目的真实运行结果。
- 为 Word2Vec 添加词向量二维可视化。
- 为 Seq2Seq 添加 Attention 热力图。
- 为 Transformer 添加多头注意力矩阵展示。
- 部署到 GitHub Pages。
