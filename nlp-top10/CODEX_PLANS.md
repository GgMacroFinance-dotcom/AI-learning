# Codex 项目生成规划：NLP 十大核心算法

本文件把每个算法拆成一个适合交给 Codex 生成代码的项目任务。每个任务都包含 `goal`、`plan`、推荐目录结构、验收标准和可选扩展。建议一次只让 Codex 生成一个项目，先保证最小版本跑通，再扩展功能。

## 通用要求

所有项目默认采用以下约束：

- 语言：Python 3.10+
- 优先使用标准库、NumPy、scikit-learn、PyTorch；传统算法尽量从零实现核心逻辑。
- 每个项目必须包含：`README.md`、`requirements.txt`、`src/`、`tests/`、`data/sample.*`。
- 每个项目必须提供一个命令行 Demo。
- 每个项目必须写清楚：输入、输出、核心公式或伪代码、如何运行、如何测试。
- 每个项目要支持小数据快速验证，不依赖大型外部数据集才能启动。

推荐统一目录：

```text
projects/<project-name>/
|-- README.md
|-- requirements.txt
|-- data/
|   `-- sample.txt
|-- src/
|   |-- __init__.py
|   |-- preprocess.py
|   |-- model.py
|   |-- train.py 或 index.py
|   `-- predict.py 或 search.py
`-- tests/
    `-- test_core.py
```

---

## 01. N-gram 语言模型

### Goal

实现一个从零编写的 N-gram 语言模型，能够训练 unigram、bigram、trigram，支持加一平滑、句子概率计算、困惑度计算和下一个词预测。

### Codex Plan

```text
请在 projects/01-ngram-language-model 下生成一个 Python 项目，实现从零编写的 N-gram 语言模型。

要求：
1. 编写 tokenizer，支持英文空格分词和简单中文字符级分词两种模式。
2. 在 src/model.py 中实现 NGramLanguageModel 类。
3. 支持 n=1、n=2、n=3。
4. 训练时统计 n-gram 计数和上下文计数。
5. 支持 add-k smoothing，默认 k=1.0。
6. 实现 sentence_log_probability(sentence)、perplexity(sentences)、predict_next(context, top_k)。
7. 提供 CLI：
   - python -m src.train --data data/sample.txt --n 3
   - python -m src.predict --context "I love" --top-k 5
8. 写 tests/test_core.py，测试计数、概率非零、top-k 预测、困惑度为正数。
9. README 中解释 N-gram、平滑、困惑度，并给出运行示例。

不要调用现成语言模型库。核心概率计算必须自己实现。
```

### 验收标准

- 小样本训练能运行完成。
- 未见过的 n-gram 仍有非零概率。
- `predict_next` 能返回按概率排序的候选词。
- `pytest` 全部通过。

### 可选扩展

- 实现 Kneser-Ney smoothing。
- 增加随机句子生成。
- 可视化 n-gram 频率。

---

## 02. TF-IDF / BM25 搜索引擎

### Goal

实现一个小型文档搜索系统，支持 TF-IDF 和 BM25 两种排序方式，能够对输入查询返回相关文档列表，并展示关键词贡献分数。

### Codex Plan

```text
请在 projects/02-tfidf-bm25-search 下生成一个 Python 项目，实现小型文本搜索引擎。

要求：
1. 准备 data/docs.jsonl，每行包含 {"id": str, "title": str, "text": str}。
2. 在 src/preprocess.py 中实现分词、大小写归一化、停用词过滤。
3. 在 src/index.py 中实现倒排索引、TF、IDF、文档长度统计。
4. 在 src/rankers.py 中实现：
   - TF-IDF + cosine similarity
   - BM25，参数 k1=1.5, b=0.75
5. 在 src/search.py 中提供 CLI：
   - python -m src.search --query "neural language model" --ranker bm25 --top-k 5
6. 搜索结果输出文档 id、标题、分数、命中的关键词。
7. tests 中验证：倒排索引正确、IDF 为正、BM25 排序可运行、空查询不崩溃。
8. README 中解释 TF-IDF、BM25、倒排索引和二者差异。

核心索引和排序逻辑请自己实现，不要直接调用成熟搜索引擎。
```

### 验收标准

- 同一个查询可分别使用 `tfidf` 与 `bm25` 检索。
- 排名结果稳定、分数可解释。
- 支持新增文档后重建索引。

### 可选扩展

- 加入 query highlighting。
- 实现 nDCG、MRR 等检索评估指标。
- 输出一个简单 HTML 搜索页。

---

## 03. 朴素贝叶斯文本分类器

### Goal

实现一个多类别文本分类器，用词袋特征和多项式朴素贝叶斯完成训练、预测和评估。

### Codex Plan

```text
请在 projects/03-naive-bayes-classifier 下生成一个 Python 项目，实现多项式朴素贝叶斯文本分类器。

要求：
1. data/train.csv 包含 text,label 两列，准备一个小型样例数据集。
2. 实现 CountVectorizerLite：构建词表、文本转词频向量。
3. 实现 MultinomialNaiveBayes：
   - fit(X, y)
   - predict(X)
   - predict_proba(X)
   - 使用 log probability 避免数值下溢
   - 使用 Laplace smoothing
4. 提供 CLI：
   - python -m src.train --data data/train.csv --model models/nb.json
   - python -m src.predict --model models/nb.json --text "free prize click now"
5. 实现 accuracy、precision、recall、f1。
6. tests 验证：词表构建、类别先验、预测输出、保存加载模型。
7. README 中用垃圾邮件或情感分类例子解释朴素贝叶斯。

除可选评估对比外，核心模型不要直接调用 sklearn.naive_bayes。
```

### 验收标准

- 能训练、保存、加载、预测。
- 对小样本至少能输出合理类别。
- 所有概率计算使用 log 空间。

### 可选扩展

- 对比 scikit-learn 的 MultinomialNB。
- 支持 n-gram 特征。
- 加入错误样例分析报告。

---

## 04. HMM + Viterbi 序列标注器

### Goal

实现一个 HMM 序列标注器，用 Viterbi 算法预测最可能的标签序列，可用于玩具版词性标注或命名实体识别。

### Codex Plan

```text
请在 projects/04-hmm-viterbi-tagger 下生成一个 Python 项目，实现 HMM + Viterbi 序列标注器。

要求：
1. data/train.conll 使用每行 token tag，句子之间空行分隔。
2. 实现数据读取器，返回 token 序列和 tag 序列。
3. 在 src/model.py 中实现 HMMTagger：
   - 统计初始概率、转移概率、发射概率
   - 使用加一平滑处理未见 token 和转移
   - 使用 log probability
   - 实现 viterbi_decode(tokens)
4. 提供 CLI：
   - python -m src.train --data data/train.conll --model models/hmm.json
   - python -m src.predict --model models/hmm.json --sentence "John works at OpenAI"
5. 实现 token-level accuracy。
6. tests 验证：概率表完整、Viterbi 返回与输入等长的标签序列、未知词不崩溃。
7. README 中解释初始概率、转移概率、发射概率、Viterbi 动态规划。

核心 HMM 与 Viterbi 逻辑必须自己实现。
```

### 验收标准

- 能从 CoNLL 风格数据训练。
- 预测标签数量与 token 数量一致。
- 未见词能通过 `<UNK>` 或平滑处理。

### 可选扩展

- 加入中文 BMES 分词任务。
- 输出 Viterbi 动态规划表。
- 可视化最优路径。

---

## 05. CRF 序列标注器

### Goal

实现一个 CRF 序列标注项目，重点展示特征模板、序列级预测和与 HMM 的差异。核心可使用 sklearn-crfsuite，另需写清楚特征工程。

### Codex Plan

```text
请在 projects/05-crf-sequence-labeler 下生成一个 Python 项目，实现 CRF 序列标注器。

要求：
1. 使用 CoNLL 风格数据：每行 token tag，空行分隔句子。
2. 在 src/features.py 中实现 token2features(sentence, i)，特征包括：
   - 当前词 lower、suffix、prefix
   - 是否首字母大写、是否全数字
   - 前一个词和后一个词特征
   - BOS/EOS 标记
3. 使用 sklearn-crfsuite 或 python-crfsuite 训练 CRF。
4. 提供 CLI：
   - python -m src.train --data data/train.conll --model models/crf.pkl
   - python -m src.predict --model models/crf.pkl --sentence "John works at OpenAI"
5. 实现实体级 precision、recall、f1 的简化评估。
6. tests 验证：特征生成、训练、预测长度、模型保存加载。
7. README 中解释 CRF 为什么能全局考虑标签序列，并比较 HMM 与 CRF。
```

### 验收标准

- 能训练并保存 CRF 模型。
- 输出序列标签与输入 token 对齐。
- README 中有至少 5 个特征模板例子。

### 可选扩展

- 支持 BIO/BIES 标签合法性检查。
- 可视化特征权重。
- 与 HMM 项目共用同一测试集做对比。

---

## 06. Word2Vec 词向量训练器

### Goal

实现一个 Skip-gram + Negative Sampling 的小型 Word2Vec 训练器，能够训练词向量、查询相似词、做简单词向量类比。

### Codex Plan

```text
请在 projects/06-word2vec-embeddings 下生成一个 Python 项目，实现小型 Word2Vec 训练器。

要求：
1. 使用 PyTorch 实现 Skip-gram with Negative Sampling。
2. data/corpus.txt 提供小型训练语料。
3. 实现词表构建、低频词过滤、负采样分布。
4. 在 src/model.py 中实现 SkipGramNegSampling。
5. 在 src/train.py 中支持参数：embedding_dim、window_size、negative_samples、epochs、lr。
6. 在 src/query.py 中支持：
   - most_similar(word, top_k)
   - analogy(a, b, c)，例如 king - man + woman
7. 保存 embeddings 到 models/embeddings.npy 和 vocab.json。
8. tests 验证：训练 batch 形状、模型 forward、相似词接口不崩溃。
9. README 中解释 CBOW、Skip-gram、负采样和词向量相似度。
```

### 验收标准

- 训练过程 loss 可下降或至少稳定运行。
- 可以查询一个词的 top-k 相似词。
- 保存和加载词向量成功。

### 可选扩展

- 加入 t-SNE 或 PCA 可视化。
- 支持预训练词向量加载。
- 对比 Gensim Word2Vec。

---

## 07. TextCNN 文本分类器

### Goal

实现一个 TextCNN 文本分类项目，用多尺寸卷积核捕捉短语特征，完成情感分类或意图分类。

### Codex Plan

```text
请在 projects/07-textcnn-classifier 下生成一个 PyTorch 项目，实现 TextCNN 文本分类器。

要求：
1. data/train.csv 包含 text,label 两列。
2. 实现分词、词表、Dataset、DataLoader、padding。
3. 在 src/model.py 中实现 TextCNN：
   - Embedding layer
   - Conv1d kernels: 2, 3, 4
   - ReLU
   - Global max pooling
   - Dropout
   - Linear classifier
4. 训练脚本支持 epochs、batch_size、lr、embedding_dim。
5. 推理脚本支持：python -m src.predict --text "this movie is wonderful"
6. 输出 accuracy、macro-F1。
7. tests 验证：batch shape、forward 输出 shape、单步训练可运行。
8. README 中解释卷积核如何捕捉 n-gram 短语模式。
```

### 验收标准

- 模型能完成一次训练和预测。
- forward 输出形状为 `[batch_size, num_classes]`。
- README 有结构图或文字版结构说明。

### 可选扩展

- 加入预训练词向量初始化。
- 输出最激活的短语。
- 与朴素贝叶斯分类器做对比。

---

## 08. BiLSTM / GRU 序列模型

### Goal

实现一个基于 BiLSTM 或 GRU 的序列模型，可用于文本分类或序列标注，重点展示门控循环网络如何建模上下文。

### Codex Plan

```text
请在 projects/08-bilstm-gru-sequence-model 下生成一个 PyTorch 项目，实现 BiLSTM/GRU 文本分类器，并预留序列标注扩展。

要求：
1. data/train.csv 包含 text,label。
2. 实现 Dataset、padding、lengths、pack_padded_sequence。
3. 在 src/model.py 中实现 RNNClassifier：
   - embedding
   - 可选 rnn_type: lstm 或 gru
   - bidirectional=True
   - dropout
   - pooling: last 或 mean
   - classifier
4. 训练脚本支持选择 lstm/gru。
5. 推理脚本输出类别和置信度。
6. tests 验证：变长输入、forward shape、训练一步可运行。
7. README 中解释 RNN、LSTM、GRU 的差异，以及为什么需要 padding/mask。
```

### 验收标准

- LSTM 和 GRU 两种模式都能跑通。
- 变长 batch 不因 padding 出错。
- 能输出预测类别。

### 可选扩展

- 改造成 BiLSTM-CRF 序列标注器。
- 加入 attention pooling。
- 可视化隐藏状态。

---

## 09. Seq2Seq + Attention

### Goal

实现一个小型 Encoder-Decoder + Attention 模型，用玩具平行语料完成序列到序列任务，例如日期格式转换、简易翻译或文本复述。

### Codex Plan

```text
请在 projects/09-seq2seq-attention 下生成一个 PyTorch 项目，实现 Seq2Seq + Attention。

要求：
1. data/pairs.tsv 包含 source<TAB>target。
2. 实现源语言和目标语言词表，包含 <pad>、<sos>、<eos>、<unk>。
3. Encoder 使用 GRU 或 LSTM。
4. Decoder 使用 GRU 或 LSTM，并实现 Bahdanau attention 或 Luong attention。
5. 支持 teacher forcing ratio。
6. 训练脚本输出 loss。
7. 推理脚本支持 greedy decoding：
   - python -m src.translate --text "i love you"
8. 保存模型和词表。
9. tests 验证：attention 权重 shape、decoder 输出 shape、端到端小 batch 可运行。
10. README 中解释 Encoder、Decoder、Attention、Teacher Forcing。
```

### 验收标准

- 训练脚本可在 toy data 上运行。
- 推理能生成非空目标序列。
- Attention 权重长度与输入序列长度对齐。

### 可选扩展

- 实现 beam search。
- 可视化 attention heatmap。
- 增加 BLEU 或 exact match 评估。

---

## 10. Mini Transformer / BERT 风格实验室

### Goal

实现一个小型 Transformer 学习实验室，包含 Self-Attention 可视化、Transformer Encoder 文本分类，以及可选的 masked language modeling 玩具预训练。

### Codex Plan

```text
请在 projects/10-transformer-mini-lab 下生成一个 PyTorch 项目，实现 Mini Transformer NLP 实验室。

要求：
1. data/train.csv 包含 text,label，另提供 data/corpus.txt 用于可选 MLM。
2. 在 src/attention.py 中从零实现 scaled dot-product attention 和 multi-head attention。
3. 在 src/model.py 中实现 TransformerEncoderClassifier：
   - token embedding
   - positional encoding
   - N 层 Transformer encoder block
   - mean pooling 或 [CLS] pooling
   - classification head
4. 训练脚本完成文本分类。
5. 推理脚本输出预测类别。
6. 提供 attention 可视化函数，把 tokens 与 attention matrix 输出为 JSON 或 HTML。
7. tests 验证：attention shape、mask 生效、encoder forward shape、训练一步可运行。
8. README 中解释 Self-Attention、Multi-Head、Position Encoding、BERT 与 GPT 的区别。

可以使用 PyTorch 张量运算，但核心 attention 逻辑不要直接调用 nn.MultiheadAttention。
```

### 验收标准

- Attention 输出形状正确。
- 分类模型能完成一次训练和预测。
- 可导出至少一层注意力矩阵供网站展示。

### 可选扩展

- 增加 masked language modeling。
- 增加 tiny GPT decoder-only 生成模型。
- 加入 Hugging Face BERT 微调版本作为对照。

---

## 综合 Codex 任务：可视化学习网站

### Goal

把十大算法整理成一个交互式静态网站，帮助学习者按阶段、难度和任务类型理解算法关系。

### Codex Plan

```text
请在 nlp-top10/site 下生成一个无需构建工具的静态网站，使用 HTML、CSS、原生 JavaScript 实现。

要求：
1. 首页展示 NLP 十大算法学习地图。
2. 支持按任务类型筛选：表示、检索、分类、序列标注、生成、预训练。
3. 每个算法卡片展示：名称、一句话理解、输入、输出、核心思想、难度、适合项目。
4. 增加时间线视图：统计方法 -> 序列模型 -> 神经网络 -> 注意力 -> Transformer。
5. 增加对比面板：选择两个算法，对比它们的优势、局限和适用场景。
6. 增加学习进度功能：用户点击“已掌握”，用 localStorage 保存进度。
7. 页面应移动端友好，不依赖外部 CDN。
8. 代码拆分为 index.html、styles.css、app.js。
```

### 验收标准

- 直接打开 `index.html` 即可使用。
- 筛选、对比、进度保存均可用。
- 页面文字通俗，适合初学者。

---

## 十个项目的推荐完成顺序

1. `01-ngram-language-model`：先理解概率语言模型。
2. `02-tfidf-bm25-search`：理解词袋、权重和搜索。
3. `03-naive-bayes-classifier`：完成第一个分类模型。
4. `04-hmm-viterbi-tagger`：进入序列标注。
5. `05-crf-sequence-labeler`：理解特征工程与全局序列打分。
6. `06-word2vec-embeddings`：把词从离散符号变成向量。
7. `07-textcnn-classifier`：用神经网络抓局部文本模式。
8. `08-bilstm-gru-sequence-model`：学习上下文记忆。
9. `09-seq2seq-attention`：理解生成式任务。
10. `10-transformer-mini-lab`：进入现代大模型基础。

## 每个项目交付时的 README 模板

```markdown
# 项目名

## 目标

## 算法通俗解释

## 数据格式

## 安装

## 训练或索引

## 推理或搜索

## 测试

## 核心实现说明

## 常见问题

## 下一步扩展
```
