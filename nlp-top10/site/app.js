const algorithms = [
  {
    id: "ngram",
    rank: 1,
    name: "N-gram 语言模型",
    short: "用前面几个词猜下一个词，像短期记忆版自动补全。",
    input: "分词后的文本序列",
    output: "句子概率、下一个词候选、困惑度",
    idea: "统计局部上下文中词的共现概率。",
    strength: "简单、可解释、适合理解语言模型入门。",
    weakness: "只看短窗口，难处理长距离依赖和语义泛化。",
    project: "从零实现 unigram/bigram/trigram、平滑、困惑度和 top-k 预测。",
    categories: ["表示", "生成"],
    difficulty: 2,
    era: "统计方法"
  },
  {
    id: "tfidf-bm25",
    rank: 2,
    name: "TF-IDF / BM25",
    short: "越能代表当前文档、越少出现在其他文档中的词，越重要。",
    input: "文档集合与用户查询",
    output: "关键词权重、相关文档排序",
    idea: "用词频、逆文档频率和文档长度归一化衡量相关性。",
    strength: "搜索召回强、速度快、分数容易解释。",
    weakness: "依赖词面匹配，不天然理解同义词和语境。",
    project: "实现倒排索引、TF-IDF、BM25、命中词解释和搜索 CLI。",
    categories: ["检索", "表示"],
    difficulty: 2,
    era: "统计方法"
  },
  {
    id: "naive-bayes",
    rank: 3,
    name: "朴素贝叶斯",
    short: "看到一组词，估计这段文本最像哪个类别。",
    input: "文本词袋特征",
    output: "类别概率与预测标签",
    idea: "用贝叶斯公式结合类别先验和词的似然概率。",
    strength: "训练快、小数据可用、适合分类基线。",
    weakness: "独立性假设过强，难表达复杂词序和上下文。",
    project: "从零实现多项式朴素贝叶斯、log 概率、拉普拉斯平滑和评估。",
    categories: ["分类"],
    difficulty: 2,
    era: "统计方法"
  },
  {
    id: "hmm",
    rank: 4,
    name: "HMM + Viterbi",
    short: "词是台前可见结果，标签是幕后状态，用概率找最可能路径。",
    input: "token 序列",
    output: "最可能的标签序列",
    idea: "用初始、转移、发射概率建模隐藏状态序列。",
    strength: "序列标注入门经典，Viterbi 动态规划清晰。",
    weakness: "特征表达能力有限，依赖强假设。",
    project: "实现 CoNLL 数据读取、HMM 训练、Viterbi 解码和未知词处理。",
    categories: ["序列标注"],
    difficulty: 3,
    era: "序列模型"
  },
  {
    id: "crf",
    rank: 5,
    name: "CRF",
    short: "不逐个猜标签，而是给整条标签路径打分。",
    input: "token 序列与人工特征",
    output: "全局最优标签序列",
    idea: "综合当前词、上下文和标签转移特征进行序列级归一化。",
    strength: "适合 NER、分词、槽位识别，能加入丰富特征。",
    weakness: "依赖特征工程，深层语义表示能力有限。",
    project: "实现特征模板、CRF 训练预测、实体级 F1 和 HMM 对比。",
    categories: ["序列标注"],
    difficulty: 4,
    era: "序列模型"
  },
  {
    id: "word2vec",
    rank: 6,
    name: "Word2Vec",
    short: "看一个词身边常出现谁，就能学到它的语义位置。",
    input: "大规模无标注语料",
    output: "词向量、相似词、类比关系",
    idea: "通过上下文预测学习稠密向量表示。",
    strength: "让词可计算相似度，是神经 NLP 的基础表示。",
    weakness: "静态词向量难处理一词多义和上下文变化。",
    project: "用 PyTorch 实现 Skip-gram + Negative Sampling、相似词查询和类比。",
    categories: ["表示", "预训练"],
    difficulty: 3,
    era: "神经网络"
  },
  {
    id: "textcnn",
    rank: 7,
    name: "TextCNN",
    short: "用不同长度的卷积窗口扫描句子，抓住关键短语。",
    input: "token id 序列",
    output: "文本类别",
    idea: "卷积核提取局部 n-gram 模式，池化保留最强信号。",
    strength: "短文本分类强、结构简单、训练高效。",
    weakness: "主要捕捉局部模式，长距离依赖能力弱。",
    project: "实现 Embedding、Conv1d、多窗口卷积、Max Pooling 和分类训练。",
    categories: ["分类"],
    difficulty: 3,
    era: "神经网络"
  },
  {
    id: "rnn-lstm-gru",
    rank: 8,
    name: "RNN / LSTM / GRU",
    short: "像边读边记笔记，逐词更新上下文记忆。",
    input: "有顺序的 token 序列",
    output: "文本类别、序列表示或标签序列",
    idea: "循环更新隐藏状态，LSTM/GRU 用门控决定记住与遗忘。",
    strength: "天然建模顺序，适合学习上下文记忆机制。",
    weakness: "顺序计算慢，长文本依赖仍然困难。",
    project: "实现 BiLSTM/GRU 分类器、变长 batch、pack_padded_sequence 和推理。",
    categories: ["分类", "序列标注", "生成"],
    difficulty: 4,
    era: "神经网络"
  },
  {
    id: "seq2seq-attention",
    rank: 9,
    name: "Seq2Seq + Attention",
    short: "一个模型读输入，另一个模型生成输出，生成时回看重点。",
    input: "源文本序列",
    output: "目标文本序列",
    idea: "Encoder 编码输入，Decoder 解码输出，Attention 对齐关键上下文。",
    strength: "适合翻译、摘要、对话等生成式任务。",
    weakness: "训练和评估复杂，生成质量不只看 loss。",
    project: "实现 Encoder-Decoder、Bahdanau/Luong Attention、Teacher Forcing 和 Greedy Decode。",
    categories: ["生成", "注意力"],
    difficulty: 5,
    era: "注意力机制"
  },
  {
    id: "transformer",
    rank: 10,
    name: "Transformer / BERT / GPT",
    short: "每个词直接关注所有词，用注意力构建上下文理解与生成。",
    input: "token 序列、位置编码、注意力 mask",
    output: "上下文表示、分类结果或生成文本",
    idea: "Self-Attention + 多头机制 + 前馈网络并行建模全局关系。",
    strength: "现代 NLP 和大模型基础，迁移能力强。",
    weakness: "数据、算力、训练目标和评估体系同样关键。",
    project: "从零实现 scaled dot-product attention、mini Transformer encoder 分类器和注意力可视化。",
    categories: ["预训练", "分类", "生成", "注意力"],
    difficulty: 5,
    era: "Transformer"
  }
];

const stages = [
  {
    title: "统计方法",
    desc: "先把文本当作可统计的词，理解概率、词频和文档相关性。",
    ids: ["ngram", "tfidf-bm25", "naive-bayes"]
  },
  {
    title: "序列模型",
    desc: "开始把文本看成有顺序的标签路径，学习动态规划与全局解码。",
    ids: ["hmm", "crf"]
  },
  {
    title: "神经网络",
    desc: "把词变成向量，用神经网络学习短语模式和上下文记忆。",
    ids: ["word2vec", "textcnn", "rnn-lstm-gru"]
  },
  {
    title: "注意力机制",
    desc: "让生成模型在输出时主动回看输入重点，突破固定向量瓶颈。",
    ids: ["seq2seq-attention"]
  },
  {
    title: "Transformer",
    desc: "用 Self-Attention 并行建模全局关系，进入预训练和大模型时代。",
    ids: ["transformer"]
  }
];

const categories = ["全部", "表示", "检索", "分类", "序列标注", "生成", "注意力", "预训练"];
const storageKey = "nlp-top10-mastered";
let activeCategory = "全部";

function loadMastered() {
  try {
    return new Set(JSON.parse(localStorage.getItem(storageKey) || "[]"));
  } catch (error) {
    return new Set();
  }
}

function saveMastered(mastered) {
  localStorage.setItem(storageKey, JSON.stringify([...mastered]));
}

function stars(value) {
  return "★".repeat(value) + "☆".repeat(5 - value);
}

function byId(id) {
  return algorithms.find((algorithm) => algorithm.id === id);
}

function renderTimeline() {
  const timeline = document.getElementById("timeline");
  timeline.innerHTML = stages.map((stage, index) => {
    const tags = stage.ids.map((id) => `<span class="tag">${byId(id).name}</span>`).join("");
    return `
      <article class="timeline-item" data-step="${index + 1}">
        <h3>${stage.title}</h3>
        <p>${stage.desc}</p>
        <div class="timeline-tags">${tags}</div>
      </article>
    `;
  }).join("");
}

function renderFilters() {
  const filters = document.getElementById("filters");
  filters.innerHTML = categories.map((category) => `
    <button class="filter-button ${category === activeCategory ? "active" : ""}" data-category="${category}" type="button">
      ${category}
    </button>
  `).join("");

  filters.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      activeCategory = button.dataset.category;
      renderFilters();
      renderCards();
    });
  });
}

function renderCards() {
  const mastered = loadMastered();
  const cardGrid = document.getElementById("cardGrid");
  const visible = activeCategory === "全部"
    ? algorithms
    : algorithms.filter((algorithm) => algorithm.categories.includes(activeCategory));

  cardGrid.innerHTML = visible.map((algorithm) => {
    const done = mastered.has(algorithm.id);
    const tags = algorithm.categories.map((category) => `<span class="tag">${category}</span>`).join("");
    return `
      <article class="algorithm-card">
        <div class="card-top">
          <span class="rank">${algorithm.rank}</span>
          <div>
            <div class="card-title-row">
              <h3>${algorithm.name}</h3>
              <span class="difficulty" title="难度">${stars(algorithm.difficulty)}</span>
            </div>
            <p class="card-plain">${algorithm.short}</p>
          </div>
        </div>
        <div class="detail-grid">
          <div class="detail"><span>输入</span>${algorithm.input}</div>
          <div class="detail"><span>输出</span>${algorithm.output}</div>
          <div class="detail"><span>核心思想</span>${algorithm.idea}</div>
          <div class="detail"><span>Codex 项目</span>${algorithm.project}</div>
        </div>
        <div class="card-tags">
          <span class="tag">${algorithm.era}</span>
          ${tags}
        </div>
        <button class="master-button ${done ? "done" : ""}" type="button" data-id="${algorithm.id}">
          ${done ? "已掌握，点击取消" : "标记掌握"}
        </button>
      </article>
    `;
  }).join("");

  cardGrid.querySelectorAll(".master-button").forEach((button) => {
    button.addEventListener("click", () => {
      const id = button.dataset.id;
      const current = loadMastered();
      if (current.has(id)) {
        current.delete(id);
      } else {
        current.add(id);
      }
      saveMastered(current);
      renderCards();
      updateProgress();
    });
  });
}

function updateProgress() {
  const mastered = loadMastered();
  const count = mastered.size;
  const ratio = Math.round((count / algorithms.length) * 100);
  document.getElementById("progressText").textContent = `${count} / ${algorithms.length}`;
  document.getElementById("progressBar").style.width = `${ratio}%`;
  document.getElementById("progressHint").textContent = count === 0
    ? "点击算法卡片里的“标记掌握”，开始记录你的学习进度。"
    : count === algorithms.length
      ? "你已经完成十大算法地图，可以开始做综合项目了。"
      : `已完成 ${ratio}%。下一步建议继续按路线推进并完成对应项目。`;
}

function renderCompareOptions() {
  const options = algorithms.map((algorithm) => `
    <option value="${algorithm.id}">${algorithm.rank}. ${algorithm.name}</option>
  `).join("");

  const compareA = document.getElementById("compareA");
  const compareB = document.getElementById("compareB");
  compareA.innerHTML = options;
  compareB.innerHTML = options;
  compareA.value = "tfidf-bm25";
  compareB.value = "transformer";

  compareA.addEventListener("change", renderCompare);
  compareB.addEventListener("change", renderCompare);
}

function compareCard(algorithm) {
  return `
    <article class="compare-card">
      <h3>${algorithm.name}</h3>
      <p>${algorithm.short}</p>
      <dl>
        <dt>核心思路</dt>
        <dd>${algorithm.idea}</dd>
        <dt>优势</dt>
        <dd>${algorithm.strength}</dd>
        <dt>局限</dt>
        <dd>${algorithm.weakness}</dd>
        <dt>适合任务</dt>
        <dd>${algorithm.categories.join("、")}</dd>
        <dt>实现项目</dt>
        <dd>${algorithm.project}</dd>
      </dl>
    </article>
  `;
}

function renderCompare() {
  const a = byId(document.getElementById("compareA").value);
  const b = byId(document.getElementById("compareB").value);
  document.getElementById("compareGrid").innerHTML = compareCard(a) + compareCard(b);
}

function init() {
  renderTimeline();
  renderFilters();
  renderCards();
  updateProgress();
  renderCompareOptions();
  renderCompare();
}

init();
