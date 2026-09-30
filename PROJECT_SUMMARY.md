# SynthText4CH 项目回顾

> 目标：为 SynthText 合成中文场景文本。核心两件事——从中文维基语料**无监督构建中文词表 + 字频**，并接入渲染器按概率采"字 / 词 / 句子"渲染。

---

## 一、任务背景

- 仓库是 SynthText_CH（基于 `SynthText_Chinese_py3`），渲染合成场景文本图像。
- 原始文本源是英文 newsgroup（`text_utils.py` 里旧的 `_LegacyTextSource`），字频是 `char_freq.cp`。
- 需要：中文语料 `data/corpus/wiki_zh`（1.2GB、1274 个 JSONL 文件）→ 词表 + 字频 → 渲染器文本源。

---

## 二、核心模块与产出

| 模块 | 作用 |
|---|---|
| `build_vocab.py` | 三阶段词表管线：`NGram`(计数) → `PMI`(PPMI预过滤+收左右邻字) → `entropy`(打分输出) |
| `text_source.py` | 新文本源：中文 CHAR/WORD/LINE/PARA + 英文 newsgroup，中英 7:3 路由 |
| `test_text_source.py` | char_freq/word_freq 采样测试脚本 |
| `text_utils.py` / `synthgen.py` / `gen.py` | 渲染器接线 + 兼容性修复 |

**产出文件**（`build_vocab.py`）：

| 文件 | 内容 |
|---|---|
| `char_freq.cp` | `{字符: 归一化频率}`，键 = 字符表（通用汉字表 + 0-9 a-z A-Z） |
| `word_freq.cp` | `{词: 词频次数}`，2~6 字纯汉字词 |
| `word_scores.tsv` | `词 \t 词频 \t PPMI \t 左熵 \t 右熵 \t 综合分` |
| `words.txt` | 每行一词，按综合分降序 |
| `_ngram.pkl` / `_context.pkl` | 中间产物（分阶段缓存，断点续跑） |

---

## 三、算法与设计决策

### 1. 新词发现（无监督）

- 候选：纯汉字 n-gram，**2~6 字**；字母/数字作边界，不成词。
- **凝固度**：`PPMI = max( min 切分 PMI, 0 )`，`PMI = log₂( freq(w)·N / (freq(左)·freq(右)) )`。
- **自由度**：左右信息熵，筛 `min(左熵,右熵) ≥ 阈值`。
- **综合分**：`Score = √log₂(freq) × PPMI × min(左熵,右熵)`。

### 2. 三阶段 + 中间产物落盘

- `NGram` → 字频 + n-gram 计数（粗滤 `MIN_COUNT`）。
- `PMI` → 先按 PPMI 预过滤候选，再收左右邻字（省内存）。
- `entropy` → 算熵 + 综合分 + 输出。

好处：调 PPMI/熵/分阈值只需重跑 `entropy`（秒级），不重扫 1.2GB。

### 3. 内存优化（全量 1.2GB 能跑的关键）

- **有界计数器**：每个长度 n-gram 计数器上限 `CAP=5M`，超出抬 floor 删低频（lossy 近似）。
- **PPMI 预过滤**：候选 200 万 → PPMI≥7 后 27 万，`_context.pkl` 降到 126MB。

### 4. 全量结果

- 3.88 亿字符 → 候选 201 万 → 通过过滤 **99,022 词**（`MIN_COUNT=30, PPMI≥7, 熵≥1.5, 分≥18`）。

---

## 四、兼容性修复清单（Python 2→3 + 旧依赖→新依赖）

| 坑 | 文件 | 修复 |
|---|---|---|
| `char_freq.cp` 是 Python2 pickle（ascii 解码失败） | `data/models/` | 用新管线产物 `char_freq_ch.cp` 覆盖 |
| `Image.ANTIALIAS` 在 Pillow 10 移除 | `gen.py` | → `Image.LANCZOS` |
| `cv2.filter2D` 但只 `import cv2 as cv` | `gen.py` | → `cv.filter2D` |
| `cv2.findContours` OpenCV 4 返回 2 值（旧代码要 3 值） | `synthgen.py` | `img, contour, hier` → `contour, hier` |
| `signal.SIGALRM` Windows 不存在 | `common.py` | `time_limit` 加 `hasattr(signal,'SIGALRM')` no-op |
| `np.float` 在 numpy 1.24 移除 | `text_utils.py` | → `float` |
| `plt.hold` 在 matplotlib 3 移除 | `synthgen.py`、`poisson_reconstruct.py` | 删除 4 处 |
| `fftconvolve` FFT 内存爆（`std::bad_alloc`） | `text_utils.py` | → `cv2.matchTemplate(..., TM_CCORR)` |
| `np.min(空数组)` 崩 | `colorize3_poisson.py` | 加 `if loc[0].size == 0: continue` |
| **`font.render_to` 返回绝对坐标，旧代码又加一次坐标** | `text_utils.py` | 删 3 处多余坐标调整 |

### 关键 bug：`font.render_to` 坐标双加

```python
# pygame 2.5.2 的 render_to 已返回绝对坐标，下面两行是错的：
ch_bounds.x = x + ch_bounds.x   # x 翻倍
ch_bounds.y = y - ch_bounds.y   # y 归零
```

→ 同时导致「**有框无字**」（`crop_safe` 裁空）和「**文本与框不匹配**」（bbox 错位）。

---

## 五、踩坑经验与方法论

### 1. 规模不能治假阳性

n-gram + PPMI + 左右熵分不清"词"和"搭配/碎片"（如 `证据表明`、`了自己的`）。实测：语料 30→200 文件，假阳性**不消失、分数反而更高**（统计更自信）。只能靠：虚字过滤 / 词典校验（jieba）/ 人工审 top-N。

### 2. 先量再优化

用 `--limit` 子集实测耗时/内存，再决定是否上优化，别一开始就过度设计。

### 3. 阈值调参技巧

- `MIN_COUNT` 是阶段1参数（改它要重扫语料）。
- `PPMI_THRESH` / `ENTROPY_MIN` / `SCORE_THRESH` 是阶段3参数（秒级重调）。
- `SCORE_THRESH` 是"统一旋钮"，比单独调三者更直接。

### 4. 环境隔离

- `.venv` 是开发环境（numpy 1.26.4 / scipy 1.10.1 / pygame 2.5.2 等）。
- `--out-dir` 让测试产物与正式产物物理隔离，避免污染。

---

## 六、关键文件改动速查

| 文件 | 主要改动 |
|---|---|
| `build_vocab.py` | 新建：三阶段词表管线 + 有界计数器 + PPMI 预过滤 + loguru + --out-dir |
| `text_source.py` | 新建：中文/英文本源 + 预计算采样权重 |
| `test_text_source.py` | 新建：采样测试 |
| `text_utils.py` | 接线新 TextSource、`np.float`→`float`、`fftconvolve`→`matchTemplate`、修 `render_to` 坐标 |
| `synthgen.py` | `findContours` 3→2 值、删 `plt.hold` |
| `gen.py` | 路径、`ANTIALIAS`→`LANCZOS`、`cv2`→`cv`、NUM_IMG=2 |
| `common.py` | `time_limit` Windows no-op |
| `colorize3_poisson.py` | 空数组保护 |
| `poisson_reconstruct.py` | 删 `plt.hold` |
| `data/characters.txt` | 补充 0-9 a-z A-Z |
| `data/models/char_freq.cp` | 用新产物覆盖（原为 Python2 pickle） |
| `requirements.txt` | 升级依赖 + 加 loguru |
