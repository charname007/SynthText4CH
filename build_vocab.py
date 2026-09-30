# -*- coding: utf-8 -*-
"""
从中文维基语料构建中文词表：n-gram + 词频 + PPMI(凝固度) + 左右熵(自由度)。

分三阶段执行，中间结果落盘，可断点续跑、调阈值不重扫语料：

    python build_vocab.py --stage NGram     # 阶段1：字频 + n-gram 计数（扫1遍语料）
    python build_vocab.py --stage PMI       # 阶段2：PPMI，候选词左右邻字 → （扫1遍语料）
    python build_vocab.py --stage entropy   # 阶段3：左右熵 + 综合分 + 输出（不扫语料）
    python build_vocab.py                   # 默认 all：依次 1→2→3
    python build_vocab.py --limit 200       # 调试：只读前 N 个语料文件
    python build_vocab.py --out-dir ./tmp/ --limit 200  # 测试：产物隔离，不污染正式结果

产物：
  1. data/characters.txt          补充 0-9 a-z A-Z（幂等，阶段1执行）
  2. data/models/char_freq.cp     {字符: 归一化频率}（阶段1）
  3. data/models/_ngram.pkl       候选 n-gram 计数 + 总字数（阶段1中间产物）
  4. data/models/_context.pkl     左右邻字分布（阶段2中间产物）
  5. data/models/word_freq.cp     {词: 词频}（阶段3）
  6. data/models/word_scores.tsv  词\\t词频\\tPMI\\t左熵\\t右熵\\t综合分（阶段3）
  7. data/words.txt               每行一词，按综合分降序（阶段3）

主体逻辑版：用朴素 Counter。内存优化点见文中 TODO（伪代码），未实现。
"""
import os
import re
import sys
import json
import math
import pickle as cp
import argparse
from collections import Counter, defaultdict

from loguru import logger

# ---------------- 配置 ----------------
CORPUS_DIR      = './data/corpus/wiki_zh'
CHARS_FILE      = './data/characters.txt'
CHAR_FREQ_OUT   = './data/models/char_freq_ch.cp'
WORD_FREQ_OUT   = './data/models/word_freq_ch.cp'
WORD_SCORES_OUT = './data/models/word_scores.tsv'
WORDS_TXT_OUT   = './data/words.txt'
NGRAM_PKL       = './data/models/_ngram.pkl'
CONTEXT_PKL     = './data/models/_context.pkl'
LOG_FILE        = './build_vocab.log'

NGRAM_MIN   = 2
NGRAM_MAX   = 6
MIN_COUNT   = 30      # 词频粗滤阈值
PPMI_THRESH = 7.0    # 凝固度阈值 (log2)
ENTROPY_MIN = 2.0    # 左右熵下限 (log2)
SCORE_THRESH = 18.0   # 综合分下限（0 = 不过滤）
CAP = 5_000_000       # 各长度 n-gram 计数器上限（防 OOM，超出则抬 floor 删低频）

HANZI_ALNUM_SPLIT = re.compile(pattern=r'[A-Za-z0-9]+')


# ---------------- 字符判定 ----------------
def is_hanzi(ch):
    return '一' <= ch <= '鿿'


def is_alnum(ch):
    return ('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ('0' <= ch <= '9')


# ---------------- 语料读取 / 清洗 ----------------
def iter_cleaned_texts(corpus_dir, limit=None):
    """
    生成器：逐文件逐行解析 wikizh_2019(JSONL格式)，取 text 字段，清洗后产出字符串。
    清洗规则：汉字 + 字母数字 保留，其余字符替换为空格（作边界）。
    """
    cnt = 0
    for sub in sorted(os.listdir(corpus_dir)):
        subdir = os.path.join(corpus_dir, sub)
        if not os.path.isdir(subdir):
            continue
        for fn in sorted(os.listdir(subdir)):
            if limit is not None and cnt >= limit:
                return
            cnt += 1
            path = os.path.join(subdir, fn)
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        logger.debug(f'跳过无法解析的行 ({path}): {e}')
                        continue
                    text = obj.get('text', '')
                    if not text:
                        continue
                    yield ''.join(ch if (is_hanzi(ch) or is_alnum(ch)) else ' '
                                  for ch in text)


# ---------------- 阶段1：n-gram 计数 + 字频 ----------------
def count_ngrams(texts, nmin, nmax, min_count, cap=CAP):
    """
    一遍流式统计（有界计数器，防 OOM）：
      counters[1]  -> 全量 1-gram 字频（汉字 + 字母数字），供 PMI 与 char_freq 用
      counters[L]  -> L 字纯汉字 n-gram 计数（2..nmax）
    每个长度超过 cap 时抬 floor 删低频（lossy 近似）：floor 始终远低于 min_count，
    所以常用（高频）词计数精确，只有贴近 floor 的低频词可能被低估；之后仍按
    min_count 做词频粗滤。
    """
    counters = {L: Counter() for L in range(1, nmax + 1)}
    floors = {L: 0 for L in range(nmin, nmax + 1)}
    for text in texts:
        for seg in text.split():
            if not seg:
                continue
            counters[1].update(seg)   # 字频：汉字 + 字母数字
            for hr in HANZI_ALNUM_SPLIT.split(seg):   # 词 n-gram：只统计纯汉字段
                n = len(hr)
                for L in range(nmin, min(nmax, n) + 1):
                    c = counters[L]
                    for i in range(n - L + 1):
                        c[hr[i:i + L]] += 1
        # 有界剪枝：每篇处理完检查，超 cap 就抬 floor 删低频，直到回落到 cap 内
        for L in range(nmin, nmax + 1):
            while len(counters[L]) > cap:
                floors[L] += 1
                counters[L] = Counter({g: v for g, v in counters[L].items()
                                       if v > floors[L]})
    logger.debug(f'计数完成，各长度剪枝 floor: {floors}')
    for L in range(nmin, nmax + 1):     # 词频粗滤：< min_count 丢弃
        counters[L] = Counter({w: c for w, c in counters[L].items()
                               if c >= min_count})
    return counters


# ---------------- 阶段2：收集候选词的左右邻字 ----------------
def collect_context(texts, candidates, nmin, nmax):
    """
    第二遍流式：只对候选词统计左邻字 / 右邻字分布（用于左右熵）。

    TODO(内存优化，未实现): 上下文 dict 随候选词数增长；候选极多时可分批处理，
    每批一趟流式、算完熵即释放：
        for batch in chunks(candidates, 100_000):
            left, right = 统计本批上下文
            for w in batch: 算熵、存结果、丢弃该词的上下文
    """
    left = defaultdict(Counter)
    right = defaultdict(Counter)
    for text in texts:
        for seg in text.split():
            for hr in HANZI_ALNUM_SPLIT.split(seg):
                n = len(hr)
                for L in range(nmin, min(nmax, n) + 1):
                    for i in range(n - L + 1):
                        w = hr[i:i + L]
                        if w in candidates:
                            if i > 0:
                                left[w][hr[i - 1]] += 1
                            if i + L < n:
                                right[w][hr[i + L]] += 1
    return left, right


# ---------------- 评分 ----------------
def ppmi(word, counters, total_chars):
    """
    凝固度（PPMI）：对所有切分点取最小 PMI，再 clamp 到非负。
    PPMI(w_left, w_right) = max( PMI(w_left, w_right), 0 )
    PMI = log2( freq(w) * N / (freq(left) * freq(right)) )
    """
    fw = counters[len(word)].get(word, 0)
    if fw == 0:
        return 0.0
    best = float('inf')
    L = len(word)
    for i in range(1, L):
        l, r = word[:i], word[i:]
        fl = counters[len(l)].get(l, 0)
        fr = counters[len(r)].get(r, 0)
        if fl == 0 or fr == 0:
            return 0.0
        val = math.log2(fw * total_chars / (fl * fr))
        best = min(best, val)
    return max(best, 0.0)   # PPMI = max(PMI, 0)


def entropy(counter):
    """信息熵（log2），空分布返回 0。"""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = -sum((v / total) * math.log2(v / total) for v in counter.values())
    return h + 0.0  # 归一化 -0.0 -> 0.0


def score(freq, p, hl, hr):
    """综合分 = sqrt(log2(freq)) × PPMI × min(左熵, 右熵)。"""
    return math.sqrt(math.log2(freq)) * p * min(hl, hr)


# ---------------- 字符表补充 ----------------
def supplement_characters(path):
    """因为通用汉字表只有汉字，所以在字符表末尾追加 0-9 a-z A-Z（幂等）。返回新增字符数。"""
    extras = ('0123456789'
              'abcdefghijklmnopqrstuvwxyz'
              'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    with open(path, encoding='utf-8') as f:
        data = f.read()
    existing = set(data.split())
    missing = [ch for ch in extras if ch not in existing]
    if missing:
        with open(path, 'a', encoding='utf-8') as f:
            if data and not data.endswith('\n'):
                f.write('\n')
            f.write(''.join(ch + '\n' for ch in missing))
    return len(missing)


# ---------------- 阶段实现 ----------------
def stage_NGram(limit):
    logger.info('阶段1：补充字符表 + 统计字频/n-gram ...')
    n_added = supplement_characters(CHARS_FILE)
    logger.info(f'字符表补充 {n_added} 个')
    with open(CHARS_FILE, encoding='utf-8') as f:
        char_table = f.read().split()

    counters = count_ngrams(iter_cleaned_texts(CORPUS_DIR, limit),
                            NGRAM_MIN, NGRAM_MAX, MIN_COUNT)
    total_chars = sum(counters[1].values())

    # 字频：只取字符表内字符，归一化（未出现 -> 0.0）
    char_freq = {ch: counters[1].get(ch, 0) for ch in sorted(char_table)}
    tot = sum(char_freq.values())
    if tot > 0:
        char_freq = {ch: c / tot for ch, c in char_freq.items()}
    with open(CHAR_FREQ_OUT, 'wb') as f:
        cp.dump(char_freq, f)

    with open(NGRAM_PKL, 'wb') as f:
        cp.dump({'counts': counters, 'total_chars': total_chars}, f)

    n_cand = sum(len(counters[L]) for L in range(NGRAM_MIN, NGRAM_MAX + 1))
    logger.info(f'总字符 {total_chars}, 候选词 {n_cand}')
    logger.info(f'已写 {CHAR_FREQ_OUT} 与 {NGRAM_PKL}')


def stage_PMI(limit):
    if not os.path.exists(NGRAM_PKL):
        raise SystemExit(f'缺少 {NGRAM_PKL}，请先跑 --stage NGram')
    logger.info('阶段2：先按 PPMI 过滤候选，再收集左右邻字 ...')
    with open(NGRAM_PKL, 'rb') as f:
        ngram = cp.load(f)
    counters = ngram['counts']
    total_chars = ngram['total_chars']

    # 先用词频算 PPMI（便宜），提前砍掉不达标的候选，省上下文内存
    candidates = set()
    n_all = 0
    for L in range(NGRAM_MIN, NGRAM_MAX + 1):
        for w in counters[L]:
            n_all += 1
            if ppmi(w, counters, total_chars) >= PPMI_THRESH:
                candidates.add(w)
    logger.info(f'候选 {n_all} -> PPMI 过滤后 {len(candidates)}')

    left, right = collect_context(iter_cleaned_texts(CORPUS_DIR, limit),
                                  candidates, NGRAM_MIN, NGRAM_MAX)
    with open(CONTEXT_PKL, 'wb') as f:
        cp.dump({'left': left, 'right': right}, f)
    logger.info(f'已写 {CONTEXT_PKL}')


def stage_entropy():
    for p in (NGRAM_PKL, CONTEXT_PKL):
        if not os.path.exists(p):
            raise SystemExit(f'缺少 {p}，请先跑对应阶段')
    logger.info('阶段3：左右熵 + 综合分 + 输出 ...')
    with open(NGRAM_PKL, 'rb') as f:
        ngram = cp.load(f)
    with open(CONTEXT_PKL, 'rb') as f:
        ctx = cp.load(f)
    counters = ngram['counts']
    total_chars = ngram['total_chars']
    left, right = ctx['left'], ctx['right']

    words = []
    for L in range(NGRAM_MIN, NGRAM_MAX + 1):
        for w, c in counters[L].items():
            p = ppmi(w, counters, total_chars)
            if p < PPMI_THRESH:
                continue
            hl = entropy(left.get(w, {}))
            hr = entropy(right.get(w, {}))
            if min(hl, hr) < ENTROPY_MIN:
                continue
            s = score(c, p, hl, hr)
            if s < SCORE_THRESH:
                continue
            words.append((w, c, p, hl, hr, s))

    words.sort(key=lambda x: x[5], reverse=True)
    write_word_outputs(words)
    logger.info(f'通过过滤词数 {len(words)}')
    logger.info('top 20:')
    for w, c, p, hl, hr, s in words[:20]:
        logger.info(f'{w}\t频{c}\tPPMI{p:.2f}\t左熵{hl:.2f}\t右熵{hr:.2f}\t分{s:.2f}')


# ---------------- 输出 ----------------
def write_word_outputs(words):
    """words: list of (word, count, pmi, hl, hr, score)，已按 score 降序。"""
    word_freq = {w: c for w, c, _, _, _, _ in words}
    with open(WORD_FREQ_OUT, 'wb') as f:
        cp.dump(word_freq, f)
    with open(WORD_SCORES_OUT, 'w', encoding='utf-8') as f:
        for w, c, p, hl, hr, s in words:
            f.write(f'{w}\t{c}\t{p:.4f}\t{hl:.4f}\t{hr:.4f}\t{s:.4f}\n')
    with open(WORDS_TXT_OUT, 'w', encoding='utf-8') as f:
        for w, *_ in words:
            f.write(w + '\n')


# ---------------- 日志 ----------------
def setup_logger():
    """配置 loguru：控制台 INFO（彩色）+ 文件 DEBUG（滚动）。"""
    logger.remove()  # 移除默认 handler，避免重复输出
    logger.add(sys.stderr, level='INFO', colorize=True)
    logger.add(LOG_FILE, level='DEBUG', rotation='10 MB', encoding='utf-8')
    return logger


# ---------------- 主流程 ----------------
def apply_out_dir(out_dir):
    """把输出产物重定向到 out_dir（测试时用它隔离，避免覆盖正式产物）。"""
    global CHAR_FREQ_OUT, WORD_FREQ_OUT, WORD_SCORES_OUT, WORDS_TXT_OUT, \
        NGRAM_PKL, CONTEXT_PKL, LOG_FILE
    os.makedirs(out_dir, exist_ok=True)
    CHAR_FREQ_OUT   = os.path.join(out_dir, 'char_freq_ch.cp')
    WORD_FREQ_OUT   = os.path.join(out_dir, 'word_freq_ch.cp')
    WORD_SCORES_OUT = os.path.join(out_dir, 'word_scores.tsv')
    WORDS_TXT_OUT   = os.path.join(out_dir, 'words.txt')
    NGRAM_PKL       = os.path.join(out_dir, '_ngram.pkl')
    CONTEXT_PKL     = os.path.join(out_dir, '_context.pkl')
    LOG_FILE        = os.path.join(out_dir, 'build_vocab.log')


def main():
    ap = argparse.ArgumentParser(description='中文词表构建（分步）')
    ap.add_argument('--stage', choices=['NGram', 'PMI', 'entropy', 'all'],
                    default='all', help='执行阶段（默认 all）')
    ap.add_argument('--limit', type=int, default=None,
                    help='只读前 N 个语料文件（调试用）')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='输出目录（测试时指定临时目录可隔离产物，避免覆盖正式结果）')
    args = ap.parse_args()

    if args.out_dir:
        apply_out_dir(args.out_dir)
    setup_logger()

    if args.stage in ('NGram', 'all'):
        stage_NGram(args.limit)
    if args.stage in ('PMI', 'all'):
        stage_PMI(args.limit)
    if args.stage in ('entropy', 'all'):
        stage_entropy()
    logger.info('完成')


if __name__ == '__main__':
    main()
