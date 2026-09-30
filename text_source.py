# -*- coding: utf-8 -*-
"""中英文文本采样：字 / 词 / 行 / 段。

- 英文：原论文 Sec 2.1，从 data/newsgroup.txt 采 WORD / LINE / PARA（连续行 + Beta）。
- 中文：CHAR / WORD 走频率表（char_freq / word_freq），LINE / PARA 惰性随机读
  wiki 语料（data/corpus/wiki_zh，JSONL）+ 滑动窗口截取定长文本。
"""
import os
import re
import json
import random
import os.path as osp
import pickle as cp
import numpy as np
from scipy import stats as sstat


def is_hanzi(ch):
    return '一' <= ch <= '鿿'


def is_alnum(ch):
    return ('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ('0' <= ch <= '9')


def sample_weighted(p_dict):
    keys = list(p_dict.keys())
    vals = np.array([p_dict[k] for k in keys], dtype='float64')
    vals += 1e-12
    vals /= vals.sum()
    return keys[np.random.choice(len(keys), p=vals)]


class WikiSentenceSource(object):
    """（备选）从 data/corpus/wiki_zh（JSONL）随机抽母句。

    当前中文 LINE/PARA 已改用故事语料惰性采样，本类保留备用；
    wiki 仍是 char_freq / word_freq 的离线来源（build_vocab.py）。
    """
    _SPLIT = re.compile(r'[。！？!?；;：:，,、]')

    def __init__(self, root, pool_size=20000):
        self.root = root
        self.pool_size = pool_size
        self._file_list = []
        for dp, _, fs in os.walk(root):
            for f in fs:
                self._file_list.append(osp.join(dp, f))
        self._pool = []
        self._refill()

    def _sentences_from_file(self, path):
        out = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get('text', '')
                text = text.replace('\\n', '\n')
                for seg in text.split('\n'):
                    for s in self._SPLIT.split(seg):
                        s = s.strip()
                        if s and any(is_hanzi(ch) for ch in s):
                            out.append(s)
        return out

    def _refill(self):
        random.shuffle(self._file_list)
        while len(self._pool) < self.pool_size and self._file_list:
            sents = self._sentences_from_file(self._file_list.pop())
            need = self.pool_size - len(self._pool)
            self._pool.extend(sents if len(sents) <= need else random.sample(sents, need))
        random.shuffle(self._pool)

    def sample_parent(self, min_len, niter=100):
        for _ in range(niter):
            if not self._pool:
                self._refill()
            if not self._pool:
                return ''
            s = random.choice(self._pool)
            if len(s) >= min_len:
                return s
        return max(self._pool, key=len) if self._pool else ''


class CorpusTextSource(object):
    """中英文共用骨架：符号占比过滤 / 行有效性 / 居中排版 / 加权采样 / 类型分派。

    子类提供 `fdict`、`p_text` 及各具体采样方法（WORD/LINE/PARA，中文另有 CHAR）。
    """

    def __init__(self, min_nchar):
        self.min_nchar = min_nchar
        self.fdict={}

    def check_symb_frac(self, txt, f=0.35):
        return np.sum([not ch.isalnum() for ch in txt]) / (len(txt) + 0.0) <= f

    def is_good(self, txt, f=0.35):
        def is_txt(l):
            char_ex = ['i', 'I', 'o', 'O', '0', '-']
            return not np.all([ch in char_ex for ch in l])
        return [(len(l) > self.min_nchar
                 and self.check_symb_frac(l, f)
                 and is_txt(l)) for l in txt]

    def center_align(self, lines):
        ls = [len(l) for l in lines]
        max_l = max(ls)
        for i in range(len(lines)):
            l = lines[i].strip()
            dl = max_l - ls[i]
            lspace = dl // 2
            rspace = dl - lspace
            lines[i] = ' ' * lspace + l + ' ' * rspace
        return lines

    def _weighted_freq(self, freq_dict):
        keys = list(freq_dict.keys())
        w = np.array([freq_dict[k] for k in keys], dtype='float64')
        w += 1e-12
        w /= w.sum()
        return keys[np.random.choice(len(keys), p=w)]

    def _weighted_inverse(self, freq_dict):
        keys = list(freq_dict.keys())
        vals = np.array([freq_dict[k] for k in keys], dtype='float64')
        # count 场景用 +1 平滑；归一化频率(最大值<1)用小 eps 防退化成均匀
        eps = 1.0 if vals.max() >= 1.0 else 1e-8
        w = 1.0 / (vals + eps)
        w /= w.sum()
        return keys[np.random.choice(len(keys), p=w)]

    def _weighted_power(self, items, alpha):
        ws = [it[0] for it in items]
        vals = np.array([it[1] for it in items], dtype='float64')
        vals = vals ** alpha
        vals += 1e-12
        vals /= vals.sum()
        return ws[np.random.choice(len(ws), p=vals)]

    def sample(self, nline_max, nchar_max, kind=None):
        k = kind if kind is not None else sample_weighted(self.p_text)
        return self.fdict[k](nline_max, nchar_max)


class EnglishTextSource(CorpusTextSource):
    """原论文 Sec 2.1：从 data/newsgroup.txt（20-newsgroups）采 WORD/LINE/PARA。"""
    _HEADER = re.compile(r'^[A-Za-z][A-Za-z0-9-]*:')

    def __init__(self, min_nchar, fn):
        super().__init__(min_nchar)
        self.fdict = {'WORD': self.sample_word,
                      'LINE': self.sample_line,
                      'PARA': self.sample_para}
        self.p_text = {'WORD': 0.60, 'LINE': 0.30, 'PARA': 0.10}

        with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for l in f:
                l = l.strip()
                if not l or self._HEADER.match(l):
                    continue  # 跳过空行与新闻组头部（From/Subject/Lines 等）
                l = re.sub(r'^>+\s*', '', l)  # 去掉引用回复的 ">" 标记
                if l:
                    lines.append(l)
        self.txt = lines

        # LINE/PARA 的行数、词数分布
        self.p_line_nline = np.array([0.85, 0.10, 0.05])  # 1/2/3 行概率
        self.p_line_nword = [4, 3, 12]        # beta(a, b) + 词数上限
        self.p_para_nline = [1.0, 1.0]
        self.p_para_nword = [1.7, 3.0, 10]
        self.center_para = 0.5

    def get_lines(self, nline, nword, nchar_max, f=0.35, niter=100):
        def h_lines(niter=100):
            lines = ['']
            iter = 0
            while not np.all(self.is_good(lines, f)) and iter < niter:
                iter += 1
                line_start = np.random.choice(len(self.txt) - nline)
                lines = [self.txt[line_start + i] for i in range(nline)]
            return lines

        lines = ['']
        iter = 0
        while not np.all(self.is_good(lines, f)) and iter < niter:
            iter += 1
            lines = h_lines(niter=100)
            nline = len(lines)
            for i in range(nline):
                words = lines[i].split()
                dw = len(words) - nword[i]
                if dw > 0:
                    first_word_index = random.choice(range(dw + 1))
                    lines[i] = ' '.join(words[first_word_index:first_word_index + nword[i]])
                while len(lines[i]) > nchar_max:  # 超长从行尾按词截断
                    if not np.any([ch.isspace() for ch in lines[i]]):
                        lines[i] = ''
                    else:
                        lines[i] = lines[i][:len(lines[i]) - lines[i][::-1].find(' ')].strip()

        if not np.all(self.is_good(lines, f)):
            return None
        return lines

    def sample_word(self, nline_max, nchar_max, niter=100):
        rand_word = ''
        iter = 0
        while iter < niter:
            rand_line = self.txt[np.random.choice(len(self.txt))]
            words = rand_line.split()
            if not words:
                iter += 1
                continue
            rand_word = random.choice(words)
            if self.is_good([rand_word])[0] and len(rand_word) <= nchar_max:
                return rand_word
            iter += 1
        return []

    def sample_line(self, nline_max, nchar_max):
        nline = nline_max + 1
        while nline > nline_max:
            nline = np.random.choice([1, 2, 3], p=self.p_line_nline)

        nword = [self.p_line_nword[2] * sstat.beta.rvs(a=self.p_line_nword[0], b=self.p_line_nword[1])
                 for _ in range(nline)]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            return '\n'.join(lines)
        return []

    def sample_para(self, nline_max, nchar_max):
        nline = nline_max * sstat.beta.rvs(a=self.p_para_nline[0], b=self.p_para_nline[1])
        nline = max(1, int(np.ceil(nline)))

        nword = [self.p_para_nword[2] * sstat.beta.rvs(a=self.p_para_nword[0], b=self.p_para_nword[1])
                 for _ in range(nline)]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            if np.random.rand() < self.center_para:
                lines = self.center_align(lines)
            return '\n'.join(lines)
        return []


class ChineseTextSource(CorpusTextSource):
    """中文：CHAR/WORD 走频率表；LINE/PARA 惰性随机读 wiki_zh 语料 + 滑动窗口截取。"""

    def __init__(self, min_nchar, files, char_freq, words_by_len):
        super().__init__(min_nchar)
        self.files = files
        self.char_freq = char_freq
        self.words_by_len = words_by_len
        self.fdict = {'CHAR': self.sample_char,
                      'WORD': self.sample_word,
                      'LINE': self.sample_line,
                      'PARA': self.sample_para}
        self.p_text = {'CHAR': 0.08, 'WORD': 0.55, 'LINE': 0.27, 'PARA': 0.10}

        # 惰性缓存：当前文件的纯文本串（去空白）
        self._text = ''

        # 单字：70% 逆频次，30% 频率加权
        self.p_inv_char = 0.70
        # 词长度分层 + 频率幂次（alpha 温度）
        self.p_word_len = {2: 0.30, 3: 0.30, 4: 0.20, 5: 0.10, 6: 0.10}
        self.word_alpha = 0.4
        # 行文本长度下限（LINE 截取 [sent_len_min, nline*nchar] 字）
        self.sent_len_min = 7
        self.center_para = 0.5

        # ---- 预计算采样权重（避免每次采样都重建数组）----
        # 单字：频率权重 + 逆频次权重
        self._char_keys = list(char_freq.keys())
        self._char_w = np.array([char_freq[k] for k in self._char_keys], dtype='float64')
        self._char_w = self._char_w / self._char_w.sum()
        self._char_inv_w = 1.0 / (self._char_w + 1e-8)
        self._char_inv_w = self._char_inv_w / self._char_inv_w.sum()
        # 词：按长度分层的频率幂次权重
        self._word_keys = {}
        self._word_w = {}
        for L, items in words_by_len.items():
            ws = [it[0] for it in items]
            vals = np.array([it[1] for it in items], dtype='float64') ** self.word_alpha
            vals = vals + 1e-12
            vals = vals / vals.sum()
            self._word_keys[L] = ws
            self._word_w[L] = vals

    def _read_wiki_file(self, path):
        """读一个 wiki JSONL 文件，只保留汉字与 ASCII 字母数字，拼成纯文本串。"""
        parts = []
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = ''.join(ch for ch in obj.get('text', '')
                               if is_hanzi(ch) 
                            #    or is_alnum(ch)
                               )
                if text:
                    parts.append(text)
        return ''.join(parts)

    def _text_src(self):
        """惰性取文本：随机挑一个 wiki 文件读入；30% 概率换文件以增加多样性。"""
        if not self._text or np.random.rand() < 0.3:
            self._text = self._read_wiki_file(random.choice(self.files))
        return self._text

    def _window(self, L):
        """滑动窗口：从文本里随机起点截取 L 个字符。"""
        text = self._text_src()
        if not text:
            return ''
        if len(text) <= L:
            return text
        start = np.random.randint(0, len(text) - L + 1)
        return text[start:start + L]

    def get_lines(self, L, nchar_max):
        """滑动窗口截 L 字，按 nchar_max 折成多行。"""
        text = self._window(L)
        if not text:
            return []
        return [text[i:i + nchar_max] for i in range(0, len(text), nchar_max)]

    def sample_char(self, nline_max, nchar_max):
        w = self._char_inv_w if np.random.rand() < self.p_inv_char else self._char_w
        return self._char_keys[np.random.choice(len(self._char_keys), p=w)]

    def sample_word(self, nline_max, nchar_max, niter=100):
        for _ in range(niter):
            L = sample_weighted(self.p_word_len)
            if L > nchar_max:
                continue
            ks = self._word_keys.get(L)
            if not ks:
                continue
            return ks[np.random.choice(len(ks), p=self._word_w[L])]
        return []

    def sample_line(self, nline_max, nchar_max):
        max_total = nline_max * nchar_max
        if max_total >= self.sent_len_min:
            L = int(np.random.randint(self.sent_len_min, max_total + 1))
        else:
            L = max_total
        L = max(L, 1)
        lines = self.get_lines(L, nchar_max)
        if not lines:
            return []
        return '\n'.join(lines)

    def sample_para(self, nline_max, nchar_max):
        nline = max(1, int(np.ceil(nline_max * sstat.beta.rvs(1.0, 1.0))))
        lines = self.get_lines(nline * nchar_max, nchar_max)
        if not lines:
            return []
        if np.random.rand() < self.center_para:
            lines = self.center_align(lines)
        return '\n'.join(lines)


class TextSource(object):
    """顶层：中英 7:3 路由。中文走 ChineseTextSource，英文走 EnglishTextSource。"""

    def __init__(self, min_nchar, data_dir='data'):
        self.min_nchar = min_nchar
        self.data_dir = data_dir

        # 中文字频表
        char_freq_path = osp.join(data_dir, 'models', 'char_freq_ch.cp')
        with open(char_freq_path, 'rb') as f:
            char_freq = cp.load(f)

        # 中文词频表：优先 data/models/word_freq_ch.cp，缺省回退仓库根的测试产物
        word_freq_path = osp.join(data_dir, "models", "word_freq_ch.cp")
        if not osp.exists(word_freq_path):
            word_freq_path = osp.join(osp.dirname(osp.abspath(__file__)),
                                      'tmp_test_full', 'word_freq.cp')
        with open(word_freq_path, 'rb') as f:
            wf = cp.load(f)
        word_freq = {w: c for w, c in wf.items()
                     if 2 <= len(w) <= 6 and all(is_hanzi(ch) for ch in w)}
        words_by_len = {}
        for w, c in word_freq.items():
            words_by_len.setdefault(len(w), []).append((w, c))

        # 中文源：wiki_zh JSONL 语料（惰性随机读文件）
        zh_files = []
        for dp, _, fs in os.walk(osp.join(data_dir, 'corpus', 'wiki_zh')):
            for f in fs:
                zh_files.append(osp.join(dp, f))
        self.chinese = ChineseTextSource(min_nchar, zh_files, char_freq, words_by_len)

        # 英文源
        self.english = EnglishTextSource(min_nchar, osp.join(data_dir, 'newsgroup.txt'))
        self.p_en = 0.30  # 中英 7:3，30% 概率采英文

    def sample(self, nline_max, nchar_max, kind=None):
        if np.random.rand() < self.p_en:
            return self.english.sample(nline_max, nchar_max, kind)
        return self.chinese.sample(nline_max, nchar_max, kind)
