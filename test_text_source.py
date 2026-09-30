# -*- coding: utf-8 -*-
"""
测试 text_source.py 的 char_freq / word_freq 采样（CHAR / WORD）。

用法（必须用 .venv，因为它有 numpy/scipy/h5py）：
    .venv/Scripts/python.exe test_text_source.py

数据：直接读全量跑出来的 tmp_test_full/char_freq.cp 与 word_freq.cp，
     不依赖 TextSource 的 data/model/ 路径（那套还没落位）。
"""
import os
import sys
import pickle as cp
from collections import Counter

import numpy as np

import text_source as ts

REPO = os.path.dirname(os.path.abspath(__file__))
CHAR_FREQ_PATH = os.path.join(REPO, 'data', 'models', 'char_freq_ch.cp')
WORD_FREQ_PATH = os.path.join(REPO, 'data', 'models', 'word_freq_ch.cp')


def load_data():
    with open(CHAR_FREQ_PATH, 'rb') as f:
        char_freq = cp.load(f)
    with open(WORD_FREQ_PATH, 'rb') as f:
        wf = cp.load(f)
    # 与 TextSource 里一致：只保留 2~6 字纯汉字词
    word_freq = {w: c for w, c in wf.items()
                 if 2 <= len(w) <= 6 and all(ts.is_hanzi(ch) for ch in w)}
    words_by_len = {}
    for w, c in word_freq.items():
        words_by_len.setdefault(len(w), []).append((w, c))
    return char_freq, word_freq, words_by_len


def test_precompute(src):
    assert abs(src._char_w.sum() - 1) < 1e-6, 'char 频率权重未归一化'
    assert abs(src._char_inv_w.sum() - 1) < 1e-6, 'char 逆频次权重未归一化'
    for L, w in src._word_w.items():
        assert abs(w.sum() - 1) < 1e-6, f'L={L} 词权重未归一化'
    print(f'[OK] 预计算权重归一化（字符 {len(src._char_keys)} 个，'
          f'词长度 {sorted(src._word_keys)}，各长度词数 '
          f'{ {L: len(ks) for L, ks in src._word_keys.items()} }）')


def test_char(src, char_freq, n=20000):
    sampled = [src.sample_char(1, 10) for _ in range(n)]
    assert all(c in char_freq for c in sampled), '采到字符表外的字符'
    cnt = Counter(sampled)
    top = max(char_freq, key=char_freq.get)   # 最高频字符
    assert cnt[top] > 0, '最高频字符从未出现'
    print(f'[OK] CHAR 采样 {n} 次：覆盖 {len(cnt)} 个不同字符，'
          f'最高频 {top!r} 出现 {cnt[top]} 次')


def test_word(src, word_freq, n=20000):
    sampled = [src.sample_word(1, 100) for _ in range(n)]
    assert all(w in word_freq for w in sampled), '采到词表外的词'
    lens = Counter(len(w) for w in sampled)
    # 词长分布应近似 p_word_len {2:.30,3:.30,4:.20,5:.10,6:.10}
    for L, p in src.p_word_len.items():
        frac = lens.get(L, 0) / n
        assert abs(frac - p) < 0.04, f'L={L} 实测 {frac:.3f} 偏离期望 {p:.2f}'
    print(f'[OK] WORD 采样 {n} 次：词长分布 {dict(sorted(lens.items()))}')


def demo(src):
    print('\n示例采样（人工观察）：')
    print('  字: ', ''.join(src.sample_char(1, 10) for _ in range(20)))
    print('  词: ', ' '.join(src.sample_word(1, 100) for _ in range(12)))


def main():
    # 控制台统一 UTF-8（gbk 终端下生僻字不崩溃；UTF-8 终端正常显示）
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    if not os.path.exists(CHAR_FREQ_PATH) or not os.path.exists(WORD_FREQ_PATH):
        raise SystemExit(f'缺少数据文件：\n  {CHAR_FREQ_PATH}\n  {WORD_FREQ_PATH}\n'
                         f'请先用 build_vocab.py 跑全量（--out-dir ./tmp_test_full）')
    char_freq, word_freq, words_by_len = load_data()
    src = ts.ChineseTextSource(3, [], char_freq, words_by_len)

    test_precompute(src)
    test_char(src, char_freq)
    test_word(src, word_freq)
    demo(src)
    print('\n=== 全部测试通过 ===')


if __name__ == '__main__':
    main()
