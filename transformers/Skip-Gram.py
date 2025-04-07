# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Skip-Gram.py
# Time       ：2025/4/3 20:05
# Author     ：Wentao Yang
"""
sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss", "Xiaoxin is Student", "Xiaoxue is student"]
words = ''.join(sentences).split()
words_list = list(set(words))
word_to_idx = {word: idx for idx, word in enumerate(words_list)}
idx_to_word = {idx: word for idx, word in enumerate(words_list)}
voc_size = len(words_list)

