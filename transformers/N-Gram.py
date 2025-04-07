# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : N-Gram.py
# Time       ：2025/4/3 17:31
# Author     ：Wentao Yang
"""
# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : n-gram.py.py
# Time       ：2024/6/18 22:48
# Author     ：Wentao Yang
"""
import logging

log = logging.getLogger()
log.setLevel(logging.INFO)

# Add StreamHandler to log output to the console
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

corpus = ["我喜欢吃苹果",
          "我喜欢吃香蕉",
          "她喜欢吃葡萄",
          "他不喜欢吃香蕉",
          "他喜欢吃苹果",
          "她喜欢吃草莓"]


def tokenizer(sentence):
    return [char for char in sentence]


from collections import defaultdict, Counter

def ngram_probabilities(ngram_counts):
    ngram_probs = defaultdict(Counter)
    for prefix, counts in ngram_counts.items():
        total_count = sum(counts.values())
        for token, counts in counts.items():
            ngram_probs[prefix][token] = counts / total_count
    return ngram_probs

def count_ngrams(corpus, n):
    ngrams_count = defaultdict(Counter)
    for text in corpus:
        tokens = tokenizer(text)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            prefix = ngram[:-1]
            token = ngram[-1]
            ngrams_count[prefix][token] += 1
    return ngrams_count


def generate_next_token(prefix, ngram_probs):
    global log
    log.info("Generate one token.")
    if not prefix in ngram_probs:
        return None
    next_token_probs = ngram_probs[prefix]
    next_token = max(next_token_probs, key=next_token_probs.get)
    return next_token


def generate_text(prefix, n, ngram_probs, max_length=6):
    tokens = list(prefix)
    if len(tokens) >= max_length:
        return prefix
    for _ in range(max_length - len(prefix)):
        next_token = generate_next_token(tuple(tokens[-(n - 1):]), ngram_probs)
        if not next_token:
            break
        tokens.append(next_token)
    return "".join(tokens)

n = 3
biggram_count = count_ngrams(corpus, n)
print("Bigram 词频：")
for prefix, counts in biggram_count.items():
    print("{}：{}".format("".join(prefix), dict(counts)), flush=True)
probs = ngram_probabilities(biggram_count)
print("Bigram 概率： ")
for prefix, prob in probs.items():
    print("{}：{}".format("".join(prefix), dict(prob)), flush=True)
print("Generate: ")
x = '我喜'
gen = generate_text(x, n, dict(probs))
print(gen)
