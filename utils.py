import json
import re
import sys
import os
import random
import bisect
class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        if self.on or force:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()

def sample_range_excluding(n, k, excluding):
    skips = [j - i for i, j in enumerate(sorted(set(excluding)))]
    s = random.sample(range(n - len(skips)), k)
    return [i + bisect.bisect_right(skips, i) for i in s]

def read_data(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(x) for x in f]
    return data

def pad_values(data, token, max_len):
    return (data + [token for _ in range(max_len)])[:max_len]

def remove_punctuation(sentence):
    remove_chars = '[’!"#$%&\'()*+,-.:;<=>?@，。?★、…【】《》？“”‘’！\\^_`{|}~]+'
    result = re.sub(remove_chars, ' ', sentence)
    result = ' '.join(result.split())
    return result

