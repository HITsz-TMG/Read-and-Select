
import random
import json
import re
import torch
from torch.utils.data import Dataset, DataLoader



class AttentionDataset(Dataset):
    def __init__(self, mentions, kb, tokenizer, args, istest):
        super(AttentionDataset, self).__init__()
        self.mentions = mentions
        self.kb = kb
        self.all_candidates = list(kb.keys())
        self.tokenizer = tokenizer
        self.info_token_num = args.info_token_num
        self.cand_num = args.cand_num
        self.max_ent_len = args.max_ent_len - 1 - args.info_token_num
        self.max_text_len = args.max_text_len
        self.max_length = args.max_len
        self.or_token = "[text]"

        self.istest = istest

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        if self.istest:
            return self.eval_dataset(index)
        else:
            return self.train_dataset(index)

    def pad_values(self, tokens, value, max_len):
        return (tokens + [value] * max_len)[:max_len]

    def train_dataset(self, index):
        data = self.mentions[index]
        info_token = [f"[info{i}]" for i in range(self.info_token_num)]
        text = data["text"]
        mention_data = data["mention_data"]
        splited_text = text.split(" ")
        mention_start, mention_end = splited_text.index("[E1]"), splited_text.index("[\E1]")
        mention = " ".join(splited_text[mention_start + 1:mention_end])

        mention_token = self.tokenizer.tokenize(mention)

        kb_id = mention_data["kb_id"]
        candidates = mention_data["candidates"][:self.cand_num]

        if kb_id not in candidates:
            candidates = candidates[:self.cand_num - 1] + [kb_id]
        if len(candidates) < self.cand_num:
            sim_neg = random.sample(list(self.all_candidates), k=self.cand_num - len(candidates))
            candidates += sim_neg

        assert len(candidates) == self.cand_num
        random.shuffle(candidates)
        labels = [kb_id == candidate for candidate in candidates]
        assert sum(labels) > 0

        max_half_text = (self.max_text_len - len(mention_token)) // 2 - 1

        text_tokens = self.tokenizer.tokenize(text)
        # pattern_tokens = self.tokenizer.tokenize(pattern)
        men_start = text_tokens.index("[E1]")
        men_end = text_tokens.index("[\E1]")
        text_tokens = text_tokens[max(0, men_start - max_half_text):men_end + max_half_text][:self.max_text_len - 2]
        text_tokens = [self.tokenizer.cls_token] + text_tokens + [self.tokenizer.sep_token]

        text_input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        mention_pos = [0, text_tokens.index("[E1]"), text_tokens.index("[\E1]")]
        text_input_ids = self.pad_values(text_input_ids, self.tokenizer.pad_token_id, self.max_text_len)
        text_attention_mask = [1] * len(text_tokens)
        text_attention_mask = self.pad_values(text_attention_mask, 0, self.max_text_len)
        assert self.tokenizer.sep_token_id in text_input_ids
        candiates_input_ids = []
        candidates_attention_masks = []

        for i in range(len(labels)):
            entity_id = candidates[i]

            entity_names = self.kb[entity_id]

            entity_text = remove_punctuation(entity_names["text"])

            entity_text = " ".join((entity_text.split(" "))[:self.max_ent_len])
            entity_text_tokens = self.tokenizer.tokenize(entity_text)

            entity_name_tokens = info_token + entity_text_tokens + [self.tokenizer.sep_token] + \
                                 text_tokens[1:]

            entity_name_ids = self.tokenizer.convert_tokens_to_ids(entity_name_tokens)
            entity_attention_mask = [1] * len(entity_name_tokens)

            entity_name_ids = self.pad_values(entity_name_ids, self.tokenizer.pad_token_id, self.max_length)
            entity_attention_mask = self.pad_values(entity_attention_mask, 0, self.max_length)
            candiates_input_ids.append(entity_name_ids)
            candidates_attention_masks.append(entity_attention_mask)
        labels = [[i] * self.info_token_num for i in labels]
        text_input_ids = torch.tensor(text_input_ids).long()
        text_attention_mask = torch.tensor(text_attention_mask).long()
        candidates_input_ids = torch.tensor(candiates_input_ids).long()
        candidates_attention_masks = torch.tensor(candidates_attention_masks).long()
        mention_pos = torch.tensor(mention_pos).long()
        labels = torch.tensor(labels).view(-1).float()
        return text_input_ids, text_attention_mask, candidates_input_ids, candidates_attention_masks, \
               mention_pos, labels

    def eval_dataset(self, index):
        data = self.mentions[index]
        info_token = [f"[info{i}]" for i in range(self.info_token_num)]
        text = data["text"]

        mention_data = data["mention_data"]
        kb_id = mention_data["kb_id"]
        splited_text = text.split(" ")
        mention_start, mention_end = splited_text.index("[E1]"), splited_text.index("[\E1]")
        mention = " ".join(splited_text[mention_start + 1:mention_end])
        mention_token = self.tokenizer.tokenize(mention)

        candidates = mention_data["candidates"]
        if not candidates:
            candidates = random.sample(self.all_candidates, k=self.cand_num)

        labels = [candidate == kb_id for candidate in candidates]

        assert sum(labels) > 0

        max_half_text = (self.max_text_len - len(mention_token)) // 2 - 1

        text_tokens = self.tokenizer.tokenize(text)
        # pattern_tokens = self.tokenizer.tokenize(pattern)
        men_start = text_tokens.index("[E1]")
        men_end = text_tokens.index("[\E1]")
        text_tokens = text_tokens[max(0, men_start - max_half_text):men_end + max_half_text][:self.max_text_len - 2]
        text_tokens = [self.tokenizer.cls_token] + text_tokens + [self.tokenizer.sep_token]
        text_input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        mention_pos = [0, text_tokens.index("[E1]"), text_tokens.index("[\E1]")]
        text_input_ids = self.pad_values(text_input_ids, self.tokenizer.pad_token_id, self.max_text_len)
        text_attention_mask = [1] * len(text_tokens)
        text_attention_mask = self.pad_values(text_attention_mask, 0, self.max_text_len)
        assert self.tokenizer.sep_token_id in text_input_ids
        candiates_input_ids = []
        candidates_attention_masks = []

        for i in range(len(labels)):
            entity_id = candidates[i]

            entity_names = self.kb[entity_id]
            title = entity_names["title"]
            entity_text = remove_punctuation(entity_names["text"])
            entity_text = " ".join(entity_text.split(" ")[:self.max_ent_len])
            entity_text_tokens = self.tokenizer.tokenize(entity_text)

            entity_name_tokens = info_token + entity_text_tokens + [self.tokenizer.sep_token] + \
                                 text_tokens[1:]

            entity_name_ids = self.tokenizer.convert_tokens_to_ids(entity_name_tokens)
            entity_attention_mask = [1] * len(entity_name_tokens)

            entity_name_ids = self.pad_values(entity_name_ids, self.tokenizer.pad_token_id, self.max_length)
            entity_attention_mask = self.pad_values(entity_attention_mask, 0, self.max_length)
            candiates_input_ids.append(entity_name_ids)
            candidates_attention_masks.append(entity_attention_mask)
        labels = [[i] * self.info_token_num for i in labels]
        text_input_ids = torch.tensor(text_input_ids).long()
        text_attention_mask = torch.tensor(text_attention_mask).long()
        candiates_input_ids = torch.tensor(candiates_input_ids).long()
        candidates_attention_masks = torch.tensor(candidates_attention_masks).long()
        mention_pos = torch.tensor(mention_pos).long()
        labels = torch.tensor(labels).view(-1).float()
        return text_input_ids, text_attention_mask, candiates_input_ids, candidates_attention_masks, \
               mention_pos, labels


def generate_samples(batch):
    input_ids, attention_masks, labels = [], [], []
    for b in batch:
        input_ids += b["input_ids"]
        attention_masks += b["attention_masks"]
        labels += b["labels"]
    input_ids = torch.tensor(input_ids).long()
    attention_masks = torch.tensor(attention_masks).long()
    labels = torch.tensor(labels).float()
    return input_ids, attention_masks, labels


def load_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def load_entities(data_path):
    with open(data_path, encoding="utf-8") as f:
        data = json.loads(f.read())
    return data


def make_single_loader(data_set, bsz, shuffle, coll_fn=None):
    if coll_fn is not None:
        loader = DataLoader(data_set, bsz, shuffle=shuffle, collate_fn=coll_fn)
    else:
        loader = DataLoader(data_set, bsz, shuffle=shuffle)
    return loader


def remove_punctuation(sentence):
    remove_chars = '[’!"#$%&\'()*+,-.:;<=>?@，。?★、…【】《》？“”‘’！\\^_`{|}~]+'
    result = re.sub(remove_chars, ' ', sentence)
    result = ' '.join(result.split())
    return result


def get_attention_mention_loader(samples, kb, tokenizer, shuffle, is_test, args):
    samples_set = AttentionDataset(samples, kb, tokenizer, args, is_test)
    if is_test:
        return make_single_loader(samples_set, 1, False)
    return make_single_loader(samples_set, args.batch, shuffle)

