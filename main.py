import torch
import os,sys
import pickle
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import nltk
import matplotlib.pyplot as plt
from copy import deepcopy
from tensorboardX import SummaryWriter
import json
from nltk.corpus import wordnet, stopwords
from datetime import datetime
from os.path import join, exists
from model.common import evaluate, count_parameters, make_infinite
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from model import config
import warnings
from transformers import AutoTokenizer
import math
import dgl
from model.EmpSOI import EmpSOI
import re
warnings.filterwarnings("ignore")

DATA_FILES = lambda data_dir: {
    "train": [
        f"{data_dir}/sys_dialog_texts.train.npy",
        f"{data_dir}/sys_target_texts.train.npy",
        f"{data_dir}/sys_emotion_texts.train.npy",
        f"{data_dir}/sys_situation_texts.train.npy",
    ],
    "dev": [
        f"{data_dir}/sys_dialog_texts.dev.npy",
        f"{data_dir}/sys_target_texts.dev.npy",
        f"{data_dir}/sys_emotion_texts.dev.npy",
        f"{data_dir}/sys_situation_texts.dev.npy",
    ],
    "test": [
        f"{data_dir}/sys_dialog_texts.test.npy",
        f"{data_dir}/sys_target_texts.test.npy",
        f"{data_dir}/sys_emotion_texts.test.npy",
        f"{data_dir}/sys_situation_texts.test.npy",
    ],
}

word_pairs = {
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "you'd": "you would",
    "you're": "you are",
    "you'll": "you will",
    "i'm": "i am",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "couldn't": "could not",
    "i've": "i have",
    "we've": "we have",
    "can't": "cannot",
    "i'd": "i would",
    "i'd": "i would",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "there's": "there is",
    "there're": "there are",
}
emo_map = {
    "surprised": 0,
    "excited": 1,
    "annoyed": 2,
    "proud": 3,
    "angry": 4,
    "sad": 5,
    "grateful": 6,
    "lonely": 7,
    "impressed": 8,
    "afraid": 9,
    "disgusted": 10,
    "confident": 11,
    "terrified": 12,
    "hopeful": 13,
    "anxious": 14,
    "disappointed": 15,
    "joyful": 16,
    "prepared": 17,
    "guilty": 18,
    "furious": 19,
    "nostalgic": 20,
    "jealous": 21,
    "anticipating": 22,
    "embarrassed": 23,
    "content": 24,
    "devastated": 25,
    "sentimental": 26,
    "caring": 27,
    "trusting": 28,
    "ashamed": 29,
    "apprehensive": 30,
    "faithful": 31,
}

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6

class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

vocab=Lang(
                {
                    0: "UNK",
                    1: "PAD",
                    2: "EOS",
                    3: "SOS",
                    4: "OTH",
                    5: "SEL",
                    6: "CLS",
                }
            )

def set_seed():
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cs_refine(data):
    temp_cs= []
    num = 0
    for idx,item in enumerate(data['context']):
        temp = []
        for i in range(num,num+len(item)):
            temp.append(data['utt_cs'][i])
        temp_cs.append(temp)
        num = num+len(item)
    data['utt_cs'] = temp_cs
    return data

class Dataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()  

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        conv_len = len(item["context_text"])
        item["situation_text"] = self.data["situation"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]

        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"],item["x_cls_index"] ,item["other_mask"], item["self_mask"] = self.preprocess(item["context_text"])#mask=dialogue state
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )
        # (
        #     item["emotion_context"],
        #     item["emotion_context_mask"],
        # ) = self.preprocess(item["emotion_context"])

        item["x_intent"] = torch.empty(0)
        item["x_attr"] = torch.empty(0)
        item["x_want"] = torch.empty(0)
        item["x_effect"] = torch.empty(0)
        item["x_need"] = torch.empty(0)
        item["x_react"] = torch.empty(0)
        for i in range(conv_len):
            item["cs_text"] = self.data["utt_cs"][index]
            item["x_intent_txt"] = item["cs_text"][i][0]
            item["x_attr_txt"] = item["cs_text"][i][1]
            item["x_want_txt"] = item["cs_text"][i][2]
            item["x_effect_txt"] = item["cs_text"][i][3]
            item["x_need_txt"] = item["cs_text"][i][4]
            item["x_react_txt"] = item["cs_text"][i][5]

            item["x_intent"] = torch.cat([item["x_intent"],self.preprocess(item["x_intent_txt"], cs=True)],dim=0)
            item["x_attr"] = torch.cat([item["x_attr"],self.preprocess(item["x_attr_txt"], cs=True, no_mark=True)],dim=0)
            item["x_want"] = torch.cat([item["x_want"],self.preprocess(item["x_want_txt"], cs=True)],dim=0)
            item["x_effect"] = torch.cat([item["x_effect"],self.preprocess(item["x_effect_txt"], cs=True)],dim=0)
            item["x_need"] = torch.cat([item["x_need"],self.preprocess(item["x_need_txt"], cs=True)],dim=0)
            item["x_react"] = torch.cat([item["x_react"],self.preprocess(item["x_react_txt"], cs=True)],dim=0)

        item["last_user_cls_index"] = self.preprocess(item["context"], select=True)
        return item
        
    def preprocess(self, arr, anw=False, cs=None, emo=False, select=False, no_mark=False):
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else UNK_idx
                for word in arr
            ] + [EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:
            if(no_mark==True):
                sequence = []
            else:
                sequence = [config.CLS_idx]
            for sent in arr:
                sequence += [
                    self.vocab.word2index[word]
                    for word in sent
                    if word in self.vocab.word2index and word not in ["to", "none"]
                ]

            return torch.LongTensor(sequence)
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)
        elif select:
            last_user_cls_index=0
            for i in range(arr.size(0)):
                if(arr[i]==8):
                    last_user_cls_index=i
            return last_user_cls_index
        else:
            x_dial = []
            x_mask = []
            x_cls_index = []
            self_mask = np.zeros((6*len(arr)+4, 6*len(arr)+4)) # utt + csk + 2 perspective_taking+2 states
            other_mask = np.zeros((6*len(arr)+4, 6*len(arr)+4))
            self_mask[-1][-1] = 1
            self_mask[-2][-2] = 1
            other_mask[-3][-3] = 1
            other_mask[-4][-4] = 1
            self_mask[-1][-2] = 1
            self_mask[-2][-1] = 1
            other_mask[-3][-4] = 1 
            other_mask[-4][-3] = 1
            cs_index = len(arr)
            for i, sentence in enumerate(arr):
                x_dial += [config.CLS_idx] + [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in sentence
                ]
                x_cls_index += [len(x_dial) - len(sentence) - 1]
                j = i
                while j >= 0:
                    self_mask[i][j] = 1 # utt-connection
                    self_mask[j][i] = 1
                    other_mask[i][j] = 1 # utt-connection
                    other_mask[j][i] = 1
                    self_mask[-2][j] = 1 # self_perspective_taking connection with utt
                    self_mask[j][-2] = 1
                    self_mask[-1][j] = 1 # self_high_emo_state connection with utt
                    self_mask[j][-1] = 1
                    other_mask[-3][j] = 1 # other_high_emo_state connection with utt
                    other_mask[j][-3] = 1
                    other_mask[-4][j] = 1 # other_perspective_taking connection with utt
                    other_mask[j][-4] = 1
                    j -= 1
                for j in range(5):
                    self_mask[cs_index+j][cs_index+j] = 1 # csk self-loop
                    other_mask[cs_index+j][cs_index+j] = 1
                    self_mask[i][cs_index+j] = 1 # csk-utt connection
                    self_mask[cs_index+j][i] = 1
                    other_mask[i][cs_index+j] = 1 # csk-utt connection
                    other_mask[cs_index+j][i] = 1
                    self_mask[-2][cs_index+j] = 1 # csk-self_perspective_taking connection
                    self_mask[cs_index+j][-2] = 1
                    other_mask[-4][cs_index+j] = 1 # csk-other_perspective_taking connection
                    other_mask[cs_index+j][-4] = 1
                    self_mask[-1][cs_index+j] = 1 # csk-self_high_emo_state connection
                    self_mask[cs_index+j][-1] = 1
                    other_mask[-3][cs_index+j] = 1 # csk-other_high_emo_state connection
                    other_mask[cs_index+j][-3] = 1
                cs_index = cs_index+5
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence)+1)]


            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask),torch.LongTensor(x_cls_index) , torch.FloatTensor(other_mask), torch.FloatTensor(self_mask)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]
    



def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths
    
    def pad_matrix(matrix, padding_index=0):
        max_len = max(i.size(0) for i in matrix)
        batch_matrix = []
        for item in matrix:
            item = item.numpy()
            batch_matrix.append(np.pad(item, ((0, max_len-len(item)), (0, max_len-len(item))), 'constant', constant_values=(padding_index, padding_index)))
        return torch.FloatTensor(batch_matrix)
    
    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    x_cls_index = pad_sequence(item_info["x_cls_index"], True)
    other_mask = pad_matrix(item_info["other_mask"])
    self_mask = pad_matrix(item_info["self_mask"])
    # emotion_batch, emotion_lengths = merge(item_info["emotion_context"])

    ## Target
    target_batch, target_lengths = merge(item_info["target"])

    input_batch = input_batch.to(device)
    mask_input = mask_input.to(device)
    x_cls_index = x_cls_index.to(config.device)
    target_batch = target_batch.to(device)
    other_mask=other_mask.to(device)
    self_mask=self_mask.to(device)
    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["x_cls_index"] = x_cls_index
    d["target_lengths"] = torch.LongTensor(target_lengths)
    # d["emotion_context_batch"] = emotion_batch.to(device)

    ##program
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    ##text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]
    d["last_user_cls_index"] = item_info["last_user_cls_index"]
    d["other_mask"] = other_mask
    d["self_mask"] = self_mask

    relations = ["x_intent", "x_attr", "x_want", "x_effect", "x_need","x_react"]
    # for r in relations:
    #     pad_batch, _ = merge(item_info[r])
    #     pad_batch = pad_batch.to(device)
    #     d[r] = pad_batch
    #     d[f"{r}_txt"] = item_info[f"{r}_txt"]
    for r in relations:
        pad_batch = pad_sequence(item_info[r], batch_first=True, padding_value=config.PAD_idx).to(config.device)
        d[r] = pad_batch
    return d

def train(model, train_set, dev_set):
    check_iter = 2000
    try:
        model.train()
        best_ppl = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(train_set)
        for n_iter in tqdm(range(100000)):
            loss, ppl, bce, acc,_, _ = model.train_one_batch(next(data_iter), n_iter)
            writer.add_scalars("loss", {"loss_train": loss}, n_iter)
            writer.add_scalars("ppl", {"ppl_train": ppl}, n_iter)
            writer.add_scalars("bce", {"bce_train": bce}, n_iter)
            writer.add_scalars("accuracy", {"acc_train": acc}, n_iter)
            if (n_iter + 1) % check_iter == 0:
                model.eval()
                model.epoch = n_iter
                loss_val, ppl_val, bce_val, acc_val, _ = evaluate(
                    model, dev_set, ty="valid", max_dec_step=50
                )
                print("lr", {"learning_rata": model.optimizer._rate}, n_iter)
                writer.add_scalars("loss", {"loss_valid": loss_val}, n_iter)
                writer.add_scalars("ppl", {"ppl_valid": ppl_val}, n_iter)
                writer.add_scalars("bce", {"bce_valid": bce_val}, n_iter)
                writer.add_scalars("accuracy", {"acc_train": acc_val}, n_iter)
                model.train()
                if n_iter < 8000:
                    continue
                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    model.save_model(best_ppl, n_iter)
                    torch.save(model.state_dict(),"empsoi")
                    weights_best = deepcopy(model.state_dict())
                else:
                    patient += 1
                if patient > 1:
                    break

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(best_ppl, n_iter)
        weights_best = deepcopy(model.state_dict())

    return weights_best


def test(model, test_set):
    model.eval()
    model.is_eval = True
    loss_test, ppl_test, bce_test, acc_test, results = evaluate(
        model, test_set, ty="test", max_dec_step=50,
    )
    file_summary = config.save_path + "/EmpSOI.txt"
    with open(file_summary, "w",encoding='utf-8') as f:
        f.write("EVAL\tLoss\tPPL\tAccuracy\n")
        f.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                loss_test, ppl_test, bce_test, acc_test
            )
        )
        for r in results:
            f.write(r)

if __name__ == "__main__":
    # set_seed()
    cache_file = "dataset_preproc1.2.p"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    
    data_tra = cs_refine(data_tra)
    data_val = cs_refine(data_val)
    data_tst = cs_refine(data_tst)

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(data_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_valid = Dataset(data_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(data_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    is_eval = config.test
    model = EmpSOI(
                vocab,
                emo_number=32,
                is_eval=is_eval,
                model_file_path=config.model_path if is_eval else None,
            )

    model.to(device)
    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb ):
            xavier_uniform_(p)
    # weights_dict = torch.load("empsoi", map_location=device)
    # model.load_state_dict(weights_dict,strict=False)
    
    if config.test:
        test(model, data_loader_tst)
    else:
        weights_best = train(model, data_loader_tra, data_loader_val)
        model.epoch = 1
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        test(model, data_loader_tst)