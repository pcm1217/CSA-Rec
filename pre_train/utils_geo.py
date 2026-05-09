import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import os
from datetime import datetime
from pytz import timezone
from torch.utils.data import Dataset
import pickle


itemid2geohash = None
geohash_prefix2items = None

item2geo_path = ''
prefix2items_path = ''

with open(item2geo_path, 'rb') as f:
    itemid2geohash = pickle.load(f)

with open(prefix2items_path, 'rb') as f:
    geohash_prefix2items = pickle.load(f)


def geo_neg_sample(visited_pois, last_poi, poinum):

    visited_set = set(visited_pois)
    visited_list = list(visited_pois)

    if last_poi in itemid2geohash:
        prefix = itemid2geohash[last_poi][:4]
        candidates = geohash_prefix2items.get(prefix, set()) - visited_set
        if candidates:
            return random.choice(list(candidates))

    for anchor_poi in reversed(visited_list[:-1]):
        if anchor_poi in itemid2geohash:
            prefix = itemid2geohash[anchor_poi][:4]
            candidates = geohash_prefix2items.get(prefix, set()) - visited_set
            if candidates:
                return random.choice(list(candidates))

    for _ in range(10):
        t = np.random.randint(1, poinum + 1)
        if t not in visited_set:
            return t

    return np.random.randint(1, poinum + 1)


def sample_function(user_train, usernum, poinum, geonum, batch_size, maxlen, result_queue, SEED):

    def sample():
        user = np.random.randint(1, usernum + 1)

        while len(user_train[user][0]) <= 1:
            user = np.random.randint(1, usernum + 1)

        poi_seq, geo_seq, time_seq = user_train[user]

        seq = np.zeros([maxlen], dtype=np.int32)
        geo = np.zeros([maxlen], dtype=np.int32)
        time_ = np.zeros([maxlen], dtype=np.float32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = poi_seq[-1]
        idx = maxlen - 1
        visited = set(poi_seq)
        last = poi_seq[-1]

        for i in reversed(range(len(poi_seq) - 1)):
            seq[idx] = poi_seq[i]
            geo[idx] = geo_seq[i]
            time_[idx] = time_seq[i]
            pos[idx] = nxt

            if nxt != 0:
                neg[idx] = geo_neg_sample(visited, last, poinum)

            nxt = poi_seq[i]
            idx -= 1
            if idx == -1:
                break

        return (user, seq, geo, time_, pos, neg)

    np.random.seed(SEED)

    while True:
        one_batch = [sample() for _ in range(batch_size)]
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, poinum, geonum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        for _ in range(n_workers):
            p = Process(target=sample_function,
                        args=(User, usernum, poinum, geonum,
                              batch_size, maxlen,
                              self.result_queue,
                              np.random.randint(2e9)))
            p.daemon = True
            p.start()
            self.processors.append(p)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


class SeqDataset(Dataset):
    def __init__(self, user_train, num_user, num_poi, max_len):
        self.user_train = user_train
        self.num_user = num_user
        self.num_poi = num_poi
        self.max_len = max_len

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user_id = idx + 1
        poi_seq, geo_seq_raw, time_seq_raw = self.user_train[user_id]

        seq = np.zeros([self.max_len], dtype=np.int32)
        geo_seq = np.zeros([self.max_len], dtype=np.int32)
        time_seq = np.zeros([self.max_len], dtype=np.float32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        nxt = poi_seq[-1]
        idx_ptr = self.max_len - 1
        visited = set(poi_seq)
        last = poi_seq[-1]

        for i in reversed(range(len(poi_seq) - 1)):
            seq[idx_ptr] = poi_seq[i]
            geo_seq[idx_ptr] = geo_seq_raw[i]
            time_seq[idx_ptr] = time_seq_raw[i]
            pos[idx_ptr] = nxt

            if nxt != 0:
                neg[idx_ptr] = geo_neg_sample(visited, last, self.num_poi)

            nxt = poi_seq[i]
            idx_ptr -= 1
            if idx_ptr == -1:
                break

        return user_id, seq, geo_seq, time_seq, pos, neg


def data_partition(fname, path=None):
    usernum, poinum, geonum = 0, 0, 0

    User = defaultdict(list)
    UserGeo = defaultdict(list)
    UserTime = defaultdict(list)

    user_train, user_valid, user_test = {}, {}, {}

    if path is None:
        f = open('' % fname, 'r')
    else:
        f = open(path, 'r')

    for line in f:
        u, i, time, geo, geo_id = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        g = int(geo_id)

        usernum = max(u, usernum)
        poinum = max(i, poinum)
        geonum = max(g, geonum)

        User[u].append(i)
        UserGeo[u].append(g)
        UserTime[u].append(time)

    for user in User:
        n = len(User[user])
        if n < 3:
            user_train[user] = (User[user], UserGeo[user], UserTime[user])
            user_valid[user] = ([], [], [])
            user_test[user] = ([], [], [])
        else:
            user_train[user] = (User[user][:-2], UserGeo[user][:-2], UserTime[user][:-2])
            user_valid[user] = ([User[user][-2]], [UserGeo[user][-2]], [UserTime[user][-2]])
            user_test[user] = ([User[user][-1]], [UserGeo[user][-1]], [UserTime[user][-1]])

    return [user_train, user_valid, user_test, usernum, poinum, geonum + 1]


def evaluate(model, dataset, args):
    [train, valid, test, usernum, poinum, geonum] = copy.deepcopy(dataset)

    NDCG, HT, valid_user = 0.0, 0.0, 0.0

    users = random.sample(range(1, usernum + 1), 10000) if usernum > 10000 else range(1, usernum + 1)

    for u in users:
        if len(train[u][0]) < 1 or len(test[u][0]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        geo_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.float32)

        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        geo_seq[idx] = valid[u][1][0]
        time_seq[idx] = valid[u][2][0]

        idx -= 1
        for poi, geo, time in zip(reversed(train[u][0]), reversed(train[u][1]), reversed(train[u][2])):
            seq[idx] = poi
            geo_seq[idx] = geo
            time_seq[idx] = time
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u][0])
        rated.add(0)
        last = train[u][0][-1]

        poi_idx = [test[u][0][0]]
        for _ in range(19):
            poi_idx.append(geo_neg_sample(rated, last, poinum))

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [geo_seq], [time_seq], poi_idx]])
        rank = predictions[0].argsort().argsort()[0].item()

        valid_user += 1
        if rank < 1:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user