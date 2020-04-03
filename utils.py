from __future__ import print_function

import networkx as nx
import numpy as np
import random
import json
import math
import re

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]


WALK_LEN=5
N_WALKS=50


def load_data(tweet_id, normalize=True):
    # Read mapping file...
    id_dict = dict()
    id_set = set()
    mapping_file = tweet_id + '/mapping_' + tweet_id + '.txt'
    with open(mapping_file) as infile:
        for line in infile:
            l_spl = re.split(r'[,]', line.rstrip())
            id_dict[l_spl[1]] = int(l_spl[0])
            id_set.add(int(l_spl[0]))

    # Read spreaders file (exclude spreaders whose network could not be extracted)
    i = 0
    spreaderFile = tweet_id + '/retweets_' + tweet_id + '.txt'
    spreaderSet = set()
    with open(spreaderFile) as infile:
        for line in infile:
            l_spl = re.split(',', line.rstrip())
            try:
                if id_dict[l_spl[2]] in id_set:
                    spreaderSet.add(id_dict[l_spl[2]])
            except KeyError:
                i += 1
                continue

    print('Spreaders not in network: ', i)

    # Read network file
    i = 0
    network_file = tweet_id + '/clean_network_' + tweet_id + '.txt'
    G = nx.DiGraph()
    with open(network_file) as infile:
        for line in infile:
            l_spl = re.split(',', line.rstrip())
            try:
                G.add_edge(id_dict[l_spl[0]], id_dict[l_spl[1]])
            except KeyError:
                i += 1
                continue

    train_nodes = set()
    test_nodes = set()
    val_nodes = set()
    boundary_nodes = set()
    # Read nodes (train/test/val split)
    com = 0
    comFile = tweet_id + '/com_NBC_' + tweet_id + '.txt'
    with open(comFile) as infile:
        for line in infile:
            dict_NBC = json.loads(line)
            b_nodes = set([node for node in dict_NBC['core']])
            tr_nodes = set(random.sample(b_nodes, int(0.4 * (len(b_nodes)))))
            te_nodes = set(random.sample(b_nodes.difference(tr_nodes), int(0.3 * (len(b_nodes)))))
            v_nodes = b_nodes.difference(tr_nodes.union(te_nodes))
            boundary_nodes = boundary_nodes.union(b_nodes)
            train_nodes = train_nodes.union(tr_nodes)
            test_nodes = test_nodes.union(te_nodes)
            val_nodes = val_nodes.union(v_nodes)

    print('Train size: ', len(train_nodes), ',Infected train nodes: ', len(train_nodes.intersection(spreaderSet)))
    print('Val size: ', len(val_nodes), ',Infected val nodes: ', len(val_nodes.intersection(spreaderSet)))
    print('Test size: ', len(test_nodes), ',Infected test nodes: ', len(test_nodes.intersection(spreaderSet)))

    n = 1
    train_sp =train_nodes.intersection(spreaderSet)
    train_n_sp = set(random.sample(train_nodes.difference(train_sp), int(math.ceil(n*len(train_sp)))))
    val_sp =val_nodes.intersection(spreaderSet)
    val_n_sp = set(random.sample(val_nodes.difference(val_sp), int(math.ceil(n*len(val_sp)))))
    test_sp =test_nodes.intersection(spreaderSet)
    test_n_sp = set(random.sample(test_nodes.difference(test_sp), int(math.ceil(n*len(test_sp)))))

    featureFile = tweet_id + '/feat_' + tweet_id + '.txt'
    labels = {}
    feats = []
    with open(featureFile) as infile:
        for line in infile:
            l_spl = re.split(',', line.rstrip())
            feats.append(np.array(l_spl[1:], dtype="float64"))  # [1:3]: top-, [3:]: int-, [1:]: - top+int
            if id_dict[l_spl[0]] in spreaderSet:
                labels[id_dict[l_spl[0]]] = [1, 0]
            else:
                labels[id_dict[l_spl[0]]] = [0, 1]
    feats = np.array(feats)

    # Normalize features
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array(list(train_nodes))
        train_feats = feats[train_ids, :]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    return G, feats, labels, train_sp.union(train_n_sp), test_sp.union(test_n_sp), val_sp.union(val_n_sp)

