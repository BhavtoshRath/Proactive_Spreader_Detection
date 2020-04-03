from __future__ import division
from __future__ import print_function

import numpy as np
import re
# from supervised_train import tweet_id

np.random.seed(123)


class NodeMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists  #BR: What is 'downsampled adjacency lists '?
    """

    def __init__(self, G,
                 placeholders, label_map, train_nodes, test_nodes, val_nodes, num_classes,
                 batch_size=100, max_degree=25,
                 **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes
        print('Sampler started...')
        self.adj, self.deg = self.construct_adj() # Random sampler
        # self.adj, self.deg = self.weighted_sampler(0) # 0:Topology sampler, 1:Activity-based
        print('Sampler ended...')
        self.test_adj = self.construct_test_adj()

        self.val_nodes = list(val_nodes)
        self.test_nodes = list(test_nodes)

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = train_nodes
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[n] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.nodes) * np.ones((len(self.nodes) + 1, self.max_degree))
        deg = np.zeros((len(self.nodes),))

        for nodeid in self.G.nodes():
            succ = [succ for succ in self.G.successors(nodeid)]
            pred = [pred for pred in self.G.predecessors(nodeid)]
            neighbors = np.append(succ, pred)
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.nodes) * np.ones((len(self.nodes) + 1, self.max_degree))
        for nodeid in self.G.nodes():
            succ = [succ for succ in self.G.successors(nodeid)]
            pred = [pred for pred in self.G.predecessors(nodeid)]
            neighbors = np.append(succ, pred)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj

    def weighted_sampler(self, wgt): # wgt = 0 (bel-weighted), = 1 (retweet-weighted)
        tweet_id = '' # Add tweet id here...
        id_dict = dict()
        id_set = set()
        mapping_file = tweet_id + '/mapping_' + tweet_id + '.txt'
        with open(mapping_file) as infile:
            for line in infile:
                l_spl = re.split(r'[,]', line.rstrip())
                id_dict[l_spl[1]] = int(l_spl[0])
                id_set.add(int(l_spl[0]))

        adj = len(self.nodes) * np.ones((len(self.nodes) + 1, self.max_degree))
        deg = np.zeros((len(self.nodes),))
        W = {}
        edge_w_file = tweet_id + '/wgt_' + tweet_id + '.txt'
        with open(edge_w_file) as infile:
            for line in infile:
                l_spl = re.split(',', line.rstrip())
                W[(id_dict[l_spl[0]], id_dict[l_spl[1]])] = float(l_spl[wgt + 2])
        for nodeid in self.G.nodes():
            succ_l = [succ for succ in self.G.successors(nodeid)]
            if len(succ_l) > 0:
                succ_w_l = [W[(nodeid, neighbor)] for neighbor in self.G.successors(nodeid)]
                succ_w_norm_l = [w / float(sum(succ_w_l)) for w in succ_w_l]
                succ_final = np.array(np.random.choice(succ_l, self.max_degree, p=succ_w_norm_l))
            else:
                succ_final = []

            pred_l = [pred for pred in self.G.predecessors(nodeid)]
            if len(pred_l) > 0:
                pred_w_l = [W[(neighbor, nodeid)] for neighbor in self.G.predecessors(nodeid)]
                pred_w_norm_l = [w / float(sum(pred_w_l)) for w in pred_w_l]
                pred_final = np.array(np.random.choice(pred_l, self.max_degree, p=pred_w_norm_l))
            else:
                pred_final = []

            neighbors_l = np.append(succ_l, pred_l)
            neighbors = np.append(succ_final, pred_final)
            deg[nodeid] = len(neighbors)
            adj[nodeid, :] = np.random.choice(neighbors, self.max_degree)

        return adj, deg

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1 = batch_nodes

        labels = np.vstack([self._make_label_vec(node) for node in batch1])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels, batch1

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1], ret_val[2]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num * size:min((iter_num + 1) * size,
                                                        len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num + 1) * size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num * size:min((iter_num + 1) * size,
                                                  len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num + 1) * size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0
