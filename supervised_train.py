from __future__ import division
from __future__ import print_function

import os
import csv
import sys
import json
import time
import datetime
from collections import defaultdict
import tensorflow as tf
import numpy as np
from sklearn import metrics

from supervised_models import SupervisedGraphsage
from models import SAGEInfo
from minibatch import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from utils import load_data

tweet_id = sys.argv[1]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
current_time = str(datetime.datetime.now())
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', './example_data/ppi', 'prefix identifying training data. must be specified.')
# left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 50*100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 50*100, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 0, 'Set to 0 as network contains only immediate fol/frnd')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', False, 'Whether to use random context  or direct edges')
flags.DEFINE_integer('batch_size', 8, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 2, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 8, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 1, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 5, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

results_folder = tweet_id+'_results'
if os.path.isdir(results_folder) is False:
    os.mkdir(results_folder)


def eval(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

    f1_score = metrics.f1_score(y_true, y_pred)
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return acc, prec, rec, f1_score, conf_mat, fpr, tpr, thresholds


# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    feed_dict_val, labels, batch = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                        feed_dict=feed_dict_val)

    with open('labels_and_preds.txt', 'a') as f:
        writer = csv.writer(f)
        for i in range(len(batch)):
            l = []
            l.append(batch[i])
            l.extend(labels[i])
            l.extend(node_outs_val[0][i])
            writer.writerow(l)

    acc, prec, rec, f1_score, conf_mat, fpr, tpr, thresholds = eval(labels, node_outs_val[0])
    return node_outs_val[1], acc, prec, rec, f1_score, conf_mat, fpr, tpr, thresholds


def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    losses = []
    preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                         feed_dict=feed_dict_val)
        preds.append(node_outs_val[0])
        labels.append(batch_labels)
        losses.append(node_outs_val[1])
        iter_num += 1
    preds = np.vstack(preds)
    labels = np.vstack(labels)

    if test is True:
        with open(tweet_id + '_results/' + current_time + '_labels_and_preds_test.txt', 'w') as f:
            writer = csv.writer(f)
            for i in range(len(labels)):
                l = []
                l.append(minibatch_iter.test_nodes[i])
                l.extend(labels[i])
                l.extend(preds[i])
                writer.writerow(l)

        with open(tweet_id + '_results/' + 'sp.txt', 'a') as f:
            d = defaultdict(list)
            d['time'] = current_time
            for i in range(len(labels)):
                if labels[i][0] == 1 and preds[i][0] > preds[i][1]:
                    d['sp_false'].append(minibatch_iter.test_nodes[i])
                if labels[i][0] == 1 and preds[i][0] < preds[i][1]:
                    d['sp_true'].append(minibatch_iter.test_nodes[i])
            dict_str = json.dumps(dict(d))
            f.write(dict_str + '\n')

    acc, prec, rec, f1_score, conf_mat, fpr, tpr, thresholds = eval(labels, preds)
    return np.mean(losses), acc, prec, rec, f1_score, conf_mat


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.compat.v1.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data):

    G = train_data[0]
    features = train_data[1]
    labels = train_data[2]
    train_nodes = train_data[3]
    test_nodes = train_data[4]
    val_nodes = train_data[5]
    num_classes = 2

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G,
            placeholders,
            labels,
            train_nodes,
            test_nodes,
            val_nodes,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                           SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

    model = SupervisedGraphsage(num_classes, placeholders,
                                features,
                                adj_info,
                                minibatch.deg,
                                layer_infos,
                                model_size=FLAGS.model_size,
                                sigmoid_loss=FLAGS.sigmoid,
                                identity_dim=FLAGS.identity_dim,
                                logging=True)

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(results_folder, sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    # Train model
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels, batch = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    val_cost,  acc, prec, rec, f1_score, conf_mat = incremental_evaluate(sess, model, minibatch,
                                                                                      FLAGS.batch_size)
                else:
                    val_cost,  acc, prec, rec, f1_score, conf_mat, fpr, tpr, thresholds = evaluate(sess, model, minibatch,
                                                                             FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_acc, train_prec, train_rec, train_f1_score, train_conf_mat, fpr, tpr, thresholds = eval(labels, outs[-1])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_accuracy=", "{:.5f}".format(train_acc),
                      "train_precision=", "{:.5f}".format(train_prec),
                      "train_recall=", "{:.5f}".format(train_rec),
                      "train_f1_score=", "{:.5f}".format(train_f1_score))
                with open(results_folder + "/train_stats.txt", "a") as fp:
                    fp.write("Iter:{:d} loss={:.5f} acc={:.5f} prec={:.5f} rec={:.5f} f1={:.5f} tp={:d} fp={:d} fn={:d} tn={:d}\n"
                            .format(iter, train_cost, train_acc, train_prec, train_rec, train_f1_score,
                            train_conf_mat[0][0], train_conf_mat[0][1], train_conf_mat[1][0], train_conf_mat[1][1]))

            iter += 1
            total_steps += 1

            if total_steps > int(FLAGS.max_total_steps):
                break

        if total_steps > int(FLAGS.max_total_steps):
            break

    print("Optimization Finished!")
    sess.run(val_adj_info.op)
    val_cost, val_acc, val_prec, val_rec, val_f1_score, val_conf_mat = incremental_evaluate(sess, model, minibatch,
                                                                        FLAGS.batch_size)
    print("Full validation stats:",
          "val_cost=", "{:.5f}".format(val_cost),
          "val_acc=", "{:.5f}".format(val_acc),
          "val_prec=", "{:.5f}".format(val_prec),
          "val_rec=", "{:.5f}".format(val_rec),
          "val_f1_score=", "{:.5f}".format(val_f1_score),
          "val_conf_mat=", val_conf_mat)
    with open(results_folder + "/val_stats.txt", "a") as fp:
        fp.write("loss={:.5f} acc={:.5f} prec={:.5f} rec={:.5f} f1={:.5f} tp={:d} fp={:d} fn={:d} tn={:d} time=={:s}\n".
                 format(val_cost, val_acc, val_prec, val_rec, val_f1_score,
                        val_conf_mat[0][0], val_conf_mat[0][1], val_conf_mat[1][0], val_conf_mat[1][1], current_time))
    print("Writing test set stats to file")
    test_cost, test_acc, test_prec, test_rec, test_f1_score, test_conf_mat = incremental_evaluate(sess, model, minibatch,
                                                                                FLAGS.batch_size, test=True)
    print("Full test stats:",
          "test_cost=", "{:.5f}".format(test_cost),
          "test_acc=", "{:.5f}".format(test_acc),
          "test_prec=", "{:.5f}".format(test_prec),
          "test_rec=", "{:.5f}".format(test_rec),
          "test_f1_score=", "{:.5f}".format(test_f1_score),
          "test_conf_mat=", test_conf_mat)
    with open(results_folder + "/test_stats.txt", "a") as fp:
        fp.write("loss={:.5f} acc={:.5f} prec={:.5f} rec={:.5f} f1={:.5f} tp={:d} fp={:d} fn={:d} tn={:d} time=={:s}\n"
                .format(test_cost, test_acc, test_prec, test_rec, test_f1_score,
                        test_conf_mat[0][0], test_conf_mat[0][1], test_conf_mat[1][0], test_conf_mat[1][1],current_time))


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(tweet_id)
    print("Done loading training data..")
    train(train_data)


if __name__ == '__main__':
    tf.app.run()
