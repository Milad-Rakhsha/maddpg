import pickle
import argparse
import numpy as np
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    return parser.parse_args()


if __name__ == '__main__':
    arglist = parse_args()
    rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
    print(rew_file_name)
    with open(rew_file_name, 'rb') as f:
        # rew = pickle.load(f)
        f_myfile = open(rew_file_name, 'rb')

    agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
    with open(agrew_file_name, 'rb') as f:
        agrew = pickle.load(f)

    sess = tf.compat.v1.Session()
    writer = tf.compat.v1.summary.FileWriter(arglist.plots_dir + arglist.exp_name + '_tensorboard')
    for i in range(len(agrew)):
        summary = tf.Summary(value=[tf.Summary.Value(tag="ag_rew", simple_value=agrew[i])])
        writer.add_summary(summary, global_step=i)
