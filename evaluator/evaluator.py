import os
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'distilled_results')
sys.path.append(ROOT_DIR)

from evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import argparse
import logging
import torch

class Evaluator:

    def __init__(self):
        pass

    def load_data(self, data_dir, data_file, args):
        self.dst_train, self.dst_test = EvaluatorUtils.get_dataset(args)
        # return immediately if we use whole dataset to train
        if args.method == 'whole':
            return
        data_loader = EvaluatorUtils.get_data_loader(args.method)
        self.train_images, self.train_labels = data_loader.load_data(data_dir, args.dataset, args.ipc, data_file)

    
    def evaluate(self, args):
        '''
        do the acutual evaluation
        '''
        model = NetworkUtils.create_network(args)
        # using whole dataset to train
        if args.method == 'whole':
            _, _, test_accuracy = EvaluatorUtils.evaluate_synset_dataset(0, model, self.dst_train, self.dst_test, args, logging)
        else:
            _, _, test_accuracy = EvaluatorUtils.evaluate_synset(0, model, self.train_images, self.train_labels, self.dst_test, args, logging)
        # print("model_name: %s, accuracy: %.4f"%(args.model, test_accuracy))
        return test_accuracy

def prepare_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--gpu', type=str, default='auto', help='gpu ID(s)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--method', type=str, default='dc', help='which dc method to evaluate')
    parser.add_argument('--model', type=str, default='convnet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--dsa', action="store_true", default=False, help='dsa')
    parser.add_argument('--aug', type=str, default='', help='augmentation method')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--normalize_data', action="store_true", default=True, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--eval_gap', type=int, default=1001, help='run evaluation every eval_gap epochs.')
    parser.add_argument('--optimizer', type=str, default="sgd", help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(EvaluatorUtils.pick_gpu_lowest_memory()) if args.gpu == 'auto' else args.gpu
    args.dc_aug_param = EvaluatorUtils.get_daparam(args.dataset, args.model, '', args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
    # setup DSA parameters.
    if args.dsa:
        args.dsa_param = EvaluatorUtils.ParamDiffAug()
        args.epoch_eval_train = 1000
        args.dc_aug_param = None
    if args.aug != '':
        args.epoch_eval_train = 1000
        args.dc_aug_param = None
    if args.dc_aug_param != None and args.dc_aug_param['strategy'] != 'none':
        args.epoch_eval_train = 1000
    return args

if __name__ == '__main__':
    args = prepare_args()
    evaluator = Evaluator()
    data_file = EvaluatorUtils.get_data_file_name(args.method, args.dataset, args.ipc)
    print("using data file: ", data_file)
    evaluator.load_data(DATA_DIR, data_file, args)
    avg_acc = []
    for i in range(args.num_eval):
        print("current iteration: ", i)
        result = evaluator.evaluate(args)
        avg_acc.append(result)
    mean, std = EvaluatorUtils.compute_std_mean(avg_acc)
    print("%s: final acc is: %.2f +- %.2f, dataset: %s, IPC: %d, DSA:%r, num_eval: %d, aug:%s , model: %s, optimizer: %s"%( 
        args.method.upper(),
        mean * 100, std * 100,
        args.dataset, 
        args.ipc,
        args.dsa,
        args.num_eval,
        args.aug,
        args.model,
        args.optimizer
    ))

