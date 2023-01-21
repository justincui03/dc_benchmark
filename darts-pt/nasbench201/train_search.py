import os
import sys
sys.path.insert(0, '../')
import time
import glob
import numpy as np
import torch
import shutil
import nasbench201.utils as ig_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset


from nasbench201.search_model_darts import TinyNetworkDarts
from nasbench201.search_model_darts_proj import TinyNetworkDartsProj
from nasbench201.cell_operations import SearchSpaceNames
from nasbench201.architect_ig import Architect
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API
from nasbench201.projection import pt_project

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet16-120'], help='choose dataset')
parser.add_argument('--method', type=str, default='dirichlet', help='choose nas method')
parser.add_argument('--search_space', type=str, default='nas-bench-201')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for alpha')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
#### common
parser.add_argument('--fast', action='store_true', default=False, help='skip loading api which is slow')
parser.add_argument('--resume_epoch', type=int, default=0, help='0: from scratch; -1: resume from latest checkpoints')
parser.add_argument('--resume_expid', type=str, default='', help='e.g. search-darts-201-2')
parser.add_argument('--dev', type=str, default=None, help='separate supernet traininig and projection phases')
parser.add_argument('--ckpt_interval', type=int, default=20, help='frequency for ckpting')
parser.add_argument('--expid_tag', type=str, default='none', help='extra tag for exp identification')
parser.add_argument('--log_tag', type=str, default='', help='extra tag for log during arch projection')
#### projection
parser.add_argument('--edge_decision', type=str, default='random', choices=['random'], help='which edge to be projected next')
parser.add_argument('--proj_crit', type=str, default='acc', choices=['loss', 'acc'], help='criteria for projection')
parser.add_argument('--proj_intv', type=int, default=5, help='fine tune epochs between two projections')
parser.add_argument('--dc_method', type=str, default='random', help='random,kmeans,dc,dsa,dm,tm')
args = parser.parse_args()


#### macros


#### args augment
expid = args.save
args.save = '../experiments/nasbench201/search-{}-{}'.format(args.save, args.seed)
if not args.dataset == 'cifar10':
    args.save += '-' + args.dataset
if args.expid_tag != 'none': args.save += '-' + args.expid_tag


#### logging
if args.resume_epoch > 0: # do not delete dir if resume:
    args.save = '../experiments/nasbench201/{}'.format(args.resume_expid)
    if not os.path.exists(args.save):
        print('no such directory {}'.format(args.save))
else:
    scripts_to_save = glob.glob('*.py') + ['../exp_scripts/{}.sh'.format(expid)]
    if os.path.exists(args.save):
        if input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
            print('proceed to override saving directory')
            shutil.rmtree(args.save)
        else:
            exit(0)
    ig_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

# log_format = '%(asctime)s %(message)s'
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.WARNING,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.resume_epoch > 0:
    log_file = 'log_resume-{}_dev-{}_seed-{}_intv-{}'.format(args.resume_epoch, args.dev, args.seed, args.proj_intv)
    if args.log_tag != '':
        log_file += args.log_tag
else:
    log_file = args.dc_method + '_log'
if args.log_tag == 'debug':
    log_file = 'log_debug'
log_file += '.txt'
log_path = os.path.join(args.save, log_file)
logging.info('======> log filename: %s', log_file)

if args.log_tag != 'debug' and os.path.exists(log_path):
    if input("WARNING: {} exists, override?[y/n]".format(log_file)) == 'y':
        print('proceed to override log file directory')
    else:
        exit(0)

fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')


#### macros
if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet16-120':
    n_classes = 120
else:
    n_classes = 10


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    gpu = ig_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info('gpu device = %d' % gpu)

    if not args.fast:
        api = API('../data/NAS-Bench-201-v1_0-e61699.pth')

    evaluated_models = []
    # set up logging title.
    logging.warning("no\t test\t cifar10_test\t cifar100_test\t tinyimagenet_test")
    class TensorDataset(Dataset):
        def __init__(self, images, labels): # images: n x c x h x w tensor
            self.images = images.detach().float()
            self.labels = labels.detach()

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

        def __len__(self):
            return self.images.shape[0]

    networks = torch.load(os.getcwd() + '/networks.pt')

    #### data
    import torchvision.transforms as transforms
    if args.dataset == 'cifar10':
        # train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        if args.dc_method == 'kip':
            valid_transform = None

        real_train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)

        if args.dc_method == 'kip':
            images = torch.tensor(np.transpose(real_train_data.data, (0, 3, 1, 2))) / 255.0
            labels = torch.tensor(real_train_data.targets)

            orig_shape = images.shape
            images = torch.reshape(images, (orig_shape[0], -1))

            images = images - torch.mean(images, dim=1, keepdim=True)
            # Normalize
            train_norms = torch.norm(images, dim=1, keepdim=True)
            # Make features unit norm
            images = images / train_norms

            n_train = images.shape[0]

            cov = 1.0 / n_train * torch.matmul(torch.t(images), images)

            reg_amount = 0.1 * torch.trace(cov) / cov.shape[0]

            u, s, _ = torch.svd(cov + reg_amount * torch.eye(cov.shape[0]))

            inv_sqrt_zca_eigs = s ** (-1 / 2)

            whitening_transform = torch.einsum('ij,j,kj->ik', u, inv_sqrt_zca_eigs, u)

            images = torch.matmul(images, whitening_transform)

            images = torch.reshape(images, orig_shape)

            real_train_data = TensorDataset(images, labels)

        # valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        # random
        num_train = len(real_train_data)
        indices = list(range(num_train))
        np.random.RandomState(10).shuffle(indices)
        split = int(np.floor(args.train_portion * num_train))
        print("---------number of test set:", num_train - split)
        print("current path:", os.getcwd())
        test_queue = torch.utils.data.DataLoader(
            real_train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
            pin_memory=True)
        data_path = os.getcwd() + '/../../distilled_results'
        if args.dc_method != 'whole':
            if args.dc_method == 'random':
                training_images = torch.load(data_path + '/random/CIFAR10/IPC50/CIFAR10_IPC50_normalize_images.pt')
                training_labels = torch.load(data_path + '/random/CIFAR10/IPC50/CIFAR10_IPC50_normalize_labels.pt')
            elif args.dc_method == 'tm':
                training_images = torch.load(data_path + '/TM/CIFAR10/IPC50/images_best.pt')
                training_labels = torch.load(data_path + '/TM/CIFAR10/IPC50/labels_best.pt')
            elif args.dc_method == 'dm':
                dm_data = torch.load(data_path + '/DM/CIFAR10/IPC50/res_DM_CIFAR10_ConvNet_50ipc.pt')
                training_data = dm_data['data']
                training_images, training_labels = training_data[-1]
            elif args.dc_method == 'dc':
                dc_data = torch.load(data_path + '/DC/CIFAR10/IPC50/res_DC_CIFAR10_ConvNet_50ipc.pt')
                training_data = dc_data['data']
                training_images, training_labels = training_data[-1]
            elif args.dc_method == 'dsa':
                dsa_data = torch.load(data_path + '/DSA/CIFAR10/IPC50/res_DSA_CIFAR10_ConvNet_50ipc.pt')
                training_data = dsa_data['data']
                training_images, training_labels = training_data[-1]
            elif args.dc_method == 'kmeans':
                training_images = torch.load(data_path + '/kmeans-emb/CIFAR10/IPC50/CIFAR10_IPC50_images.pt')
                training_labels = torch.load(data_path + '/kmeans-emb/CIFAR10/IPC50/CIFAR10_IPC50_labels.pt')
            elif args.dc_method == 'kip':
                data = np.load(data_path + '/KIP/CIFAR10/IPC50/kip_cifar10_ConvNet3_ssize500_zca_nol_noaug_ckpt1000.npz')
                training_images = torch.from_numpy(data['images']).permute(0, 3, 1, 2)
                labels = torch.from_numpy(data['labels'])
                training_labels = torch.argmax(labels, dim=1)
            train_data = TensorDataset(training_images, training_labels.long())
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size)
        else:
            # only train full data with 10 epochs, we already get best performance within 10 epochs.
            args.epochs = 10
            train_queue = torch.utils.data.DataLoader(
                real_train_data, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                pin_memory=True)
    for model_number in range(len(networks)):
        #### model
        criterion = nn.CrossEntropyLoss()
        search_space = SearchSpaceNames[args.search_space]
        args.network_init = networks[model_number]
        if args.method in ['darts', 'blank']:
            model = TinyNetworkDarts(C=args.init_channels, N=1, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
        elif args.method in ['darts-proj', 'blank-proj']:
            model = TinyNetworkDartsProj(C=args.init_channels, N=1, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
        genotype = model.genotype()
        model = model.cuda()
        logging.info("param size = %fMB", ig_utils.count_parameters_in_MB(model))

        architect = Architect(model, args)

        #### scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model.optimizer, float(args.epochs), eta_min=args.learning_rate_min)


        #### resume
        #### training
        for epoch in range(args.epochs):
            lr = scheduler.get_lr()[0]
            ## data aug
            if args.cutout:
                train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
                logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                            train_transform.transforms[-1].cutout_prob)
            else:
                logging.info('epoch %d lr %e', epoch, lr)

            ## pre logging
            genotype = model.genotype()
            # logging.info('genotype = %s', genotype)
            # model.printing(logging)

            ## training
            # for i in range(200):
            train_acc, train_obj = train(train_queue, model, architect, model.optimizer, lr, epoch)
            # logging.info('train_acc  %f', train_acc)
            print('%d train_acc  %f' % (epoch, train_acc))
            # logging.info('train_loss %f', train_obj)

            ## eval
            # valid_acc, valid_obj = infer(valid_queue, model, criterion, log=False)
            # logging.info('valid_acc  %f', valid_acc)
            # logging.info('valid_loss %f', valid_obj)

            # logging.warning("%.2f\t %.2f\t %.2f \t %.2f" % (test_acc, cifar10_test, cifar100_test, imagenet16_test))
            # test_acc, test_obj = infer(test_queue, model, criterion, log=False)
            # print("-------------------------------")
            # print('test_acc  %f' % test_acc)
            # print('test_loss %f' % test_obj)

            #### scheduling
            scheduler.step()
        test_acc, test_obj = infer(test_queue, model, criterion, log=False)
        print("-------------------------------")
        print('test_acc  %f' % test_acc)
        print('test_loss %f' % test_obj)
        # nasbench201
        cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
            cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = query(api, model.genotype(), logging)

        logging.warning("%d\t %.2f\t %.2f\t %.2f \t %.2f" % (model_number, test_acc, cifar10_test, cifar100_test, imagenet16_test))
        writer.close()


def train(train_queue, model, architect, optimizer, lr, epoch):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()

    sample_count = [0 for i in range(10)]

    # for step in range(len(train_queue)):
    # sample_count = [0 for i in range(10)]
    for step, datum in enumerate(train_queue):
        model.train()

        ## data
        input, target = datum[0], datum[1]
        # for i in target:
        #     sample_count[i] += 1
        input = input.cuda(); target = target.cuda(non_blocking=True)

        ## train alpha
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        # shared = architect.step(input, target, input_search, target_search,
        #                         eta=lr, network_optimizer=optimizer)

        ## train weight
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        logits, loss = model.step(input, target, args, shared=None)

        ## logging
        prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        if args.fast:
            break

    # print(sample_count)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion,
          log=True, eval=True, weights=None, double=False, bn_est=False):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()
    model.eval() if eval else model.train() # disable running stats for projection
    
    if bn_est:
        _data_loader = deepcopy(valid_queue)
        for step, (input, target) in enumerate(_data_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits = model(input)
        model.eval()

    # sample_count = [0 for i in range(10)]
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            # for i in target:
            #     sample_count[i] += 1
            target = target.cuda(non_blocking=True)
            if double:
                input = input.double(); target = target.double()
            
            logits = model(input) if weights is None else model(input, weights=weights)
            loss = criterion(logits, target)

            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if log and step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            if args.fast:
                break
    # print("--------------test sample count")
    # print(sample_count)
    return top1.avg, objs.avg


#### util functions
def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def query(api, genotype, logging):
    result = api.query_by_arch(genotype, hp="200")
    logging.info('{:}'.format(result))
    cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)
    logging.info('cifar10 train %f test %f', cifar10_train, cifar10_test)
    logging.info('cifar100 train %f valid %f test %f', cifar100_train, cifar100_valid, cifar100_test)
    logging.info('imagenet16 train %f valid %f test %f', imagenet16_train, imagenet16_valid, imagenet16_test)
    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
           cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


if __name__ == '__main__':
    # print("generating network architectures.")
    # networks = []
    # for i in range(110):
    #     data = torch.randint(0, 5, (6,))
    #     networks.append(data)
    # torch.save(networks, 'networks.pt')
    main()
