import os
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import TensorDataset


class KIPDataLoader:

    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        data_path = os.path.join(root_dir, "KIP", dataset, 'IPC' + str(ipc), data_file)
        data = np.load(data_path)
        images = torch.from_numpy(data['images']).permute(0, 3, 1, 2)
        labels = torch.from_numpy(data['labels'])
        # convert label to numbers using argmax
        labels = torch.argmax(labels, dim=1)
        return (images, labels)

    @staticmethod
    def get_data_file_name(method, dataset, ipc):
        dataset = dataset.lower()
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tinyimagenet':
            num_classes = 200
        # KIP doesn't have data for tinyimagenet
        # Will add it here once they release the data.
        if dataset == 'cifar10':
            return ('%s_%s_ConvNet3_ssize%d_zca_nol_noaug_ckpt1000.npz' % (method.lower(), dataset.lower(), num_classes * ipc))
        elif dataset == 'cifar100':
            # KIP doesn't have data for ConvNet3 for cifar100
            # Will add it here once they release the data.
            return ('%s_%s_ConvNet_ssize%d_zca_nol_noaug_ckpt1000.npz' % (method.lower(), dataset.lower(), num_classes * ipc))


    @staticmethod
    def load_dataset(dataset):
        if dataset == 'CIFAR10':
            dst_train = datasets.CIFAR10("../data", train=True, download=True, transform=None)
            dst_test = datasets.CIFAR10("../data", train=False, download=True, transform=None)
        elif dataset == 'CIFAR100':
            dst_train = datasets.CIFAR100("../data", train=True, download=True, transform=None)
            dst_test = datasets.CIFAR100("../data", train=False, download=True, transform=None)
        elif dataset == 'tinyimagenet':
            dst_train = datasets.ImageFolder(os.path.join('/nfs/data/justincui/data/tiny-imagenet-200', "train"), transform=transform)
            dst_test = datasets.ImageFolder(os.path.join('/nfs/data/justincui/data/tiny-imagenet-200', "val", "images"), transform=transform)

        images = torch.tensor(np.transpose(dst_train.data, (0, 3, 1, 2))) / 255.0
        labels = torch.tensor(dst_train.targets)

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

        dst_train = TensorDataset(images, labels)

        test_images = torch.tensor(np.transpose(dst_test.data, (0, 3, 1, 2))) / 255.0
        test_labels = torch.tensor(dst_test.targets)

        orig_shape = test_images.shape

        test_images = torch.reshape(test_images, (orig_shape[0], -1))

        test_images = test_images - torch.mean(test_images, dim=1, keepdim=True)
        test_norms = torch.norm(test_images, dim=1, keepdim=True)
        test_images = test_images / test_norms

        test_images = torch.matmul(test_images, whitening_transform)

        test_images = torch.reshape(test_images, orig_shape)

        dst_test = TensorDataset(test_images, test_labels)
        dst_test = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
        return dst_train, dst_test


if __name__ == '__main__':
    
    images, labels = KIPDataLoader.load_data('/nfs/data/justincui/dc_benchmark/distilled_results', 'CIFAR10', 10, 'kip_cifar10_ConvNet_ssize100_zca_nol_noaug_ckpt1000.npz')
    print(images.shape)
    print(labels.shape)
    print(labels.max(), labels.min())
    print(images.max(), images.min())
    # print((labels + 1/10).long())