import torch

class DDDataLoader:

    @staticmethod
    def load_data(path_to_pt_file):
        data = torch.load(path_to_pt_file)
        return data[-1][0], data[-1][1]



if __name__ == '__main__':
    
    images, labels = DDDataLoader.load_data('/home/justincui/dc_benchmark/dc_benchmark/distilled_results/DD/CIFAR10/IPC10/results.pth')
    print(images.shape)
    print(labels.shape)