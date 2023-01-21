import os
import torch

class TMDataLoader:

    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        image_path = data_file[0]
        label_path = data_file[1]
        image_path = os.path.join(root_dir, "TM", dataset, 'IPC' + str(ipc), image_path)
        label_path = os.path.join(root_dir, "TM", dataset, 'IPC' + str(ipc), label_path)

        if dataset == 'tinyimagenet' and ipc == 50:
            # the data are broken down into 5 files.
            images = []
            labels = []
            for i in range(5):
                images.append(torch.load(image_path.replace('best', 'best_'+str(i))))
                labels.append(torch.load(label_path.replace('best', 'best_' + str(i))).long())
            return torch.cat(images), torch.cat(labels)
        else:
            return torch.load(image_path), torch.load(label_path).long()

    @staticmethod
    def get_data_file_name(method, dataset, ipc):
        return ('images_best.pt', 'labels_best.pt')