import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from argparse import ArgumentParser
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, sampler


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import scipy.io
import warnings
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, roc_curve


from model import TaskOriNet
from density import GaussianDensityTorch
from utils import plot_roc, plot_pr


warnings.filterwarnings("ignore")




def random_scale(imgs, num_classes, length, cut_len, paste_len):
    labels = []
    rets = torch.zeros((imgs.size(0), imgs.size(1), imgs.size(2), length))
    for i in range(imgs.size(0)):
        scale = random.randint(0, num_classes - 1)
        labels.append(scale)
        rets[i] = cutpaste_and_center_crop_img(imgs[i], scale, length, cut_len, paste_len)
    return rets, torch.LongTensor(labels)




def cutpaste_and_center_crop_img(img_tensor, scale, length, cut_len=100, paste_len=50):
    output = img_tensor.clone()

    if scale == 0:
        return output


    if scale == 1:
        paste_len = random.randint(4, length)
        to_location_x = int(random.uniform(0, length - paste_len))
        mul_factor = random.uniform(2.0, 4.0)
        output[:, :, to_location_x:to_location_x + paste_len] *= mul_factor
        return output


    if scale == 2:
        k = random.randint(0, 1)

        if k == 0:
            freq_factor = random.uniform(0.1, 0.5)
        else:
            freq_factor = random.randint(2, 4)

        output = torch.nn.functional.interpolate(img_tensor, scale_factor=freq_factor, mode='linear',
                                                 align_corners=True)

        output_ = output.clone()
        while output.shape[2] < length:
            output = torch.cat([output, output_], 2)

        num = int(random.uniform(0, output.shape[2] - length - 1))
        output = output[:, :, num:num + length]

        return output










def detm_scale_and_crop(imgs, scale, length, cut_len, paste_len):
    rets = torch.zeros((imgs.size(0), imgs.size(1), imgs.size(2), length))
    for i in range(imgs.size(0)):
        rets[i] = cutpaste_and_center_crop_img(imgs[i], scale, length, cut_len, paste_len)
    return rets








class MyData(Dataset):  
    def __init__(self, txt_file, datatype="train", transform=None):  
        self.txt_file = txt_file  
        self.transform = transform  
        self.files = []
        self.datatype = datatype

        with open(self.txt_file, 'r') as f:
            files = f.readlines()
            for file in files:
                file = file.split('\n')[0]
                self.files.append(file)



    def __len__(self):  
        return len(self.files)



    def __getitem__(self, index):  
        file_path = self.files[index]  
        file = scipy.io.loadmat(file_path)  
        if "abnormal" in file_path:
            label = np.array(1)
        else:
            label = np.array(0)
        transformed_data = np.expand_dims(file['data'], 0)
        return transformed_data, label





def load_data(batch_size=16):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    splits = ['train', 'validate', 'test']
    drop_last_batch = {'train': False, 'validate': False, 'test': False}
    shuffle = {'train': True, 'validate': True, 'test': False}


    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((8.68e-10,), (2.09e-07,))
    ])

    dataset = {}
  
    dataset['train'] = MyData(
        txt_file="./data/all_TXT_3s/train.txt",
        datatype="train", transform=transform_train)

    dataset['train'], dataset['validate'] = torch.utils.data.random_split(dataset['train'],
                                                                          [int(0.75 * len(dataset['train']) + 0.5),
                                                                           int(0.25 * len(dataset['train']) + 0.5)])

    dataset['test'] = MyData(
        txt_file="./data/all_TXT_3s/test.txt",
        datatype="test", transform=transform_train)



    print('load train set {} eegs'.format(len(dataset['train'])))
    print('load val set {} eegs'.format(len(dataset['validate'])))
    print('load test set {} eegs'.format(len(dataset['test'])))


    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=batch_size,
                                                 shuffle=shuffle[x],
                                                 num_workers=0,
                                                 drop_last=drop_last_batch[x],
                                                 pin_memory=True)
                  for x in splits}

    return dataloader








def train(data_loader, epochs, NUM_CLASSES, length, inplane, learning_rate,
          optim_name, model_dir, cut_len, paste_len, density=GaussianDensityTorch()):

    model = TaskOriNet(num_classes=NUM_CLASSES).to(device)

    scheduler = None

    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00003)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    writer_path = './AD_runs/{}'.format(model_dir)
    print('tensorboard runs=', writer_path)
    writer = SummaryWriter(log_dir=writer_path, comment=model_dir)

    best_auc = 0
    for epoch in range(epochs):
        losses = []
        correct_num = 0
        model.train()
        for (x, _) in data_loader['train']:
            x_, label = random_scale(x, num_classes=NUM_CLASSES, length=length, cut_len=cut_len, paste_len=paste_len)
            x_, label = x_.to(device), label.to(device)
            embed, out = model(x_)
            output = F.softmax(out, dim=-1)
            _, index = torch.max(output.cpu(), 1)
            correct_num += torch.sum(index == label.cpu()).item()
            loss = F.cross_entropy(out + 1e-8, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()
        acc_train = correct_num / (len(data_loader['train'].dataset))

        print('-------------------------------------------------')
        print('Child_AD_Density_Features_{}'.format(model_dir))
        print('Ori/Cutpaste Classfication')
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, np.mean(losses)))
        print('====> Epoch: {} Train Accuracy: {:.4f}'.format(
            epoch, acc_train))

        writer.add_scalar('Loss', np.mean(losses), global_step=epoch)
        writer.add_scalar('Train Accuracy', acc_train, global_step=epoch)


        if epoch != 0 and epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_embed = []
                for (x, _) in data_loader['train']:
                    x = x.to(device, dtype=torch.float32)
                    embed, _ = model(x)
                    train_embed.append(embed.cpu())

                train_embed = torch.cat(train_embed)
                train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)
                density.fit(train_embed)

                y_true = []
                embeds = []
                for (x, label) in data_loader['validate']:
                    x = x.to(device, dtype=torch.float32)
                    embed, out = model(x)
                    embeds.append(embed.cpu())
                    y_true.append(label)

                for (x, label) in data_loader['test']:
                    x = x.to(device, dtype=torch.float32)
                    embed, out = model(x)
                    embeds.append(embed.cpu())
                    y_true.append(label)

                y_true = np.concatenate(y_true)
                embeds = torch.cat(embeds)
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
                distances = density.predict(embeds)

                roc_auc = plot_roc(y_true, distances)
                print('====> Epoch: {} AUC: {:.4f}'.format(epoch, roc_auc))
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    torch.save(model.state_dict(), './AD_models/{}/best.pth'.format(model_dir))


        if epoch == epochs - 1:
            model.eval()
            torch.save(model.state_dict(), './AD_models/{}/epochs{}.pth'.format(model_dir, epochs))




    writer.close()







def test_gde(data_loader, NUM_CLASSES, inplane, epochs, model_dir, save_roc_path, save_pr_path, save_plots,
         density=GaussianDensityTorch()):
    model = TaskOriNet(num_classes=NUM_CLASSES).to(device)
    model_name = './AD_models/{}/epochs{}.pth'.format(model_dir, epochs)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    print('gde')

    with torch.no_grad():
        train_embed = []
        for (x, _) in data_loader['train']:
            x = x.to(device, dtype=torch.float32)
            embed, _ = model(x)
            train_embed.append(embed.cpu())

        train_embed = torch.cat(train_embed)
        train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)
        density.fit(train_embed)

        y_true = []
        embeds = []
        for (x, label) in data_loader['validate']:
            x = x.to(device, dtype=torch.float32)
            embed, out = model(x)
            embeds.append(embed.cpu())
            y_true.append(label)

        for (x, label) in data_loader['test']:
            x = x.to(device, dtype=torch.float32)
            embed, out = model(x)
            embeds.append(embed.cpu())
            y_true.append(label)

        y_true = np.concatenate(y_true)
        embeds = torch.cat(embeds)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
        distances = density.predict(embeds)


        roc_auc = plot_roc(y_true, distances)
        pr_auc = plot_pr(y_true, distances)

        fpr, tpr, threshold = roc_curve(y_true, distances)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        precision, recall, thresholds = precision_recall_curve(y_true, distances)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_score = f1_scores[np.where(thresholds == eer_threshold)[0][0]]

        print("roc auc = {:.3f}".format(roc_auc))
        print('f1 score = {:.3f}'.format(f1_score))
        print('EER = {:.3f}'.format(EER))

        return roc_auc, f1_score, EER









def test_ocsvm(data_loader, NUM_CLASSES, inplane, epochs, model_dir, save_roc_path, save_pr_path, save_plots):
    model = TaskOriNet(num_classes=NUM_CLASSES).to(device)
    model_name = './AD_models/{}/epochs{}.pth'.format(model_dir, epochs)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    with torch.no_grad():
        train_embed = []
        for (x, _) in data_loader['train']:
            x = x.to(device, dtype=torch.float32)
            embed, _ = model(x)
            train_embed.append(embed.cpu())

        train_embed = torch.cat(train_embed)
        train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

        gamma = 10. / (torch.var(train_embed) * train_embed.shape[1])
        clf = OneClassSVM(kernel='rbf', gamma=gamma).fit(train_embed)

        y_true = []
        embeds = []
        for (x, label) in data_loader['validate']:
            x = x.to(device, dtype=torch.float32)
            embed, out = model(x)
            embeds.append(embed.cpu())
            y_true.append(label)

        for (x, label) in data_loader['test']:
            x = x.to(device, dtype=torch.float32)
            embed, out = model(x)
            embeds.append(embed.cpu())
            y_true.append(label)

        y_true = np.concatenate(y_true)
        embeds = torch.cat(embeds)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

        scores = -clf.score_samples(embeds)

        roc_auc = plot_roc(y_true, scores)
        pr_auc = plot_pr(y_true, scores)

        fpr, tpr, threshold = roc_curve(y_true, scores)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_score = f1_scores[np.where(thresholds == eer_threshold)[0][0]]

        print("roc auc = {:.3f}".format(roc_auc))
        print('f1 score = {:.3f}'.format(f1_score))
        print('EER = {:.3f}'.format(EER))

        return roc_auc, f1_score, EER












if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of epochs to train the model , (default: 300)')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--cuda', dest='cuda', type=int, default=1, help='cuda number')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='num classes')
    parser.add_argument('--length', default=769, type=int,
                        help='length')
    parser.add_argument('--inplane', default=18, type=int,
                        help='inplane')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning_rate')
    parser.add_argument('--cut_len', default=300, type=int,
                        help='cut_len')
    parser.add_argument('--paste_len', default=500, type=int,
                        help='paste_len')
    parser.add_argument('--optim', default="adam",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "adam")')
    parser.add_argument('--save_plots', default="True", type=bool,
                        help='save roc-curve and pr-curve or not')


    options = parser.parse_args()

    device = torch.device('cuda:{}'.format(options.cuda))


    if not options.eval:
        for seed in [2020, 2021, 2022, 2023, 2024]:
            torch.manual_seed(seed)
            model_dir = '{}/{}'.format("ours3", seed)

            if not os.path.exists('./AD_models/{}'.format(model_dir)):
                os.makedirs('./AD_models/{}'.format(model_dir))

            data_loader = load_data(options.batch_size)
            print("training.")
            train(data_loader, options.epochs, options.num_classes, options.length, options.inplane,
                  options.learning_rate, options.optim, model_dir, options.cut_len, options.paste_len)



    else:
        print("testing.")
        each_aucs = []
        each_f1scores = []
        each_eers = []
        for seed in [2020, 2021, 2022, 2023, 2024]:
            torch.manual_seed(seed)

            if seed == 2020 or seed == 2021:
                options.epochs = 400
            elif seed == 2024:
                options.epochs = 300
            else:
                options.epochs = -1

            model_dir = '{}/{}'.format("ours3", seed)
            print('model_dir =', model_dir)

            data_loader = load_data(options.batch_size)

            roc_path = "./AD_models/{}/roc_all.png".format(model_dir)
            pr_path = "./AD_models/{}/pr_all.png".format(model_dir)

            # auc, f1, eer = test_gde(data_loader, options.num_classes, options.inplane, options.epochs, model_dir,
            #                         roc_path, pr_path, options.save_plots)


            auc, f1, eer = test_ocsvm(data_loader, options.num_classes, options.inplane, options.epochs, model_dir,
                                      roc_path, pr_path, options.save_plots)


            each_aucs.append(auc)
            each_f1scores.append(f1)
            each_eers.append(eer)



        print('-----------------------')
        print()
        print('AUC')
        print('mean = {:.7f}'.format(np.mean(each_aucs)))
        print('std = {:.7f}'.format(np.std(each_aucs, ddof=1)))
        print('f1-score')
        print('mean = {:.7f}'.format(np.mean(each_f1scores)))
        print('std = {:.7f}'.format(np.std(each_f1scores, ddof=1)))
        print('EER')
        print('mean = {:.7f}'.format(np.mean(each_eers)))
        print('std = {:.7f}'.format(np.std(each_eers, ddof=1)))
