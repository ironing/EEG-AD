import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # import this after torch or it will break everything
from io import BytesIO
import torch
from sklearn.metrics import roc_curve,auc,precision_recall_curve,precision_score,recall_score






def squared_difference(a, b, do_normalization=True):
    """Computes (a-b) ** 2."""
    if do_normalization:
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)
        return -2. * a.mm(b.t())
    return torch.norm(a, dim=1, keepdim=True)**2 + \
           (torch.norm(b, dim=1, keepdim=True)**2).t() - \
           2. * a.mm(b.t())




def plot_roc(labels, scores, filename="", modelname="", save_plots=False):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    # plot roc
    if save_plots:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc



def plot_pr(labels, scores, filename="", modelname="", save_plots=False):
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    # plot pr
    if save_plots:
        plt.figure()
        plt.plot(recall, precision, color='darkorange',
                 lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return pr_auc
