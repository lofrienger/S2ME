import numpy as np
import torch
from medpy import metric
from utils.metrics import *
from hausdorff import hausdorff_distance

def calculate_metric_polyp(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dc = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        pre = metric.binary.precision(pred, gt)
        hd = hausdorff_distance(gt.astype(np.uint8), pred.astype(np.uint8), distance='euclidean')
        return dc, jc, pre, hd
    else:
        return np.zeros(4)


def test_polyp_batch(image, label, net):
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(image), dim=1), dim=1)
        prediction = out.cpu().detach().numpy()
    label = label.squeeze(1).cpu().detach().numpy()
    num_sample = prediction.shape[0]
    batch_metric = np.zeros(4)
    for i in range(num_sample):
        # only compute for foreground class
        batch_metric += np.array(calculate_metric_polyp(prediction[i] == 1, label[i] == 1)) #dc, jc, pre, hd
    return batch_metric / num_sample  # per batch metric
