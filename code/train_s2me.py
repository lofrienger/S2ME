import argparse
import logging
import os
import random
import sys
import copy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from dataloaders.dataset_polyp import (PolypDataset, trsf_train_image_224,
                                       trsf_valid_image_224)
from networks.net_factory import net_factory
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from utils import losses, ramps
from valid import test_polyp_batch

parser = argparse.ArgumentParser()
parser.add_argument('--ds_root', type=str,
                    default='../dataset/Polyp/SUN_SEG/data/SUN-SEG', help='dataset root dir')
parser.add_argument('--csv_root', type=str,
                    default='../data/polyp', help='data split json file dir')

parser.add_argument('--model1', type=str,
                    default='unet', help='model-1 name')
parser.add_argument('--model2', type=str,
                    default='ynet_ffc', help='model-2 name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--exp', type=str,
                    default='full_unet', help='experiment_name')

parser.add_argument('--sup', type=str,
                    default='Scribble', help='supervision type')
parser.add_argument('--sup_loss', type=str,
                    default='pce', help='supervision loss type')

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.03,
                    help='segmentation network learning rate')

parser.add_argument('--mps', type=str,
                    default='False', help='use mixed pseudo supervision or not')
parser.add_argument('--mps_type', type=str,
                    default='random', help='strategy to mix pseudo label')
parser.add_argument('--cps', type=str,
                    default='False', help='use cross pseudo supervision or not')

parser.add_argument('--max_consistency_weight', type=float, default=5.0,
                    help='segmentation network learning rate')
parser.add_argument('--consistency_rampup_length', type=int,
                    default=25000, help='consistency rampup length')
parser.add_argument('--consistency_rampup_type', type=str,
                    default='sigmoid', help='consistency rampup type')

parser.add_argument('--seed', type=int,  default=2022, help='random seed')

args = parser.parse_args()

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def train(args, snapshot_path):

    logging.info(str(args))
    ToPIL = ToPILImage()

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train = PolypDataset(ds_root=args.ds_root, csv_root=args.csv_root, split="train", label=args.sup, transform=trsf_train_image_224)
    db_valid = PolypDataset(ds_root=args.ds_root, csv_root=args.csv_root, split="valid", label='GT', transform=trsf_valid_image_224)
    db_test = PolypDataset(ds_root=args.ds_root, csv_root=args.csv_root, split="test", label='GT', transform=trsf_valid_image_224)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    validloader = DataLoader(db_valid, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=4)

    model1 = net_factory(net_type=args.model1, in_chns=3, class_num=num_classes)
    model2 = net_factory(net_type=args.model2, in_chns=3, class_num=num_classes)
    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=2)
    if args.sup == 'GT':
        dice_loss = losses.DiceLoss(num_classes)
    else:
        dice_loss = losses.pDLoss(num_classes, ignore_index=2)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1, best_performance2 = 0.0, 0.0
    best_iter1, best_iter2 = 0, 0
    best_model1, best_model2 = None, None
    iterator = tqdm(range(max_epoch), ncols=70)
    logging.info("{} iterations per epoch".format(len(trainloader)))

    for epoch_num in iterator:
        for i_batch, (images, masks) in enumerate(trainloader):
            images, masks = images.float().cuda(), masks.long().cuda()

            outputs1 = model1(images)
            outputs2 = model2(images)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            output_lab1 = torch.argmax(outputs_soft1, dim=1, keepdim=False)
            output_lab2 = torch.argmax(outputs_soft2, dim=1, keepdim=False)

            loss_ce1 = ce_loss(outputs1, masks)
            loss_ce2 = ce_loss(outputs2, masks)
            loss_sup = loss_ce1 + loss_ce2

            if args.consistency_rampup_type == 'constant':
                consistency_weight = args.max_consistency_weight
            elif args.consistency_rampup_type == 'sigmoid':
                consistency_weight = args.max_consistency_weight * ramps.sigmoid_rampup(iter_num, args.consistency_rampup_length)
            
            # mixed pseudo supervision
            if args.mps == 'True':
                if args.mps_type == 'random':
                    beta = random.random() + 1e-10
                    pseudo_soft = beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()
                elif args.mps_type == 'equal':
                    beta = 0.5
                    pseudo_soft = beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()
                elif args.mps_type == 'entropy':
                    entmap1 = losses.entropy_map(outputs_soft1)
                    entmap2 = losses.entropy_map(outputs_soft2)
                    beta_entmap = entmap1 / (entmap1+entmap2)
                    pseudo_soft = torch.zeros_like(outputs_soft1)
                    pseudo_soft = beta_entmap * outputs_soft2.detach() + (1-beta_entmap) * outputs_soft1.detach()
                
                pseudo_supervision = torch.argmax(pseudo_soft, dim=1, keepdim=False)
                mps_loss1 = dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(1)) + ce_loss(outputs1, pseudo_supervision)
                mps_loss2 = dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1)) + ce_loss(outputs2, pseudo_supervision)
                loss_mps = consistency_weight * (mps_loss1 + mps_loss2)
            else:
                loss_mps = 0
            
            # cross pseudo supervision
            if args.cps == 'True':
                cps_loss1 = ce_loss(outputs1, output_lab2) + dice_loss(outputs_soft1, output_lab2.unsqueeze(1))
                cps_loss2 = ce_loss(outputs2, output_lab1) + dice_loss(outputs_soft2, output_lab1.unsqueeze(1)) 
                loss_cps = consistency_weight * (cps_loss1 + cps_loss2)
            else:
                loss_cps = 0

            # total loss = loss_sup + loss_cps + loss_mps
            loss = loss_sup + loss_mps + loss_cps

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            logging.info(
                'iteration %d: total_loss: %f, loss_sup: %f, loss_mps: %f, loss_cps: %f' %
                (iter_num, loss, loss_sup, loss_mps, loss_cps))

            if iter_num > 0 and iter_num % 100 == 0:
                logging.info("Evaluation Started ==>")
                model1.eval()
                model2.eval()
                metric_list1, metric_list2 = np.zeros(4), np.zeros(4)
                for i_batch, (images_val, masks_val) in enumerate(validloader):
                    images_val = images_val.float().cuda()
                    metric_i1 = test_polyp_batch(
                        images_val, masks_val, model1)
                    metric_i2 = test_polyp_batch(
                        images_val, masks_val, model2)
                    metric_list1 += metric_i1
                    metric_list2 += metric_i2
                metric_list1 = metric_list1 / (i_batch+1)
                metric_list2 = metric_list2 / (i_batch+1)
                metric_dict1 = dict(dc=metric_list1[0], jc=metric_list1[1], pre=metric_list1[2], hd=metric_list1[3])
                metric_dict2 = dict(dc=metric_list2[0], jc=metric_list2[1], pre=metric_list2[2], hd=metric_list2[3])

                logging.info('==> valid iteration %d: unet metrics: %s, ynet metrics: %s.' % (iter_num, metric_dict1, metric_dict2))

                if metric_dict1['dc'] > best_performance1:
                    best_performance1 = metric_dict1['dc']
                    best_model1 = copy.deepcopy(model1)
                    best_iter1 = iter_num
                    save_mode_path = os.path.join(snapshot_path,
                                                    'iter_{}_dice_{}.pth'.format(
                                                        iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                                '{}_best_model.pth'.format(args.model1))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)
                    logging.info(
                        '==> New best valid dice for unet: %f, at iteration %d' % (metric_dict1['dc'], iter_num))
                    
                if metric_dict2['dc'] > best_performance2:
                    best_performance2 = metric_dict2['dc']
                    best_model2 = copy.deepcopy(model2)
                    best_iter2 = iter_num
                    save_mode_path = os.path.join(snapshot_path,
                                                    'iter_{}_dice_{}.pth'.format(
                                                        iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                                '{}_best_model.pth'.format(args.model2))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)
                    logging.info(
                        '==> New best valid dice for ynet: %f, at iteration %d' % (metric_dict2['dc'], iter_num))
                
                model1.train()
                model2.train()
                logging.info("Evaluation Finished!⏹️")

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(
                    snapshot_path, '{}_last_model.pth'.format(args.model1))
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save last unet model to {}".format(save_mode_path))
                save_mode_path = os.path.join(
                    snapshot_path, '{}_last_model.pth'.format(args.model2))
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save last ynet model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    logging.info("Training Finished!⏹️")
    logging.info("Unet Best Validation dice: {} in iter: {}.".format(round(best_performance1, 4), best_iter1))
    logging.info("Ynet Best Validation dice: {} in iter: {}.".format(round(best_performance2, 4), best_iter2))

    logging.info('==> Test with best models:')
    test_metric1, test_metric2 = np.zeros(4), np.zeros(4)
    for i_batch, (images_test, masks_test) in enumerate(testloader):
        images_test = images_test.float().cuda()
        metric_i1 = test_polyp_batch(
            images_test, masks_test, best_model1)
        test_metric1 += metric_i1
        metric_i2 = test_polyp_batch(
            images_test, masks_test, best_model2)
        test_metric2 += metric_i2
    test_metric1 = test_metric1 / (i_batch+1)
    test_metric2 = test_metric2 / (i_batch+1)
    metric_dict1 = dict(unet_dc=test_metric1[0], unet_jc=test_metric1[1], unet_pre=test_metric1[2], unet_hd=test_metric1[3])
    logging.info('==> Unet Test metrics: %s' % metric_dict1)
    metric_dict2 = dict(ynet_dc=test_metric2[0], ynet_jc=test_metric2[1], ynet_pre=test_metric2[2], ynet_hd=test_metric2[3])
    logging.info('==> Ynet Test metrics: %s' % metric_dict2)


if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    snapshot_path = "../model/{}/{}".format(args.exp, args.sup)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    train(args, snapshot_path)
