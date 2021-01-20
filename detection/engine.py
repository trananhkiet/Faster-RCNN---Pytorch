import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from detection.coco_utils import get_coco_api_from_dataset
from detection.coco_eval import CocoEvaluator
import detection.utils as utils




def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,writer_tensorboard):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)


    loss_all_step=[[],[],[],[]]
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        list_loss = [loss.item() for loss in loss_dict_reduced.values()]

        for index,loss in enumerate(list_loss):
            loss_all_step[index].append(loss)

    loss_classifier=sum(loss_all_step[0])/len(loss_all_step[0])
    loss_box_reg=sum(loss_all_step[1])/len(loss_all_step[1])
    loss_objectness=sum(loss_all_step[2])/len(loss_all_step[2])
    loss_rpn_box_reg=sum(loss_all_step[3])/len(loss_all_step[3])
    loss=loss_classifier+loss_box_reg+ loss_objectness+ loss_rpn_box_reg

    writer_tensorboard.add_scalars('Loss', {'train': float(loss)}, epoch)
    writer_tensorboard.add_scalars('loss_classifier', {'train': float(loss_classifier)}, epoch)
    writer_tensorboard.add_scalars('loss_box_reg', {'train': float(loss_box_reg)}, epoch)
    writer_tensorboard.add_scalars('loss_objectness', {'train': float(loss_objectness)}, epoch)
    writer_tensorboard.add_scalars('loss_rpn_box_reg', {'train': float(loss_rpn_box_reg)}, epoch)


    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):

        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def evaluate_validation(model, data_loader, device, epoch,writer_tensorboard):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation: [{}]'.format(epoch)

    loss_all_step = [[], [], [], []]
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        list_loss = [loss.item() for loss in loss_dict_reduced.values()]

        for index, loss in enumerate(list_loss):
            loss_all_step[index].append(loss)

    loss_classifier = sum(loss_all_step[0]) / len(loss_all_step[0])
    loss_box_reg = sum(loss_all_step[1]) / len(loss_all_step[1])
    loss_objectness = sum(loss_all_step[2]) / len(loss_all_step[2])
    loss_rpn_box_reg = sum(loss_all_step[3]) / len(loss_all_step[3])
    loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg

    writer_tensorboard.add_scalars('Loss', {'Validation': float(loss)}, epoch)
    # writer_tensorboard.add_scalars('loss_classifier', {'train': float(loss_classifier)}, epoch)
    # writer_tensorboard.add_scalars('loss_box_reg', {'train': float(loss_box_reg)}, epoch)
    # writer_tensorboard.add_scalars('loss_objectness', {'train': float(loss_objectness)}, epoch)
    #writer_tensorboard.add_scalars('loss_rpn_box_reg', {'train': float(loss_rpn_box_reg)}, epoch)
