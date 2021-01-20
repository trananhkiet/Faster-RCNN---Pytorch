# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import glob
import json
import torch
from PIL import Image
import cv2
import random
import torch.nn as nn


import argparse
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models.utils import load_state_dict_from_url
from detection.engine import train_one_epoch, evaluate, evaluate_validation
from utils.utils import set_GPU
from utils.write_score import score_statistics
import detection.utils as utils
import detection.transforms as T
from torchvision.transforms import ToTensor

from tensorboardX import SummaryWriter

os.environ["MKL_NUM_THREADS"] = "3" # "6"
os.environ["OMP_NUM_THREADS"] = "2" # "4"
os.environ["NUMEXPR_NUM_THREADS"] = "3" # "6"

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_intersection(box2, boxes1, area2[i], area1)
    return overlaps

def compute_intersection(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = np.asarray([b if box_area >= b else box_area for b in boxes_area])
    iou = intersection / union
    return iou


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.transform_path=""
        # load all image files, sorting them to
        # ensure that they are aligned
        if root.split("\\")[-1]=="Train":
            self.imgs=list(sorted(glob.glob(os.path.join(root,"OriginImage","*.bmp"))))
            self.transform_path=os.path.join(root,"TransformImage")
        else:
            self.imgs = list(sorted(glob.glob(os.path.join(root,"*.bmp"))))

        #self.masks = list(sorted(glob.glob(os.path.join(root,"*.bmp"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = ""

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        # random use transform image

        # get validation image
        if self.transform_path=="":
            img_path = img_path
        else:
            # get train image
            if random.randint(0, 1) == 0:
                img_path=img_path
            else:
                transform_string=['90','180','270','hor','ver']
                choice=random.choice(transform_string)
                image_name=img_path.split("\\")[-1]
                img_path=os.path.join(self.transform_path,choice+"_"+image_name)


        # **** Load Image
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Error image path: ",img_path)


        # Load mask ( use for mask rcnn)
        # mask = Image.open(mask_path)
        # mask = np.array(mask)
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]
        #
        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]

        # Read Json of image
        json_path=img_path+".json"
        with open(json_path, 'r') as f:
            file = json.load(f)

        obj_ids=file["classId"]
        num_objs = len(obj_ids)
        boxes = []
        for index in range(num_objs):
            listx = file["regions"][str(index)]["List_X"]
            listy = file["regions"][str(index)]["List_Y"]
            xmin = min(listx)
            xmax = np.max(listx)
            ymin = np.min(listy)
            ymax = np.max(listy)
            boxes.append([xmin, ymin, xmax, ymax])

        # get bounding box coordinates for each mask
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        #masks = torch.as_tensor(masks, dtype=torch.uint8)


        class_str=["","Bridging defect", "Bridging defect 1","Overkill","Single Overkill"]
        for i in range(num_objs):
            labels[i]=class_str.index(obj_ids[i])


        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model(backbone_string,num_classes,init_weights=""):
    pretrained_backbone = False
    backbone = resnet_fpn_backbone(backbone_string, pretrained_backbone, trainable_layers=5)
    model = FasterRCNN(backbone, num_classes=91)

    if init_weights.split("\\")[-1]=="coco_weights.pth":
        state_dict = torch.load(args.weights)
        model.load_state_dict(state_dict['model'])

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def init_and_load_weight(weight_path):
    num_classes=5

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(weight_path))
    model.eval()
    x = model.to(device)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == "__main__":

    parser=argparse.ArgumentParser("Faster RCNN")

    parser.add_argument('command',type=str,help='write "train" to train model or "test" to test model')
    parser.add_argument('--dataset', default='dataset', help='dataset')
    parser.add_argument('--weights', default="", help='weight of pretrain model')
    parser.add_argument('--config',default=None,help='the directory of config file')
    parser.add_argument('--logs',default='logs',help='the directory to log tensorboard and weight of model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--score_statistic_name',default=None,help='the directory to log score statistics')

    args=parser.parse_args()

    try:
        if args.command== "train":

            # train on the GPU or on the CPU, if a GPU is not available
            num_of_gpu=2
            set_GPU(num_of_gpu)
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            # our dataset has two classes only - background and person
            num_classes = 5
            # use our dataset and defined transformations

            dataset_path = args.dataset
            dataset = PennFudanDataset(os.path.join(dataset_path, "Train"), get_transform(train=False))
            dataset_test = PennFudanDataset(os.path.join(dataset_path, "Validation"), get_transform(train=False))

            # split the dataset in train and test set

            indices = torch.randperm(len(dataset)).tolist()
            # dataset = torch.utils.data.Subset(dataset, indices[:-4])
            # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

            # define training and validation data loaders
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=True, num_workers=4,
                collate_fn=utils.collate_fn)

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1, shuffle=False, num_workers=4,
                collate_fn=utils.collate_fn)

            # get the model using our helper function
            model=get_model('resnet101',num_classes,args.weights)

            if args.weights !="" and args.weights.split("\\")[-1]!="coco_weights.pth":
                state_dict = torch.load(args.weights)
                model.load_state_dict(state_dict)

            # move model to the right device
            model.to(device)
            model = nn.DataParallel(model)

            # construct an optimizer
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.0001,
                                        momentum=0.9, weight_decay=0.0001)
            # and a learning rate scheduler
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=3,
                                                           gamma=0.1)

            # let's train it for 10 epochs
            num_epochs = 300

            save_weight = os.path.join(os.getcwd(),args.logs, "weights")
            if os.path.exists(save_weight) == False:
                os.makedirs(save_weight)

            writer_tensorboard = SummaryWriter(os.path.join(os.getcwd(), args.logs))
            for epoch in range(num_epochs):
                # train for one epoch, printing every 10 iterations
                train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
                                writer_tensorboard=writer_tensorboard)
                # update the learning rate
                lr_scheduler.step()
                # save model at each epoch
                torch.save(model.state_dict(), os.path.join(save_weight, "Faster_rcnn_" + str(epoch) + ".pth"))
                # evaluate on the test dataset
                evaluate_validation(model, data_loader_test, device, epoch, writer_tensorboard)

            writer_tensorboard.close()


        if args.command=="test":
            # Hard code
            img_path = r"D:\DL Configs\DLApplication\T220_6k_image\Dataset\Data_6k_usable\Dataset_6k_relabel\Validation\1_BBI_B_BBI_DARK_Defect_1 (1)(1).bmp"


            #dataset_path = r"D:\Dataset\Test_T220\draft\UK"
            dataset_path=r"D:\Dataset\Test_T220\Test_5k_T220"
            #dataset_path=r"D:\Dataset\Test_T220\Test_5k_T220_mini"


            # train on the GPU or on the CPU, if a GPU is not available
            num_of_gpu=1
            set_GPU(num_of_gpu)
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            pretrained_backbone = False
            backbone = resnet_fpn_backbone("resnet101", pretrained_backbone, trainable_layers=5)
            model = FasterRCNN(backbone, num_classes=5)

            if args.weights!="":
                state_dict = torch.load(args.weights,map_location='cuda:0')
                model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            detection_threshold = 0.0



            if args.score_statistic_name is not None:
                score_statistics=score_statistics(args.score_statistic_name+'.xlsx')

            for set in os.listdir(dataset_path):

                matrix=[0,0,0]
                for img_name in os.listdir(os.path.join(dataset_path,set)):

                    img_path=os.path.join(dataset_path,set,img_name)
                    if img_name.split(".")[-1]!="bmp":
                        continue
                    img = Image.open(img_path).convert("RGB")
                    img = img.crop((192,192,320,320))

                    img = ToTensor()(img)  # unsqueeze to add artificial first dimension
                    # image = Variable(image)

                    # for image,a in test_data_loader:
                    #     image_a=list(img.to(device) for img in image)
                    #
                    #     outputs=model(image_a)

                    images=[img.to(device)]

                    outputs = model(images)
                    outputs=outputs[0]

                    boxes = outputs['boxes'].data.cpu().numpy()
                    classes = outputs['labels'].data.cpu().numpy()
                    scores = outputs['scores'].data.cpu().numpy()

                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    classes = classes[scores >= detection_threshold]
                    scores = scores[scores >= detection_threshold]

                    red_box=[54,54,64,64]
                    results=[]

                    data_list= [0] * 8           #data_list to log score_statistics
                    data_list[0]=img_path.split("\\")[-1]
                    data_list[1]=img_path
                    data_list[2]=img_path.split("\\")[-2]

                    Score_Fail=0
                    Score_Pass=0
                    for index,box in enumerate(boxes):
                        iou = compute_overlaps(np.asarray([box]), np.asarray([red_box]))
                        if iou>=0.05:
                            if classes[index] in [1,2] and scores[index]>=0.05:
                               results.append("Fail")
                               if Score_Fail==0:
                                   Score_Fail=scores[index]
                            if classes[index] in [3,4] and scores[index]>=0.8:
                                results.append("Pass")
                                if Score_Pass == 0:
                                    Score_Pass = scores[index]

                    if "Fail" in results:
                        matrix[1]=matrix[1]+1
                        data_list[3]='Reject'
                        data_list[4]=Score_Fail
                    elif "Pass" in results:
                        matrix[0]=matrix[0]+1
                        data_list[3]='Pass'
                        data_list[4]=Score_Pass

                        # if os.path.exists(os.path.join(dataset_path,"UK_model16")) ==False:
                        #     os.makedirs(os.path.join(dataset_path,"UK_model16"))
                        # import shutil
                        # try:
                        #     shutil.copy(img_path,os.path.join(dataset_path,"UK",img_path.split("\\")[-1]))
                        # except:
                        #     print("------------------")


                    else:
                        matrix[2]=matrix[2]+1
                        data_list[3]='Unknown'

                        # import matplotlib
                        # matplotlib.use('tkagg')
                        # import matplotlib.pyplot as plt
                        #
                        # img = cv2.imread(img_path)
                        # img = img[192:320, 192:320]
                        # plt.imshow(img)
                        # plt.show()
                        # for box in boxes:
                        #     img = cv2.rectangle(img,
                        #                         (box[0], box[1]),
                        #                         (box[2], box[3]), (255, 0, 0), 1)
                        # plt.imshow(img)
                        # plt.show()


                    if args.score_statistic_name is not None:
                        score_statistics.add_row(data_list)

                    print(matrix)
                print(set,": ",matrix)

            if args.score_statistic_name is not None:
                score_statistics.close()


    except KeyboardInterrupt:
        print("-----------------------------------------------------------------------------------------------")
        writer_tensorboard.close()
        print("Interrupt!")

    print("Done!")
