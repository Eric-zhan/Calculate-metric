from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def calculate_confusion_matrix(gt_img, pred_img, n_class):
    """如果np.vstack里是（gt，pred），那么最后混淆矩阵里，
        横坐标是gt，纵坐标是pred，eg. (0, 1) 0是gt，1是pred
        所以FN是confusion_matrix[index, :] - TP
        FP 是confusion_matrix[:, index] - TP"""
    replace_matrix = np.vstack((gt_img.flatten(), pred_img.flatten())).T
    confusion_matrix, _ = np.histogramdd(replace_matrix,
                                         bins=(n_class, n_class),
                                         range=[(0, n_class), (0, n_class)])
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


class Calculate(object):
    """accuracy: TP / (TP + FN)
        IoU: TP / (TP + FN + FP)
        dice: 2 * TP / ( 2 * TP + FN + FP)
        fwIoU: ((TP + FN) / (TP + TN + FP + FN)) * TP / (TP + FN + FP)"""

    def __init__(self, confusion_matrix):

        self.ious_list = []  # include background
        self.dice_list = []
        self.fwious_list = []
        self.accuracy_list = []
        for index in range(confusion_matrix.shape[0]):
            TP = confusion_matrix[index, index]
            FP = confusion_matrix[:, index].sum() - TP
            FN = confusion_matrix[index, :].sum() - TP

            union = TP + FP + FN
            if union == 0:
                iou = 0
                dice = 0
                fwiou = 0
                accuracy = 0
            else:
                iou = float(TP) / union
                dice = 2 * float(TP) / (TP + union)
                fwiou = ((TP + FN) / (confusion_matrix.sum())) * iou
                accuracy = float(TP) / (TP + FP)
            self.ious_list.append(iou)
            self.dice_list.append(dice)
            self.fwious_list.append(fwiou)
            self.accuracy_list.append(accuracy)

    def accuracy(self):
        return self.accuracy_list

    def IoU(self):
        return self.ious_list

    def fwIoU(self):
        return np.sum(self.fwious_list)

    def dice(self):
        return self.dice_list


if __name__ == '__main__':
    import glob

    dices_list = []
    mious_list = []
    fwious_list = []
    accuracy_list = []
    # gt_img = cv2.imread(r'D:\BUAA\code\others\robot-surgery-segmentation\Unet\450'
    #                     r'\data\test_mask2\Shot_202101242201400709_5.png', 0)
    # gt_img = cv2.imread(r'D:\BUAA\code\others\robot-surgery-segmentation\Unet\450'
    #                     r'\data\test_mask2\Shot_202101242201400709_5.png', 0)
    # gt_img[(gt_img == 128)] = 1
    # gt_img[(gt_img == 255)] = 2
    # pred_img = cv2.imread(r'D:\BUAA\code\others\robot-surgery-segmentation\Unet\450'
    #                       r'\results\data\Shot_202101242201400709_5.png', 0)
    # pred_img = cv2.imread(r'D:\BUAA\code\others\robot-surgery-segmentation\Unet\450'
    #                       r'\results\data\Shot_202101242201400709_5.png', 0)
    # pred_img[(pred_img == 128)] = 1
    # pred_img[(pred_img == 255)] = 2
    # b = calculate_confusion_matrix(gt_img, pred_img, 3)
    # print(b)
    # calculate = Calculate(b)
    # print(calculate.IoU())
    # print(np.mean(calculate.IoU()[1:]))

    pred_dirs = glob.glob(r'D:\BUAA\code\others\robot-surgery-segmentation\Unet\450\results\data\*')
    for pred_dir in pred_dirs:
        pred_name = pred_dir.split('\\')[-1]
        gt_img = cv2.imread(fr'D:\BUAA\code\others\robot-surgery-segmentation\Unet\450\data\test_mask2/{pred_name}', 0)
        # pred_img = cv2.imread(fr'D:\BUAA\code\others\robot-surgery-segmentation\Unet16\1000-50\50\results\data/{pred_name}', 0)
        pred_img = cv2.imread(
            fr'D:\BUAA\code\others\robot-surgery-segmentation\Unet\450\results\data/{pred_name}', 0)
        pred_index1 = (pred_img == 128)
        pred_index2 = (pred_img == 255)
        pred_img[pred_index1] = 1
        pred_img[pred_index2] = 2

        gt_index1 = (gt_img == 128)
        gt_index2 = (gt_img == 255)
        gt_img[gt_index1] = 1
        gt_img[gt_index2] = 2

        b = calculate_confusion_matrix(gt_img, pred_img, 3)
        calculate = Calculate(b)
        # print(calculate.accuracy())

        dices_list.append(np.mean(calculate.dice()[1:]))
        mious_list.append(np.mean(calculate.IoU()[1:]))
        fwious_list.append(np.sum(calculate.fwIoU()))
        accuracy_list.append(np.mean(calculate.accuracy()[1:]))
    print('dice:', np.mean(dices_list))
    print('miou:', np.mean(mious_list))
    print('fwious:', np.mean(fwious_list))
    print('accuracy:', np.mean(accuracy_list))
