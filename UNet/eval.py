import torch
import torch.nn.functional as F
import numpy as np
from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / (i + 1)



import numpy as np
def get_iou(pred_image,true_image):
    '''
    计算两个Mask的交并比IOU
    :param pred_image:
    :param true_image:
    :return:
    '''
    pred_image = (pred_image > 0)
    true_image = (true_image > 0)
    i = (true_image * pred_image).sum()#交集
    u = (true_image + pred_image).sum()#并集
    o=i / u if u != 0 else u
    return o

def get_batch_iou_mean(pred_images, true_images):
    '''
    计算平均IOU
    :param pred_images:
    :param true_images:
    :return:
    '''
    num_images = len(true_images)
    scores = np.zeros(num_images)
    for i in range(num_images):
        pred_image = np.where(pred_images[i] > 0.5, 1, 0)
        true_image = np.where(true_images[i] > 0.5, 1, 0)
        scores[i]=get_iou(pred_image,true_image)
    return scores.mean()

iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
def get_batch_AP(pred_images, true_images):
    '''
    get  average precision(AP),The metric sweeps over a range of IoU thresholds,
    at each point calculating an average precision value.
    The threshold values range from 0.5 to 0.95 with a step size of 0.05: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95).
    例如，在阈值为0.5时，如果预测mask与真实mask的交集大于0.5，则预测mask被认为是“命中”,即为TP
    :param pred_images: 1 - mask, 0 - background
    :param true_images: 1 - mask, 0 - background
    :return:
    '''
    num_images = len(true_images)
    scores = np.zeros(num_images)
    for i in range(num_images):
        # 先将输入的图像二值化，
        pred_image = np.where(pred_images[i] > 0.5, 1, 0)
        true_image = np.where(true_images[i] > 0.5, 1, 0)
        # 计算IOU
        iou = get_iou(pred_image,true_image)
        m=iou_thresholds <= iou
        scores[i] = m.mean()
    AP=scores.mean()
    return AP


if __name__=="__main__":
    data1 = np.arange(0, 9)
    data1 = data1.reshape([1,3, 3])/10

    data2 = np.arange(1, 10)
    data2 = data2.reshape([1,3, 3])/10
    print(data1)
    print(data2)

    print(get_iou(data1, data2))
    print(get_batch_AP(data1, data2))