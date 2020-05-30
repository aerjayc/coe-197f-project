import torch
from torch.autograd import Variable
import numpy as np


def get_iou(gts, segmentation, n_classes=4):
    iou = 0
    i_iou, n_masks = 0, 0
    eps = np.finfo(float).eps   # minimum positive float value
    for i in range(n_classes):
        gt, mask = gts[i,:,:].cpu(), segmentation[i,:,:].cpu().detach()
        # skip current class if it doesn't appear in image
        if torch.sum(gt) < eps:
            continue

        intersection = torch.sum(mask * gt)
        union = torch.ceil((mask + gt) / 2.0)
        union = torch.sum(union)
        if union > eps:
            iou = intersection / union
            i_iou += iou
            n_masks += 1

    if n_masks:
        i_iou /= n_masks    # average iou for the entire image
    else:
        i_iou = 0

    return i_iou

def get_pla(gts, segmentation):
    return 100 * (gts.cpu() == segmentation.cpu().detach()).all(axis=1).float().mean()

def get_batch_iou(gts, segmentation, n_classes=4):
    s_iou = 0
    for gt, seg in zip(gts, segmentation):
        s_iou += get_iou(gt, seg, n_classes=n_classes)

    return s_iou / len(gts)

def get_batch_pla(gts, segmentation):
    s_pla = 0
    for gt, seg in zip(gts, segmentation):
        s_pla += get_pla(gt, seg)

    return s_pla / len(gts)

def to_categorical(tensor, n_classes=None, batched=True):
    cuda = tensor.is_cuda
    if batched: # N, C, ...
        dim=1
    else:       # C, ...
        dim=0
    tensor = torch.argmax(tensor, dim=dim)   # ... or N, ...

    if n_classes is None:
        n_classes = torch.max(tensor) + 1

    indices = np.arange(n_classes)[None,...].T  # indices, reshaped to
                                                # (1, n_classes)
    comparee = np.tile(indices, np.prod(tensor.shape[dim:]))
    comparee = comparee.reshape(n_classes, *tensor.shape[dim:])
    # creates tensor with shape (num_classes, H, W) where each array in
    # dim 1 is composed solely of a single index, as in
    # ```comparee = np.array([ [[0,0,0],
    #                           [0,0,0]],
    #                          [[1,1,1],
    #                           [1,1,1]] ])```
    # for tensor.shape == (2, 2, 3) and num_class == 2

    if batched:
        tensor = np.array(tensor[:,None,...]) #  N , (C), H, W
        comparee = comparee[None,...]                  # (N),  C , H, W
    else:
        tensor = np.array(tensor[None,...])   # (C), H, W
        # comparee = comparee                 #  C , H, W
    categorical = np.equal(tensor, comparee)
    del tensor      # don't waste space
    del comparee
    categorical = torch.from_numpy(categorical).type(torch.bool)

    if cuda:
        categorical = categorical.cuda()
    
    return categorical  # N?, C, H, W
    
def segment_objects(model, images, n_classes=4):
    # image should be output by DataLoader
    images = Variable(images, volatile=True)    # to reduce memory usage
    outputs = model(images).cpu().detach()
    segmentation = to_categorical(outputs, n_classes=n_classes)

    return segmentation # boolean mask of shape (N,C,H,W)

def evaluate(model, testloader, max_n_test=None, T_print=1):
    s_iou = 0
    s_pla = 0
    # evaluate iou per test image
    for i, (imgs, gts) in enumerate(testloader):
        segmentation = segment_objects(model, imgs)

        # free up memory
        del imgs
        torch.cuda.empty_cache()

        s_iou += get_batch_iou(gts, segmentation)
        s_pla += get_batch_pla(gts, segmentation)

        # free up memory
        del segmentation
        torch.cuda.empty_cache()

        if i % T_print == T_print-1:
            print('%d) IoU = %f\tpla = %f' % (i, s_iou / (i+1), s_pla / (i+1)))

        if max_n_test:
            if i+1 >= max_n_test:
                break

    n = len(testloader)
    m_iou = s_iou / n
    m_pla = s_pla / n

    print("Mean IoU: %f" % (m_iou,))
    print("Mean Pixel level accuracy: %f" % (m_pla,))

    return m_iou, m_pla


def show_sample_segmentation(model, testloader):
    for img, gt in testloader:
        break

    from matplotlib import pyplot as plt

    plt.imshow(img[0].permute(1,2,0).cpu())
    plt.show()
    plt.imshow(gt[0,1:,:,:].permute(1,2,0).cpu()*255)
    plt.show()

    semgentation = segment_objects(model, img)
    del img
    torch.cuda.empty_cache()

    plt.imshow(segmentation[0,1:,:,:].permute(1,2,0)*255)
    plt.show()

    iou = get_iou(gt[0], segmentation[0])
    print("IoU:", iou)

    pla = get_pla(gt[0], segmentation[0])
    print("pla:", pla)

    del segmentation
    torch.cuda.empty_cache()
