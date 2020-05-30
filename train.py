from resnet import build_resnet
from model import fcn
from model_utils import lr_scheduler, restore_weights, save_model
from evaluate import evaluate
from pathlib import Path
import time


def build_model(input_shape=(1, 3, 480, 640),
                n_classes=4,
                weights_dir=None,
                pretrained_weight_fname=None,
                cuda=True):
    backbone = build_resnet(input_shape=input_shape)
    model = fcn(input_shape, backbone)
    if cuda:
        backbone = backbone.cuda()
        model = model.cuda()

    if pretrained_weight_fname is not None and weights_dir is not None:
        restore_weights(model, weights_dir, pretrained_weight_fname)

    return model, backbone

def train_loop(model, criterion, optimizer, scheduler,
               epochs=100, epoch_start=0, T_print=200, T_save=10):
    T_start = time.time()
    for epoch in range(epoch_start, epochs):
        running_loss = 0.0
        running_loss_mini = 0.0
        for i, (inputs, targets) in enumerate(trainloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            # remove the channel dimension
            targets = torch.argmax(targets, dim=1) # convert to (N, H, W)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_mini += loss.item()

            if i % T_print == T_print-1:
                T_end = time.time()
                print('%d-th minibatch\tloss: %f' % (i+1, running_loss/T_print), end='\t')
                print(T_end-T_start, 'secs elapsed')
                running_loss_mini = 0.0

        # call scheduler every epoch
        scheduler.step()

        # print statistics
        T_end = time.time()
        print('epoch %3d\tloss: %f' % (epoch + 1, running_loss))
        print(T_end-T_start, 'secs elapsed')

        # save
        if epoch % T_save == T_save-1:
            save_model(model, weights_dir, weight_fname

    print("Done!")


if __name__ == '__main__':
    # define parameters
    batch_size = 1
    shuffle = True
    epochs = 100
    T_save = 1
    T_print = 100
    num_classes = 4
    cuda = True

    pretrained_weight_fname = None
    weights_dir = 'weights/'
    data_dir = 'drinks/'
    train_gt_fname = 'segmentation_train.npy'
    test_gt_fname = 'segmentation_test.npy'

    # make path if not exist
    Path(weights_dir).mkdir(parents=True, exist_ok=True)

    # initialize dataloaders
    trainset = SemanticSegmentationDataset(data_dir, train_gt_fname, cuda=cuda)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    testset = SemanticSegmentationDataset(data_dir, test_gt_fname, cuda=cuda)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=shuffle)

    # initialize model
    channels, height, width = 3, 480, 640
    input_shape = (batch_size, channels, height, width)

    model, backbone = build_model(input_shape=input_shape,
                                  n_classes=num_classes,
                                  weights_dir=weights_dir,
                                  pretrained_weight_fname=pretrained_weight_fname,
                                  cuda=cuda)

    # initialize training parameters
    # based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    from torch import nn, optim
    criterion = nn.CrossEntropyLoss()   # categorical crossentropy
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)

    # train loop
    train_loop(model, criterion, optimizer, scheduler,
               epochs=epochs, epoch_start=epoch_start,
               T_print=T_print, T_save=T_save)

    # evaluation
    m_iou, m_pla = evaluate(model, testloader)
