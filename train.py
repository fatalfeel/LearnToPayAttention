import os
import random
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from model import AttnVGG
from utilities import *

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR100")
parser.add_argument("--img_size", type=int, default=32, help="image size")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--isAttention", type=str2bool, default=True, help='turn on attention')
#parser.add_argument("--attn_mode", type=str, default="after", help='insert attention modules before OR after maxpooling layers')
parser.add_argument('--attn_before', type=str2bool, default=True, help='insert attention modules before OR after maxpooling layers')
parser.add_argument("--normalize_attn", type=str2bool, default=True, help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument('--save_model_path', type=str, default='checkpoints/', help='save model')
parser.add_argument("--save_test_path", type=str, default="checkpoints/results/", help='test files path ')
parser.add_argument("--save_final_path", type=str, default="pretrained/", help='lastest model save path')
parser.add_argument("--images_row", type=int, default=4, help='how many images in one row')
parser.add_argument('--cuda', default=False, type=str2bool)

opts    = parser.parse_args()
device  = torch.device("cuda:0" if opts.cuda else "cpu")
kwargs  = {'num_workers': 1, 'pin_memory': True} if opts.cuda else {}

def _worker_init_fn_(worker_id):
    torch_seed = torch.initial_seed() + worker_id
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

if __name__ == "__main__":
    if not os.path.exists(opts.save_test_path):
        os.makedirs(opts.save_test_path)

    if not os.path.exists(opts.save_final_path):
        os.makedirs(opts.save_final_path)

    print('\nloading the dataset ...\n')
    #im_size = 32
    transform_train = transforms.Compose([transforms.RandomCrop(opts.img_size, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    trainset    = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform_train)
    testset     = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch_size, shuffle=True, worker_init_fn=_worker_init_fn_, **kwargs)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=opts.batch_size, shuffle=False, **kwargs)

    if opts.isAttention:
        print('\nturn on attention ...\n')
    else:
        print('\nturn off attention ...\n')

    # (grid attn only supports "before" mode)
    '''if opts.attn_before == 'before':
        print('\npay attention before maxpooling layers...\n')
        modelPA = AttnVGG_before(img_size=opts.img_size,
                             num_classes=100,
                             attention=opts.isAttention,
                             normalize_attn=opts.normalize_attn,
                             init='xavierUniform')
    elif opts.attn_before == 'after':
        print('\npay attention after maxpooling layers...\n')
        modelPA = AttnVGG_after(img_size=opts.img_size,
                            num_classes=100,
                            attention=opts.isAttention,
                            normalize_attn=opts.normalize_attn,
                            init='xavierUniform')
    else:
        raise NotImplementedError("Invalid attention mode!")'''

    modelPA = AttnVGG(  img_size        = opts.img_size,
                        num_classes     = 100,
                        isAttention     = opts.isAttention,
                        normalize_attn  = opts.normalize_attn,
                        attn_before     = opts.attn_before,
                        init            = 'xavierUniform'  ).to(device)

    loss_ceLoss = nn.CrossEntropyLoss()

    ### optimizer
    optimizer   = optim.SGD(modelPA.parameters(), lr=opts.lr, momentum=0.9, weight_decay=5e-4)
    lr_lambda   = lambda epoch : np.power(0.5, int(epoch/25))
    scheduler   = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    num_aug                 = 3
    step                    = 0
    running_avg_accuracy    = 0
    images_disp             = []
    for epoch in range(opts.epochs):
        print('\nstart training ...\n')
        images_disp.clear()
        modelPA.train()
        for aug in range(num_aug):
            for i, (train_data, train_labels) in enumerate(trainloader, 0):
                train_data      = train_data.to(device)
                train_labels    = train_labels.to(device)

                if (aug == 0) and (i == 0): # archive images in order to save to logs
                    #images_disp.append(train_data[0:36,:,:,:])
                    images_disp.append(train_data)

                # forward
                train_pred, __, __, __ = modelPA(train_data)
                loss = loss_ceLoss(train_pred, train_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # display results
                if i % 100 == 0:
                    val_pred, __, __, __ = modelPA(train_data)
                    predict = torch.argmax(val_pred, 1)
                    total   = train_labels.size(0)
                    correct = torch.eq(predict, train_labels).sum().double().item()
                    accuracy = correct / total
                    running_avg_accuracy = 0.9*running_avg_accuracy + 0.1*accuracy
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%" %
                            (epoch,
                             aug,
                             num_aug-1,
                             i,
                             len(trainloader)-1,
                             loss.item(),
                             (100*accuracy),
                             (100*running_avg_accuracy)))
                step += 1

        #torch.save(modelPA.state_dict(), os.path.join(opts.save_model_path, 'modelPA.pth'))
        torch.save(modelPA.state_dict(), os.path.join(opts.save_model_path, 'modelPA%d.pth' % epoch))
        # adjust learning rate
        scheduler.step()

        print('\nstart test accuracy\n')
        total   = 0
        correct = 0
        modelPA.eval()
        with torch.no_grad():
            # log scalars
            for i, (test_data, test_labels) in enumerate(testloader, 0):
                test_data   = test_data.to(device)
                test_labels = test_labels.to(device)

                if i == 0: # archive images in order to save to logs
                    #images_disp.append(train_data[0:36, :, :, :])
                    images_disp.append(test_data)

                test_pred, __, __, __ = modelPA(test_data)
                predict  = torch.argmax(test_pred, 1)
                total   += test_labels.size(0)
                correct += torch.eq(predict, test_labels).sum().double().item()

            #writer.add_scalar('test/accuracy', correct/total, epoch)
            print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100*correct/total))

            # log images
            print('\nsave attention maps ...\n')
            I_train = utils.make_grid(images_disp[0], nrow=opts.images_row, normalize=True, scale_each=True)
            saveimg(I_train, opts.save_test_path + 'train_epoch%d.jpg' % epoch)

            I_test = utils.make_grid(images_disp[1], nrow=opts.images_row, normalize=True, scale_each=True)
            saveimg(I_test, opts.save_test_path + 'test_epoch%d.jpg' % epoch)

            if opts.isAttention:
                if opts.attn_before: # base factor
                    min_up_factor = 1
                else:
                    min_up_factor = 2
                # sigmoid or softmax
                if opts.normalize_attn:
                    vis_fun = visualize_attn_softmax
                else:
                    vis_fun = visualize_attn_sigmoid
                # training data
                __, c1, c2, c3 = modelPA(images_disp[0])
                if c1 is not None:
                    attn1 = vis_fun(I_train, c1, up_factor=min_up_factor, nrow=opts.images_row)
                    saveimg(attn1, opts.save_test_path + 'train_attn1_epoch%d.jpg' % epoch)

                if c2 is not None:
                    attn2 = vis_fun(I_train, c2, up_factor=min_up_factor * 2, nrow=opts.images_row)
                    saveimg(attn2, opts.save_test_path + 'train_attn2_epoch%d.jpg' % epoch)

                if c3 is not None:
                    attn3 = vis_fun(I_train, c3, up_factor=min_up_factor * 4, nrow=opts.images_row)
                    saveimg(attn3, opts.save_test_path + 'train_attn3_epoch%d.jpg' % epoch)

                # test data
                __, c1, c2, c3 = modelPA(images_disp[1])
                if c1 is not None:
                    attn1 = vis_fun(I_test, c1, up_factor=min_up_factor, nrow=opts.images_row)
                    saveimg(attn1, opts.save_test_path + 'test_attn1_epoch%d.jpg' % epoch)

                if c2 is not None:
                    attn2 = vis_fun(I_test, c2, up_factor=min_up_factor * 2, nrow=opts.images_row)
                    saveimg(attn2, opts.save_test_path + 'test_attn2_epoch%d.jpg' % epoch)

                if c3 is not None:
                    attn3 = vis_fun(I_test, c3, up_factor=min_up_factor * 4, nrow=opts.images_row)
                    saveimg(attn3, opts.save_test_path + 'test_attn3_epoch%d.jpg' % epoch)

    torch.save(modelPA.state_dict(), os.path.join(opts.save_final_path, 'modelPA_final.pth'))
    