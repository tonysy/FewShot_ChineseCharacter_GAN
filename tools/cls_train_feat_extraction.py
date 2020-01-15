import torch
import time
import os
from tqdm import tqdm
import shutil

from torch import nn

from pccgan.models.classification_model import resnet18
from pccgan.util.util import AverageMeter, accuracy
from pccgan.options.train_options import TrainOptions
from pccgan.data import create_dataset


opt = TrainOptions().parse()   # get training options

model = resnet18(num_classes=57)
model = nn.DataParallel(model).cuda()

# import pdb; pdb.set_trace()
state_dict = torch.load(os.path.join(opt.checkpoints_dir,'model_best.pth.tar'))
model.load_state_dict(state_dict['state_dict'])

train_loader = create_dataset(opt) 

lr = 0.1
momentum = 0.9
weight_decay = 5e-4
schedules = [10,20,30]
lr_decay_gamma = 0.1
num_epochs = 36

# Save 
interval = 5

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


# test_acc = test()
# print("Test first time:{}".format(test_acc))

# best_test_acc = 0.0

train_loader.dataset.opt.phase = 'train'
progress = tqdm(enumerate(train_loader), desc="Loss: ", total=len(train_loader.dataloader))
model.train()

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

end = time.time() 

fc_feats = torch.zeros(
    57,
    512
).cuda()
# fc_feats_num = dict(zip(list(range(57)), [0]*57))
fc_feats_num = torch.zeros(
    57
).cuda()

for batch_idx, data in progress:
    
    data_time.update(time.time() - end)
    # import pdb; pdb.set_trace()

    images = data['A'].cuda()
    labels = data['cls_label'].cuda()
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        predicts, last_fc_feats = model(images)

        loss = criterion(predicts, labels)

        # import pdb; pdb.set_trace()
        for i in range(images.size(0)):
            # import pdb; pdb.set_trace()
            fc_feats[labels.data[i]] += last_fc_feats[i]
            fc_feats_num[labels.data[i]] += 1
    
    # import pdb; pdb.set_trace()
    # label_set = list(set(labels.view(-1).data.item()))
    # for label in label_set:

    # predicts, fc_feats = model(images)

    loss = criterion(predicts, labels)

    prec1, prec5 = accuracy(predicts.data, labels.data, topk=(1, 5))
    losses.update(loss.data.item(), images.size(0))
    top1.update(prec1.item(), images.size(0))
    top5.update(prec5.item(), images.size(0))

    # compute gradient and do SGD step
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    progress.set_description("Loss: {:.4f}, Top-1:{:.4f}, Top-5:{:.4f}".format(losses.avg, top1.avg, top5.avg))

# import pdb; pdb.set_trace()
    
mean_feat = fc_feats / fc_feats_num.view(-1, 1)
torch.save(mean_feat.data, os.path.join(opt.checkpoints_dir, "mean_fc_feat.pt"))

mean_feat_score = model.module.fc(mean_feat)

prec1, prec5 = accuracy(mean_feat_score.data, torch.Tensor(list(range(57))).cuda().data,topk=(1, 5))
print("Mean acc:Top-1:{:.4f},Top-5:{:.4f}".format(top1.avg, top5.avg))
# for epoch in range(num_epochs):
#     adjust_learning_rate(optimizer, epoch)
#     print('Epoch:{}/{}, LR:{}'.format(epoch, num_epochs, state['lr']))
#     total_loss = 0
#     train_loader.dataset.opt.phase = 'train'

#     progress = tqdm(enumerate(train_loader), desc="Loss: ", total=len(train_loader.dataloader))

#     model.train()
    
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     end = time.time() 
#     for batch_idx, data in progress:
        
#         data_time.update(time.time() - end)
#         # import pdb; pdb.set_trace()

#         images = data['A'].cuda()
#         labels = data['cls_label'].cuda()
#         # import pdb; pdb.set_trace()
#         with torch.no_grad():
#             predicts, fc_feats = model(images)

#             loss = criterion(predicts, labels)

#         # predicts, fc_feats = model(images)

#         loss = criterion(predicts, labels)

#         prec1, prec5 = accuracy(predicts.data, labels.data, topk=(1, 5))
#         losses.update(loss.data.item(), images.size(0))
#         top1.update(prec1.item(), images.size(0))
#         top5.update(prec5.item(), images.size(0))

#         # compute gradient and do SGD step
#         # optimizer.zero_grad()
#         # loss.backward()
#         # optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         progress.set_description("Loss: {:.4f}, Top-1:{:.4f}, Top-5:{:.4f}".format(losses.avg, top1.avg, top5.avg))
        # import pdb; pdb.set_trace()
    
    # test_acc = test()
    # if test_acc > best_test_acc:
    #     best_test_acc = test_acc
    #     save_checkpoint({
    #                 'epoch': epoch + 1,
    #                 'state_dict': model.state_dict(),
    #                 'acc': top1.avg,
    #                 'best_acc': best_test_acc,
    #                 'optimizer' : optimizer.state_dict(),
    #             }, is_best=True, checkpoint=opt.checkpoints_dir, filename='checkpoint_{}.pth'.format(epoch+1))
    #     print("Save The Best Model for epoch :{}, with Accuracy:{:.4f}".format(epoch+1, best_test_acc))


    # if epoch % interval == 0:
    #     save_checkpoint({
    #                 'epoch': epoch + 1,
    #                 'state_dict': model.state_dict(),
    #                 'acc': top1.avg,
    #                 # 'best_acc': best_acc,
    #                 'optimizer' : optimizer.state_dict(),
    #             }, is_best=False, checkpoint=opt.checkpoints_dir, filename='checkpoint_{}.pth'.format(epoch+1))
    #     print("Save Model for epoch :{}".format(epoch+1))
 