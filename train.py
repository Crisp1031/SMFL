import torch
import sys
sys.path.append('../')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from DAIC.combination_dataset_mmd2023 import data_generator
from DAIC.MSK_lstm import MMFL
import torch.optim as optim
from ceLoss import celoss
import numpy as np
import argparse
from DAIC import loss
# 训练数据集和测试数据集加载
Batch_Size = 1
train_loader, test_loader = data_generator(1)
# torch.manual_seed(1)    # reproducible
from sklearn.metrics  import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from torch import nn
from tqdm import tqdm
# 通过命令行可以向程序中传入参数
parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
# 增加参数新信息
# 批处理文件个数
parser.add_argument('--batch_size', type=int, default=1, metavar='N',  #batch 的大小
                    help='batch size (default: 64)')
# 是否使用cuda
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
# 按照一定的概率将其暂时从网络中丢弃，防止过拟合
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.5)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
# 训练过程中数据将轮30次
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=9,   #卷积核的大小
                    help='kernel size (default0: 7)')
parser.add_argument('--levels', type=int, default=12,  #隐藏层的层数
                    help='# of levels (default: 8)')
# 日志更新
parser.add_argument('--log-interval', type=int, default=36, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3, #学习率大小
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',  #优化器设置
                    help='optimizer to use (default: Adam)')
parser.add_argument('--resume', type=str, default='',  #优化器设置
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,  #隐藏层的通道数  默认为：[25，25，25，25]
                    help='number of hidden units per layer (default: 25)')
# 随机种子1111
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
args.cuda = True
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

length = 100*1485                #mnist数据集中单张图片的尺寸28*28=784
batch_size = args.batch_size
n_classes = 2  # 分成两类
input_channels = 1485  # all(face2d):1417 all(face3d):1485 text:768 face:136 audio:513 face3d:204
epochs = args.epochs
steps = 0

print(args)
permute = torch.Tensor(np.random.permutation(length).astype(np.float64)).long()   #将长度数据转换为浮点数张量
# channel_sizes = [args.nhid] * args.levels + [128]   #channel_sizes=[25 25 25 25 25 25 25 25]
channel_sizes = [args.nhid] * args.levels  #channel_sizes=[25 25 25 25 25 25 25 25]

kernel_size = args.ksize
# model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
# 这里是选择模型，一个是DCNN，一个是TCN
model = MMFL()
print(model)
if args.cuda:
    model.cuda()
    permute = permute.cuda()
if args.resume:
    checpoint = torch.load(r'/mnt/Data/mashukui/TCN/DAIC/experiment/best/model_GPT_11_0.8286.pth')
    model.load_state_dict(checpoint)
    print(f'load from {args.resume}')

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)

def generate_tensor(value):
    if value == 0:
        return torch.tensor([1, 0], dtype=torch.float32)
    elif value == 1:
        return torch.tensor([0, 1], dtype=torch.float32)
    else:
        raise ValueError("Invalid value. Only 0 or 1 are allowed.")


def f1_score_my(y_true, y_pred, eps=1e-10):
    from sklearn.metrics import confusion_matrix
    M = confusion_matrix(y_true, y_pred)
    # precision = M[1, 1] / (M[0, 1] + M[1, 1] + eps)
    # recall = M[1, 1] / (M[1, 0]+M[1, 1] + eps)
    # f1 = 2*precision*recall / (precision+recall+eps)
    sensitivity = M[1, 1] / (M[1, 0] + M[1, 1]+eps)
    specificity = M[0, 0] / (M[0, 0] + M[0, 1]+eps)
    # f1 = 2*sensitivity*specificity/(sensitivity + specificity + eps)
    f1 = sensitivity + specificity
    return f1


def search_threshold(y_true, y_score):
    thresholds = []
    # cross loss
    for label, score in zip(y_true, y_score):
        label = np.array(label)
        score = np.array(score)

    # bce loss
    # for label, score in zip(y_true[np.newaxis, ...], y_score[np.newaxis, ...]):

        best_score = 0
        best_thresh = 0
        for t in np.linspace(0.01, 0.99, 99):
            f1score = f1_score_my(label, score > t)
            if f1score >= best_score:
                best_thresh = t
                best_score = f1score
        print('{0:.4f}:{1}'.format(best_score, best_thresh))
        thresholds.append(best_thresh)
    thresholds = np.array(thresholds)
    return thresholds


criterion = celoss()

def train(ep):
    global steps
    train_loss = 0
    model.train()
    print('train_loader=',len(train_loader))

    for batch_idx, (data, target, numberoflines, start_stop_time) in enumerate(tqdm(train_loader)):
        data_len=numberoflines
        # print('data_len=',data_len)
        if args.cuda:
            data = data.type(torch.FloatTensor)
            data, target = data.cuda(), target.cuda()
        #data = data.view(1, input_channels, seq_length)    #reshape x to (batch=64, time_step=1, input_size=784) [64, 1, 784]
        data = data.view(1, input_channels, data_len)#reshape x to (batch=1, time_step=136, input_size=numberoflines)

        if args.permute:
            data = data[:, :, permute]
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, start_stop_time)
        # output = model(data)
        # print(output)
        # loss = F.nll_loss(output, target, weight=torch.Tensor([1., 10.]).cuda())
        loss = criterion(target, output)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        batch_idx=batch_idx+1   # 由于batch_idx是从零开始的所以对其加一
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print(r'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} '.format(
                ep, batch_idx , len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/batch_idx))
        steps += 1
    print(r'Train Epoch: {} [{}/{} (100%)] Loss: {:.6f} '.format(
                 ep, batch_idx, len(train_loader.dataset),
                  train_loss.item()/batch_idx))

# 删去了测试集中的367，原因见E:\Python\project\TCN\DAIC\CLNF_features\test\read me.txt
# 模型的性能评估
def evaluate():
    model.eval()
    # model.test()
    test_loss = 0
    correct = 0
    target_result = []
    pred_result = []

    # with torch.no_grad():
        # print('test_loader=', len(test_loader))
    for data, target, numberoflines, start_stop_time in test_loader:
        data_len = numberoflines
        if args.cuda:
            data = data.type(torch.FloatTensor)
            data, target = data.cuda(), target.cuda()
        data = data.view(1, input_channels, data_len)
        # print('data:',data)
        # print('data.size:', len(data))
        if args.permute:
            data = data[:, :, permute]
        # data, target = Variable(data), Variable(target)

        output = model(data, start_stop_time)

        # print(output)
        # test_loss += F.nll_loss(output, target, weight=torch.Tensor([1., 10.]).cuda()).item()
        test_loss += criterion(target, output).item()
        # print(test_loss)
        pred = output.data.max(1, keepdim=True)[1]
        # print(correct)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        target_result.append(target.item())
        pred_result.append(pred.item())

    # acc = 100. * correct / len(test_loader.dataset)
    acc = accuracy_score(y_true=target_result, y_pred=pred_result)
    precision = precision_score(y_true=target_result, y_pred=pred_result)
    recall = recall_score(y_true=target_result, y_pred=pred_result)
    f1 = f1_score(y_true=target_result, y_pred=pred_result)
    print(confusion_matrix(y_true=target_result, y_pred=pred_result))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy:  ({:.2f}), '
          'precision: ({:.2f}), recall: ({:.2f}), f1: ({:.2f})\n'.format(
        test_loss, acc, precision, recall,
        f1, len(test_loader.dataset),
        ))
    return test_loss, acc

if __name__ == "__main__":
    loss_ = None
    best_loss = None
    best_acc = 0
    # torch.backends.cudnn.enabled = False
    for epoch in range(1, epochs +1):


        # loss_, acc = evaluate()
        train(epoch)
        loss_, acc = evaluate()

        if best_loss is None:
            best_loss = loss_
            best_acc = acc
            torch.save(model.state_dict(), f'./experiment/all/model_tcn-sentence_{epoch}.pth')
        elif loss_ < best_loss or acc > best_acc:
            best_loss = loss_
            best_acc = acc
            torch.save(model.state_dict(), './experiment/all/model_tcn-sentence_{}_{:.4f}.pth'.format(epoch, acc))
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
               param_group['lr'] = lr

