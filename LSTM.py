import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.optim import lr_scheduler
import numpy as np
import time


# Parameters setting
# 当进行10分类时，由于数据集较小，可以选用较小的Batch_size。
BATCH_SIZE = 500
EPOCH = 200
use_gpu = torch.cuda.is_available()
# 需要分类的种类数  10/107
NumOfCategory = 107
# 每个学生采样的sample数
NumofSamples = 300

# 定义了一个单向LSTM网络，隐藏层为100个节点，通过线性分类器分为10类
class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTM, self).__init__()
        self.n_layer = n_layer
        # dimensions of the input feature
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        # self.out = nn.Linear(hidden_dim, n_class)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


# 加载经过数据预处理所得到的RHS文件
def LoadinRHS(path='Code/SampleRHS.txt'):
    f = open(path, 'r')
    a = f.read()
    SampleRHS = eval(a)
    f.close()
    return SampleRHS


# 文件名所提供的Label为字符串格式，该函数通过一个字典实现字符串形式标签和one-hot标签的对应
def convertLabel2Num(Train_RHS_Label_Sample):
    Label_dict = {}
    num = -1
    Label_return = []
    for ind in Train_RHS_Label_Sample:
        if ind in Label_dict:
            continue
        else:
            num += 1
            Label_dict[ind] = num
    for ind in Train_RHS_Label_Sample:
        Label_return.append(Label_dict[ind])
    return Label_return, Label_dict


# LSTM会返回每一个RHS sample的分类结果，由于每个学生有NumofSamples个sample，因此可以进行投票，返回最终判断
def Vote(pred):
    # print(len(pred))
    # print(type(pred))
    pred_return = []

    for i in range(int(len(pred)/NumofSamples)):
        # print(i)
        temp = pred[i * NumofSamples+1: (i + 1) * NumofSamples]
        counts = np.bincount(temp)
        # print(temp)
        # print(counts)
        # 返回众数
        ind = np.argmax(counts)
        # print(ind)
        list_ind = ind * np.ones((1, NumofSamples))
        pred_return.extend(list_ind.tolist()[0])
    return pred_return


# 训练过程分为两步： phase - Train/Test
def Train(model, criterion, optimizer, scheduler, num_epochs, Validation_Label_List):
    for epoch in range(num_epochs):
        Train_step = LenTrain / BATCH_SIZE
        print(' ')
        print('EPOCH: ' + str(epoch))
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                # 如果是Train phase 检查是否调整学习率 并开启梯度下降更新
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                # 如果是Test phase 关闭梯度下降更新
                model.train(False)  # Set model to evaluate mode

            running_corrects = 0
            AllTrain = 0
            roundnum = 0

            if phase == 'Train':
                # 如果是Train 使用加载训练数据的dataloader
                loader = Train_loader
            else:
                accum = []
                # 如果是Test 使用加载验证数据的dataloader
                loader = Validation_loader

            for data in loader:
                if phase == 'Train':
                    print(str(roundnum) + 'th Batch, All Batches: ' + str(int(Train_step)))
                # print(LenTrain)
                roundnum += 1
                # print(roundnum)
                inputs, labels = data
                inputs = inputs.view(-1, 100, 2)
                if use_gpu:
                    inputs = Variable(inputs.cuda().type(torch.cuda.FloatTensor))
                    labels = Variable(labels.cuda().long())
                else:
                    inputs, labels = Variable(inputs).type(torch.FloatTensor), Variable(labels.long())

                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                if phase == 'Test':
                    accum.extend(preds.cpu().numpy())

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

                # print(preds)

                # Analyze the accuracy
                for i in range(len(preds)):
                    AllTrain += 1
                    if preds[i] == labels.data[i]:
                        running_corrects += 1

            # 统计Loss和正确率
            if phase == 'Train':
                epoch_acc = running_corrects / LenTrain
            else:
                running_corrects = 0
                accum = Vote(accum)
                # print(type(accum))
                # print(type(Validation_Label_List))
                for i in range(len(accum)):
                    # print('accum[i]: ' + str(accum[i]))
                    # print('Validation_Label_List[i]: ' + str(Validation_Label_List[i]))
                    if accum[i] == Validation_Label_List[i]:
                        running_corrects += 1
                epoch_acc = running_corrects / LenValidation
            print(phase + ' accuracy: ' + str(epoch_acc))
            # print('All: ' + str(AllTrain))
            print(phase + ' correct: ' + str(running_corrects))

    return model

#
#
# lstm = nn.LSTM(3, 3)
# torch.manual_seed(1)

# Define Input data：  (batch_size, seq_len, dims)
print('Loading. Please wait... It may take 2-3 minutes')
since = time.time()
SampleRHS = LoadinRHS('SampleRHS_107.txt')
time_elapsed = time.time() - since
print('Loadin RHS: ' + str(time_elapsed) + 's')

since = time.time()
Train_RHS_Sample = np.array(SampleRHS['Train_RHS_Sample'])
Train_RHS_Label_Sample = SampleRHS['Train_RHS_Label_Sample']
time_elapsed = time.time() - since
print('Extract list: ' + str(time_elapsed) + 's')

since = time.time()
Train_Label_List = convertLabel2Num(Train_RHS_Label_Sample)
time_elapsed = time.time() - since
print('Convert 2 list: ' + str(time_elapsed) + 's')

Train_Label = np.transpose(np.array(Train_Label_List))
# 把数据放在数据集中并以DataLoader送入网络训练
Train_dataset = Data.TensorDataset(torch.from_numpy(Train_RHS_Sample), torch.from_numpy(Train_Label))
Train_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=Train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# Design Test Data Loader
Validation_RHS_Sample = np.array(SampleRHS['Validation_RHS_Sample'])
Validation_RHS_Label_Sample = SampleRHS['Validation_RHS_Label_Sample']
Validation_Label_List = convertLabel2Num(Validation_RHS_Label_Sample)
Validation_Label = np.transpose(np.array(Validation_Label_List))
# 把数据放在数据集中并以DataLoader送入网络训练
Validation_dataset = Data.TensorDataset(torch.from_numpy(Validation_RHS_Sample), torch.from_numpy(Validation_Label))
Validation_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=Validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)
LenValidation = len(Validation_RHS_Label_Sample)
LenTrain = len(Train_RHS_Label_Sample)


# Hidden_dim is determined by the needs
model = LSTM(in_dim=2, hidden_dim=100, n_layer=1, n_class=NumOfCategory)
# print(model)
if use_gpu:
    model = model.cuda()

# 一下损失函数、优化器和学习率调整都可以修改
criterion = torch.nn.CrossEntropyLoss()
# This weight_decay parameter was set to prevent overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)# weight_decay=1e-8)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

print('Training process started')
LSTM_model = Train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=EPOCH, Validation_Label_List=Validation_Label_List)
torch.save(LSTM_model, "Model_LSTM.pth")

