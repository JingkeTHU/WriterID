import numpy as np
import matplotlib.pyplot as plt
import glob

def LoadFile(path):
    # Input: path of .npy file to load
    # Output: numpy array load from .npy file
    File = np.load(path)
    return File

def GeneratePath(path='Data10/*/*.npy'):
    # Input: No need to care
    # Output: A dictionary including path to each .npy file under the rootpath and its label
    files = glob.glob(path)
    Train_Path, Train_Label, Validation_Path, Validation_Label = [], [], [], []
    for num in range(len(files)):
        if files[num].split('/')[1] == 'Validation_with_labels' or files[num].split('/')[1] == 'Validation Characters10':
            continue
        else:
            if files[num].split('/')[1] == 'Train':
                Train_Path.append(files[num])
                Train_Label.append(files[num].split('/')[2].split('.')[0])
            if files[num].split('/')[1] == 'Validation':
                Validation_Path.append(files[num])
                Validation_Label.append(files[num].split('/')[2].split('.')[0])
    Dataset = dict(Train_Path=Train_Path, Train_Label=Train_Label,
                   Validation_Path=Validation_Path, Validation_Label=Validation_Label)
    # print(Dataset)
    return Dataset

def SwitchAxes(Word, index):
    # Input: Word是单个字放在以原点为顶点时每个点的坐标点  index当将100个字组合成10*10的页时，当前字的位置
    # Output: 按照一个字600*600的尺寸，计算出当前字的坐标
    Scatter_x, Scatter_y = [], []
    row = int(index / 10)
    line = index%10
    for stroke in range(len(Word)):
        # print(stroke)
        for point in range(len(Word[stroke])):
            # Width and Height of the input window are 600
            Scatter_x.append(Word[stroke][point][0] + line*600)
            Scatter_y.append(600 - Word[stroke][point][1] + row*600)
    return Scatter_x, Scatter_y

def StitchWords(File, index):
    # Input: File 对应于某个学生的包含其所有字坐标的列表  index 取第几组100个字合为一页
    # Output: 返回出一整页的各点坐标
    AllScatter_x, AllScatter_y = [], []
    for word in range(index, index+100):
        Scatter_x, Scatter_y = SwitchAxes(File[word], word)
        AllScatter_x.extend(Scatter_x)
        AllScatter_y.extend(Scatter_y)
    return AllScatter_x, AllScatter_y


def CreateRHS(Dataset):
    # Input:
    # Output:
    # Integrate words into pages and return point and RHS
    Train_Path = Dataset['Train_Path']
    Train_Label = Dataset['Train_Label']
    Validation_Path = Dataset['Validation_Path']
    Validation_Label = Dataset['Validation_Label']
    Train_RHS, Train_Point, Train_RHS_Label, Validation_RHS, Validation_Point, Validation_RHS_Label = [],[],[],[],[],[]
    # Compute Train_RHS
    for student in range(len(Train_Path)):
        File = LoadFile(Train_Path[student])
        for page in range(3):
            AllScatter_x, AllScatter_y = StitchWords(File, page*100)
            RHS_temp = []
            point_temp = []
            for point in range(len(AllScatter_x)-1):
                RHS_temp.append([AllScatter_x[point+1] - AllScatter_x[point], AllScatter_y[point+1] - AllScatter_y[point]])
                point_temp.append([AllScatter_x[point], AllScatter_y[point]])
            Train_RHS.append(RHS_temp)
            Train_RHS_Label.append(Train_Label[student])
            Train_Point.append(point_temp)
    # Compute Validation_RHS
    for student in range(len(Validation_Path)):
        File = LoadFile(Validation_Path[student])
        # validation dataset only has 100 words
        for page in range(1):
            AllScatter_x, AllScatter_y = StitchWords(File, page * 100)
            RHS_temp = []
            point_temp = []
            for point in range(len(AllScatter_x) - 1):
                RHS_temp.append([AllScatter_x[point + 1] - AllScatter_x[point], AllScatter_y[point + 1] - AllScatter_y[point]])
                point_temp.append([AllScatter_x[point], AllScatter_y[point]])
            Validation_RHS.append(RHS_temp)
            Validation_RHS_Label.append(Validation_Label[student])
            Validation_Point.append(point_temp)
    RHS_Data = dict(Train_RHS=Train_RHS, Train_RHS_Label=Train_RHS_Label, Train_Point=Train_Point,
                    Validation_RHS=Validation_RHS, Validation_RHS_Label=Validation_RHS_Label, Validation_Point=Validation_Point)
    return RHS_Data


def ShowPoint(RHS_Data):
    # 功能：随机从所有页面中抽取一页并进行显示，显示数据的正确
    Train_Point = RHS_Data['Train_Point']
    ran = np.random.random_integers(len(Train_Point))
    Page = Train_Point[ran]
    plt.figure()
    plt.ion()
    plt.scatter([Page[i][0] for i in range(len(Page))], [Page[i][1] for i in range(len(Page))], marker='o',color='r',s=1)
    plt.pause(3)
    plt.close()


def ShowRHS(SampleRHS):
    # 功能： 随机从所有抽样RHS中抽取一个sample，并按次sample还原出各点坐标并显示  -  确认RHS数据的正确性
    Train_RHS_Sample = SampleRHS['Train_RHS_Sample']
    position_x, position_y = 600, 600
    Scatter_x, Scatter_y = [position_x], [position_y]
    ran = np.random.random_integers(len(Train_RHS_Sample))
    for scatter in range(len(Train_RHS_Sample[ran])):
        position_x += Train_RHS_Sample[ran][scatter][0]
        position_y += Train_RHS_Sample[ran][scatter][1]
        Scatter_x.append(position_x)
        Scatter_y.append(position_y)
    plt.figure()
    plt.ion()
    plt.scatter(Scatter_x, Scatter_y, marker='o', color='r', s=1)
    plt.pause(3)
    plt.close()



def RandomSampleRHS(RHS_Data, LenofSample, NumofSample):
    # Input: LenofSample： 单个sample的抽样点数   NumofSample: 每个学生抽取的sample数
    Train_RHS, Train_Point = RHS_Data['Train_RHS'], RHS_Data['Train_Point']
    Validation_RHS, Validation_Point = RHS_Data['Validation_RHS'], RHS_Data['Validation_Point']
    Train_RHS_Label, Validation_RHS_Label = RHS_Data['Train_RHS_Label'], RHS_Data['Validation_RHS_Label']
    Train_RHS_Sample, Validation_RHS_Sample = [], []
    Train_RHS_Label_Sample, Validation_RHS_Label_Sample = [], []
    for pages in range(len(Train_RHS)):
        for nums in range(NumofSample):
            ran = np.random.random_integers(len(Train_RHS[pages])-LenofSample)
            sample_temp = Train_RHS[pages][ran:ran+LenofSample]
            Train_RHS_Sample.append(sample_temp)
            Train_RHS_Label_Sample.append(Train_RHS_Label[pages])
    for pages in range(len(Validation_RHS)):
        for nums in range(NumofSample):
            ran = np.random.random_integers(len(Validation_RHS[pages])-LenofSample)
            sample_temp = Validation_RHS[pages][ran:ran+LenofSample]
            Validation_RHS_Sample.append(sample_temp)
            Validation_RHS_Label_Sample.append(Validation_RHS_Label[pages])
    SampleRHS = dict(Train_RHS_Sample=Train_RHS_Sample, Train_RHS_Label_Sample=Train_RHS_Label_Sample,
                     Validation_RHS_Sample=Validation_RHS_Sample, Validation_RHS_Label_Sample=Validation_RHS_Label_Sample)
    return SampleRHS


Directory = input("Please choose directory to perform data preprocessing: (Data/Data10)?")
TestDataLoadIn = input("Test whether Data was loaded rightly: yes/no")
TestRHS = input("Test whether RHS was generated rightly: yes/no")
Directory = Directory + '/*/*.npy'
path = '/Users/photon/Desktop/Patern_recognition/WriterID/Data10/Validation_with_labels/000.npy'
Dataset = GeneratePath(Directory)
RHS_Data = CreateRHS(Dataset)
if TestDataLoadIn == 'yes':
    ShowPoint(RHS_Data)
# According to the observation, there normally exist nearly 12000 points in each page
SampleRHS = RandomSampleRHS(RHS_Data, 100, 100)
if TestRHS == 'yes':
    ShowRHS(SampleRHS)