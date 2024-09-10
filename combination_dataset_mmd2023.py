import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import kaldi_io
import os.path as osp
import pandas as pd
import torch
import time
import cv2
from pathlib2 import Path


def data_generator(Batch_Size):
    # 定义读取文件的格式
    # def default_loader(path):
    #     return Image.open(path).convert('RGB')

    # 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
    class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset

        # 数据处理前准备，包括对ark文件的处理
        def __init__(self, data_dir, label_dir, mode, patient_set, combination=None,
                     transform=None, target_transform=None):  # 初始化一些需要传入的参数

            """
            :param data_dir: 数据路径
            :param label_dir: 标签路径
            :param mode: 训练或测试 train or dev
            :param combination: 组合哪些数据
            :param transform: 数据处理
            :param target_transform:
            """
            super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化

            if combination is None:
                # 选择想要结合哪些模态的特征
                combination = ['audio', 'text', 'face3d']
                # combination = ['audio', 'text']
                # combination = ['text']
                # combination = ['audio']
                # combination = ['face']
                # combination = ['face3d']
            self.audio_suffix = 'audio'
            self.text_suffix = 'text'
            self.face_suffix = 'face'
            # 训练/测试
            self.prefix = mode
            # 模态结合
            self.combination = combination
            self.data = {}
            # 数据存放路径
            self.audio_root = './data/audio_png'
            self.face_root = './data/face_features'   # 二维面部特征
            self.face3d_root = './data/face3d'        # 三维面部特征
            self.text_root = './data/text'
            self.txt_csv_root = './data/txt_csv'
            self.text_sclar = torch.load('./data/run_scaler_text.pth',
                            map_location=lambda storage, loc: storage)
            # 读取标签文件
            train_label_df = pd.read_excel('./data/label.xlsx', sheet_name='hamd') \
                .set_index('id')
            # 判断是否有数据
            train_label_df = train_label_df[train_label_df.index.isin(patient_set)]
            # 根据‘all’数值打标签‘label’：x<8?0:1
            train_label_df['label'] = train_label_df['all'].apply(lambda x: 0 if x < 8 else 1)
            # 将索引强转成字符串类型
            train_label_df.index = train_label_df.index.astype(str)
            # 定义目标列'label'
            target_type = ('label')
            # 将目标列'label'之前的所有数值转为字典序存放在 id_labels
            self.id_labels = train_label_df.loc[:, target_type].to_dict()
            # 处理多模态中的文本
            if 'text' in combination:
                # 创建元组生成器，从ark文件流中读取数据文件
                data_iter = kaldi_io.read_mat_ark(osp.join(self.text_root, 'text.ark'))   # 拼接 text.ark 文件路径
                # 读取数据文件中的数据
                for suffix in [self.text_suffix]:
                    # [suffix]标记存放的数据类型
                    self.data[suffix] = {}
                    for pid, value in data_iter:
                        self.data[suffix][pid] = value
            # 获取id_labels中的所有键
            self.ids = list(self.id_labels.keys())
            # 数据处理
            self.transform = transform
            self.target_transform = target_transform
            # self.loader = loader

        # 找到对用的音频对应的频谱图，最后返回图片的多维数组
        @staticmethod
        def read_audio_png(pid, audiodir):
            # 合成对应的音频频谱图png文件路径
            audiopng_path = osp.join(audiodir, f'{pid}.png')
            # 没有找到对应频谱图也要生成 513*1 的零矩阵
            if not Path(audiopng_path).exists():
                return np.zeros((513, 1))
            #第一个参数filename是图片路径，第二个参数flag表示图片读取模式，共有三种：
            #                                           1. cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写 1。
            #                                           2. cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写 0。
            #                                           3. cv2.IMREAD_UNCHANGED：包括alpha(包括透明度通道)，可以直接写 -1
            # 以多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引，具体图像的通道数由图片的格式来决定
            audiopng = cv2.imread(audiopng_path, cv2.IMREAD_GRAYSCALE)
            # 对输入数据进行归一化
            audiopng = audiopng / 255
            # 返回音频向网络中的输入信息
            return audiopng


        # 据pid读取对应的存放面部三维坐标的txt文件，并对坐标进行归一化处理，最后返回处理好的三维坐标数组
        def read_face_features_3d(self, pid, facedir):
            #每个点的坐标个数 x:68 y:68 z:68
            num_dot = 136+68
            filelist = []
            # 读取存放对应的面部三维坐标点文件
            fn = osp.join(facedir, f'{pid}.txt')
            with open(fn, encoding='utf-8', ) as txtfile:
                line = txtfile.readlines()
                # print('line=', len(line))
                for i, rows in enumerate(line):
                    if i in range(1, len(line)):  # 指定数据哪几行
                        filelist.append(rows)
            txtfile.close()
            numberoflines = len(filelist)
            # print('numberoflines=',numberoflines)
            returnMat = np.zeros((numberoflines, num_dot))  # 生成一个numberoflines行，68+68+68列的矩阵
            index = 0

            for line in filelist:  # 依次读取每行
                line = line.strip()  # 去掉每行头尾空白
                if '-1.#IND' in line:
                    continue
                listline = line.split(',')  # 按逗号分割数据
                returnMat[index, :] = listline[1:num_dot + 1]
                index += 1
            # 坐标矩阵转置
            returnMat = returnMat.T
            x_index = list(range(0, 68, 1))
            y_index = list(range(68, 136, 1))
            z_index = list(range(136, 204, 1))
            # 分别求x y z坐标的最小值和最大值
            x_min = np.min(returnMat[x_index, :])
            x_max = np.max(returnMat[x_index, :])
            y_min = np.min(returnMat[y_index, :])
            y_max = np.max(returnMat[y_index, :])
            z_min = np.min(returnMat[z_index, :])
            z_max = np.max(returnMat[z_index, :])
            # 对x y z坐标的归一化
            returnMat[x_index, :] = (returnMat[x_index, :] - x_min) / (x_max - x_min)
            returnMat[y_index, :] = (returnMat[y_index, :] - y_min) / (y_max - y_min)
            returnMat[z_index, :] = (returnMat[z_index, :] - z_min) / (z_max - z_min)
            # 返回处理后的特征点坐标
            return returnMat

        # 根据pid读取对应的存放面部二维坐标的txt文件，并对坐标进行归一化处理，最后返回处理好的二维坐标数组
        def read_face_features(self, pid, facedir):
            filelist = []
            # 读取二维面部特征点文件
            fn = osp.join(facedir, f'{pid}.txt')
            with open(fn, encoding='utf-8', mode='r') as txtfile:
                # 获取共有几行数据【每0.0333秒的面部68个特征点坐标】
                line = txtfile.readlines()
                # print('line=', len(line))
                for i, rows in enumerate(line):
                    if i in range(1, len(line)):  # 指定数据哪几行
                        filelist.append(rows)
            txtfile.close()
            # 获取文件行数
            numberoflines = len(filelist)
            # print('numberoflines=',numberoflines)
            # 生成一个numberoflines行【每0.0333秒生成一行】，68+68列【68个特征点对应坐标】的矩阵
            returnMat = np.zeros((numberoflines, 136))
            index = 0

            for line in filelist:  # 依次读取每行
                line = line.strip()  # 去掉每行头尾空白
                listline = line.split(',')  # 按逗号分割数据
                # 将listline文件中所有行的第五列到第140列数据存入returnMat数组中
                returnMat[index, :] = listline[4:140]
                index += 1
            # 将returnMat数组进行转置
            returnMat= returnMat.T
            # x坐标在数组的第1行到第68行
            x_index = list(range(0, 68, 1))
            # y坐标在数组的第69行到第136行
            y_index = list(range(68, 136, 1))
            # x,y坐标分别做归一化处理，再放回数组
            returnMat[x_index, :] = returnMat[x_index, :] / 5313
            returnMat[y_index, :] = returnMat[y_index, :] / 3892
            # 返回处理好的面部二维坐标数组
            return returnMat

        # 将 00：00 的时间格式转为 xxxx秒
        def minute_second2second(self, x):
            minute_str, second_str = x.split(':')
            second = int(minute_str) * 60 + int(second_str)
            return second

        # 从对应的cvs文件中读取患者说话的起止时间
        def read_start_end_time(self, pid, csv_path):
            # 根据pid值找到对应的csv文件
            csv_p = Path(csv_path, f'{pid}.csv')
            if not csv_p.exists():
                return []
            # 读取csv文件
            df = pd.read_csv(csv_p, sep='\t')
            # 读取文件中的‘speaker’列中‘患者’行
            df = df[df['speaker'] == '患者']
            # 读取文件中‘start_time’列信息，并将数据单位转为秒
            df['start_time_'] = df['start_time'].apply(lambda x: self.minute_second2second(x))
            # 读取文件中‘stop_time’列信息，并将数据单位转为秒
            df['stop_time_'] = df['stop_time'].apply(lambda x: self.minute_second2second(x))
            # 保存患者每次回答的起止时间
            start_end_time = df[['start_time_', 'stop_time_']].values
            return start_end_time

        def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
            # 在初始化__init__中定义了ids[]
            Participant_ID = self.ids[index]
            label = self.id_labels[Participant_ID]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
            # img = self.loader(fn)  # 按照路径读取图片
            # 实际上是根据label.xlsx文件中的第一列值，再在txt_csv文件夹中找到对应的csv文件，再根据文件获取视频时间
            start_stop_time = self.read_start_end_time(Participant_ID, self.txt_csv_root)
            # 保存不同模态下的输入信息
            datas = []
            # starttime = time.time()
            for data_type in self.combination:
                # 获取音频的输入
                if data_type == 'audio':
                    datas.append(self.read_audio_png(Participant_ID, self.audio_root))
                # 获取归一化后的面部二维坐标
                elif data_type == 'face':
                    datas.append(self.read_face_features(Participant_ID, self.face_root))
                # 获取归一化后的面部三维坐标
                elif data_type == 'face3d':
                    datas.append(self.read_face_features_3d(Participant_ID, self.face3d_root))
                else:
                    value = self.data[data_type][Participant_ID]
                    if data_type == 'text':
                        # 用transform处理文本
                        value = self.text_sclar.transform(value)
                        value = np.transpose(value, (1, 0))   # 按序列转置
                    datas.append(value)
                    # data_iter = self.data[data_type]
                    # true_index = None
                    #
                    # for key, (pid, _) in enumerate(data_iter):
                    #     if pid == Participant_ID:
                    #         true_index = key
                    # enu_dataiter = enumerate(data_iter)
                    # for key, (_, _) in enu_dataiter:
                    #     if key == true_index:
                    #         key, (_, value) = enu_dataiter.__next__()
                    #         if data_type == 'text':
                    #             value = self.text_sclar.transform(value)
                    #             value = np.transpose(value, (1, 0))

                    # for pid, value in data_iter:
                    #     if pid == Participant_ID:
                    #         if data_type == 'text':
                    #             value = self.text_sclar.transform(value)
                    #             value = np.transpose(value, (1, 0))
                    #         datas.append(value)
            # print(time.time() - starttime)
            lengths = [data.shape[-1] for data in datas] # datas中data的最后一维
            # 多模态下最大输入长度
            max_len = max(lengths)

            datas_pad = []

            # 查看各个模态下的要输入网络的数据长度是否等于max_len,短了用0填充
            for data in datas:
                len = data.shape[-1]
                data_pad = np.pad(data, ((0, 0), (0, max_len-len)), mode='constant', constant_values=0)
                datas_pad.append(data_pad)
            # axis参数为指定按照哪个维度进行拼接，并且要保证所要拼接的维度是相同的
            data_input = np.concatenate(datas_pad, axis=0)
            # 把数组转换成tensor，且二者共享内存
            data_input = torch.from_numpy(data_input)
            # if self.transform is not None:
            #     data_input = self.transform(data_input)  # 数据标签转换为Tensor
            return data_input, label, max_len, start_stop_time  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

        def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
            # print('imgs=',len(self.imgs))
            return len(self.id_labels)

    class ToTensor(object):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

        Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
        or if the numpy.ndarray has dtype = np.uint8

        In the other cases, tensors are returned without scaling.
        """

        def __call__(self, pic):
            """
            Args:
                pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

            Returns:
                Tensor: Converted image.
            """
            return torch.tensor(pic, dtype=torch.float32)

        def __repr__(self):
            return self.__class__.__name__ + '()'
    # 根据自己定义的那个MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    # *********************************************数据集读取完毕********************************************************************
    # 图像的初始化操作
    train_transforms = transforms.Compose([
        # 1. 图像变换:重置图像分辨率,图片缩放256 * 256
        # transforms.Resize(19458*136),
        # # 2. 裁剪: 中心裁剪 ,依据给定的size从中心裁剪
        # transforms.CenterCrop(224),
        # 3. 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1].注意事项：归一化至[0-1]是直接除以255
        ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        # # 4. 对数据按通道进行标准化，即先减均值，再除以标准差
        # transforms.Normalize(mean=[0.485], std=[0.229])
        # transforms.Normalize([0.5], [0.5])
    ])
    text_transforms = transforms.Compose([
        # 1. 图像变换:重置图像分辨率,图片缩放256 * 256
        # transforms.Resize(19458*136),
        # # 2. 裁剪: 中心裁剪 ,依据给定的size从中心裁剪
        # transforms.CenterCrop(224),
        # 3. 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1].注意事项：归一化至[0-1]是直接除以255
        ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        # # 4. 对数据按通道进行标准化，即先减均值，再除以标准差
        # transforms.Normalize(mean=[0.485], std=[0.229])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize([0.5], [0.5])
    ])
    train_set = get_train_set('./data/train_set')
    test_set = get_train_set('./data/test_set')
    # 数据集加载方式设置
    # sampler =
    train_data = MyDataset(data_dir='./data', label_dir='./labels', mode='train',
                           patient_set=train_set, transform=train_transforms)
    test_data = MyDataset(data_dir='./data', label_dir='./labels', mode='dev',
                          patient_set=test_set, transform=text_transforms)
    # train_data = MyDataset(txt=root1 + 'train_list.txt')
    # test_data = MyDataset(txt=root2 + 'test_list.txt')
    # print('num_of_trainData:', train_data.__dict__)
    # print('num_of_testData:', len(test_data))
    weights = np.loadtxt('./data/id.weights')
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True)
    # sampler = torch.utils.data.RandomSampler(train_data)
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，每次加载loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(dataset=train_data, batch_size=Batch_Size,
                              sampler=sampler, num_workers=0, drop_last=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1,
                             shuffle=False, num_workers=0, drop_last=False)
    # print('num_of_trainData:', len(train_loader))
    # print('num_of_testData:', len(test_loader))
    return train_loader, test_loader


def get_train_set(path) -> list:
    result = []
    with open(path, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            result.append(i.strip())
    return result
