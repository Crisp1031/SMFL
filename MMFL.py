import torch
from torch import nn
from FeatureFusionModule import FusionModule
from CrossModalJointATT import JointAttention
import numpy as np

class MMFL(nn.Module):

    def __init__(self):
        super(MMFL, self).__init__()
        self.conv_text = nn.Sequential(
            # TemporalConvNet(768, [9, 9, 9], kernel_size=5, dropout=0.5)
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=5),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(1),
        )
        # self.conv_text_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2)
        # self.conv_text_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2)
        self.conv_audio = nn.Sequential(
            # hidden_size：隐藏层的大小（即隐藏层节点数量），输出向量的维度等于隐藏节点数
            nn.LSTM(input_size=513, hidden_size=16, num_layers=1, batch_first=True, dropout=0.5)

        )
        # self.conv_audio = TemporalConvNet(513, [9, 9, 9, 9, 9], kernel_size=9, dropout=0.5)
        # self.linear_audio = nn.Linear(25, 9)

        # # BiLSTM层如下
        # self.conv_face = nn.Sequential(
        #     nn.LSTM(input_size=204, hidden_size=16, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True),
        # )

        self.conv_face = nn.Sequential(
            nn.LSTM(input_size=204, hidden_size=16, num_layers=1, batch_first=True, dropout=0.5),
        )
        # self.conv_face = TemporalConvNet(204, [9, 9, 9, 9, 9], kernel_size=9, dropout=0.5)
        # self.linear_face = nn.Linear(25, 9)

        # self.conv_face_all = TemporalConvNet(9, [9, 9], kernel_size=3, dropout=0.5)
        # self.conv_text_all = TemporalConvNet(9, [9, 9], kernel_size=3, dropout=0.5)
        
        # self.conv_audio_all = TemporalConvNet(9, [9, 9], kernel_size=3, dropout=0.5)
        self.conv_audio_all = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True, dropout=0.5)
        self.conv_face_all = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True, dropout=0.5)
        self.conv_text_all = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True, dropout=0.5)
        # self.attention_text = Attention(input_size=16, hidden_size=8)
        self.jointAtt = JointAttention(8,8,qkv_bias=True)


        self.fusion_layer = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(3),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(1),
        )

        self.conv_fusion_all = nn.LSTM(input_size=14, hidden_size=14, num_layers=1, batch_first=True, dropout=0.5)
        # self.conv_fusion_all = TemporalConvNet(7, [9, 9], kernel_size=3, dropout=0.5)、
        self.Fusion = FusionModule(input_size=14, hidden_size=14)
        self.linear = nn.Linear(38, 2)
        self.linear_face = nn.Linear(16, 2)

    def get_input(self, k, start_time, stop_time, data):
        if start_time == stop_time:
            stop_time = start_time + 1
        difference = int(stop_time - start_time)  # 时间段（秒）
        start_time = int(np.sum(self.seconds))
        stop_time = start_time + difference
        self.seconds.append(difference)
        audio_start = k * 172
        audio_stop = stop_time * 172
        try:
            audio_input_temp = data[:, :513, audio_start:audio_stop].transpose(1, 2)
        except Exception:
            raise Exception
        face_start = start_time * 30
        face_stop = stop_time * 30
        if face_stop == face_start:
            face_stop = face_stop+30
        face_input_temp = data[:, 513:717, face_start:face_stop].transpose(1, 2)
        text_input_temp = data[:, 717:, k:k + 1].transpose(1, 2)
        return audio_input_temp.float(), face_input_temp.float(), text_input_temp.float()

    def forward(self, data, start_stop_time):

        audio_time_output = None
        face_time_output = None
        text_time_output = None
        combination_time_output = None

        self.seconds = []
        if len(start_stop_time) == 0:
            face_input_temp = data[:, 513:717, :1000].transpose(1, 2)  # 转置
            # face_input_temp = data[:, 513:717, :1000]
            face_output = self.conv_face(face_input_temp)[0][:, -1:, :].squeeze(1)  # 视频特征降维
            output = self.linear_face(face_output)  # 通过16*2的神经网络得到 视频特征
            return output
        else:
            # h = torch.zeros((1, 1, 16)).cuda()
            # c = torch.zeros((1, 1, 16)).cuda()
            # 下面这一段是将一个时间段中的特征进行一个拼接，最后将所有特征都融合在一起
            for k, i in enumerate(start_stop_time.squeeze()):
                if k > 20:
                    break
                audio_input_temp, face_input_temp, text_input_temp = \
                    self.get_input(k, i[0], i[1], data)

                if audio_input_temp.shape[1] != 0:
                    audio_output = self.conv_audio(audio_input_temp)[0][:, -1:, :]
                else:
                    # 处理 face_input_temp 为空的情况
                    audio_output = None  # 或者根据实际需求指定其他默认值
                if face_input_temp.shape[1] != 0:
                    face_output = self.conv_face(face_input_temp)[0][:, -1:, :]
                else:
                    # 处理 face_input_temp 为空的情况
                    face_output = None  # 或者根据实际需求指定其他默认值
                text_output = self.conv_text(text_input_temp)
                # 这里也要改
                # attended_text_output = self.attention_text(text_output)
                combination_temp = torch.cat([audio_output, face_output, text_output], dim=1)  # dim=1是按列拼接
                combination_feature = self.fusion_layer(combination_temp)
                # 下面这一段是在拼接特征，如果没有特征就直接赋值，如果有特征就拼接
                combination_time_output = combination_feature if combination_time_output is None \
                    else torch.cat([combination_time_output, combination_feature], dim=1)
                audio_time_output = audio_output if audio_time_output is None \
                    else torch.cat([audio_time_output, audio_output], dim=1)
                face_time_output = face_output if face_time_output is None \
                    else torch.cat([face_output, face_output], dim=1)
                text_time_output = text_output if text_time_output is None \
                    else torch.cat([text_output, text_output], dim=1)
            audio_time_output = audio_time_output
            face_time_output = face_time_output
            text_time_output = text_time_output
            combination_time_output = combination_time_output
            audio_feature = self.conv_audio_all(audio_time_output)[0][:, -1:, :]
            face_feature = self.conv_face_all(face_time_output)[0][:, -1:, :]
            text_feature = self.conv_text_all(text_time_output)[0][:, -1:, :]
            # face_feature = self.jointAtt(face_feature,text_feature)
            audio_feature = self.jointAtt(text_feature, audio_feature)

            all_feature = self.conv_fusion_all(combination_time_output)[0][:, -1:, :]
            Fusion_all_output = self.Fusion(all_feature)
            combination_all = torch.cat([audio_feature, face_feature, text_feature, Fusion_all_output], dim=2).squeeze(1)
            output = self.linear(combination_all)
            # 最后返回的是一维向量
            return output


import torch
from torch import nn
from FeatureFusionModule import FusionModule
from CrossModalJointATT import JointAttention
import numpy as np

class MMFL(nn.Module):

    def __init__(self):
        super(MMFL, self).__init__()
        self.conv_text = nn.Sequential(
            # TemporalConvNet(768, [9, 9, 9], kernel_size=5, dropout=0.5)
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, stride=5),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(1),
        )
        # self.conv_text_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2)
        # self.conv_text_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2)
        self.conv_audio = nn.Sequential(
            # hidden_size：隐藏层的大小（即隐藏层节点数量），输出向量的维度等于隐藏节点数
            nn.LSTM(input_size=513, hidden_size=16, num_layers=1, batch_first=True, dropout=0.5)

        )
        # self.conv_audio = TemporalConvNet(513, [9, 9, 9, 9, 9], kernel_size=9, dropout=0.5)
        # self.linear_audio = nn.Linear(25, 9)

        # # BiLSTM层如下
        # self.conv_face = nn.Sequential(
        #     nn.LSTM(input_size=204, hidden_size=16, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True),
        # )

        self.conv_face = nn.Sequential(
            nn.LSTM(input_size=204, hidden_size=16, num_layers=1, batch_first=True, dropout=0.5),
        )
        # self.conv_face = TemporalConvNet(204, [9, 9, 9, 9, 9], kernel_size=9, dropout=0.5)
        # self.linear_face = nn.Linear(25, 9)

        # self.conv_face_all = TemporalConvNet(9, [9, 9], kernel_size=3, dropout=0.5)
        # self.conv_text_all = TemporalConvNet(9, [9, 9], kernel_size=3, dropout=0.5)
        
        # self.conv_audio_all = TemporalConvNet(9, [9, 9], kernel_size=3, dropout=0.5)
        self.conv_audio_all = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True, dropout=0.5)
        self.conv_face_all = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True, dropout=0.5)
        self.conv_text_all = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True, dropout=0.5)
        # self.attention_text = Attention(input_size=16, hidden_size=8)
        self.jointAtt = JointAttention(8,8,qkv_bias=True)


        self.fusion_layer = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(3),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(1),
        )

        self.conv_fusion_all = nn.LSTM(input_size=14, hidden_size=14, num_layers=1, batch_first=True, dropout=0.5)
        # self.conv_fusion_all = TemporalConvNet(7, [9, 9], kernel_size=3, dropout=0.5)、
        self.Fusion = FusionModule(input_size=14, hidden_size=14)
        self.linear = nn.Linear(38, 2)
        self.linear_face = nn.Linear(16, 2)

    def get_input(self, k, start_time, stop_time, data):
        if start_time == stop_time:
            stop_time = start_time + 1
        difference = int(stop_time - start_time)  # 时间段（秒）
        start_time = int(np.sum(self.seconds))
        stop_time = start_time + difference
        self.seconds.append(difference)
        audio_start = k * 172
        audio_stop = stop_time * 172
        try:
            audio_input_temp = data[:, :513, audio_start:audio_stop].transpose(1, 2)
        except Exception:
            raise Exception
        face_start = start_time * 30
        face_stop = stop_time * 30
        if face_stop == face_start:
            face_stop = face_stop+30
        face_input_temp = data[:, 513:717, face_start:face_stop].transpose(1, 2)
        text_input_temp = data[:, 717:, k:k + 1].transpose(1, 2)
        return audio_input_temp.float(), face_input_temp.float(), text_input_temp.float()

    def forward(self, data, start_stop_time):

        audio_time_output = None
        face_time_output = None
        text_time_output = None
        combination_time_output = None

        self.seconds = []
        if len(start_stop_time) == 0:
            face_input_temp = data[:, 513:717, :1000].transpose(1, 2)  # 转置
            # face_input_temp = data[:, 513:717, :1000]
            face_output = self.conv_face(face_input_temp)[0][:, -1:, :].squeeze(1)  # 视频特征降维
            output = self.linear_face(face_output)  # 通过16*2的神经网络得到 视频特征
            return output
        else:
            # h = torch.zeros((1, 1, 16)).cuda()
            # c = torch.zeros((1, 1, 16)).cuda()
            # 下面这一段是将一个时间段中的特征进行一个拼接，最后将所有特征都融合在一起
            for k, i in enumerate(start_stop_time.squeeze()):
                if k > 20:
                    break
                audio_input_temp, face_input_temp, text_input_temp = \
                    self.get_input(k, i[0], i[1], data)

                if audio_input_temp.shape[1] != 0:
                    audio_output = self.conv_audio(audio_input_temp)[0][:, -1:, :]
                else:
                    # 处理 face_input_temp 为空的情况
                    audio_output = None  # 或者根据实际需求指定其他默认值
                if face_input_temp.shape[1] != 0:
                    face_output = self.conv_face(face_input_temp)[0][:, -1:, :]
                else:
                    # 处理 face_input_temp 为空的情况
                    face_output = None  # 或者根据实际需求指定其他默认值
                text_output = self.conv_text(text_input_temp)
                # 这里也要改
                # attended_text_output = self.attention_text(text_output)
                combination_temp = torch.cat([audio_output, face_output, text_output], dim=1)  # dim=1是按列拼接
                combination_feature = self.fusion_layer(combination_temp)
                # 下面这一段是在拼接特征，如果没有特征就直接赋值，如果有特征就拼接
                combination_time_output = combination_feature if combination_time_output is None \
                    else torch.cat([combination_time_output, combination_feature], dim=1)
                audio_time_output = audio_output if audio_time_output is None \
                    else torch.cat([audio_time_output, audio_output], dim=1)
                face_time_output = face_output if face_time_output is None \
                    else torch.cat([face_output, face_output], dim=1)
                text_time_output = text_output if text_time_output is None \
                    else torch.cat([text_output, text_output], dim=1)
            audio_time_output = audio_time_output
            face_time_output = face_time_output
            text_time_output = text_time_output
            combination_time_output = combination_time_output
            audio_feature = self.conv_audio_all(audio_time_output)[0][:, -1:, :]
            face_feature = self.conv_face_all(face_time_output)[0][:, -1:, :]
            text_feature = self.conv_text_all(text_time_output)[0][:, -1:, :]
            # face_feature = self.jointAtt(face_feature,text_feature)
            audio_feature = self.jointAtt(text_feature, audio_feature)

            all_feature = self.conv_fusion_all(combination_time_output)[0][:, -1:, :]
            Fusion_all_output = self.Fusion(all_feature)
            combination_all = torch.cat([audio_feature, face_feature, text_feature, Fusion_all_output], dim=2).squeeze(1)
            output = self.linear(combination_all)
            # 最后返回的是一维向量
            return output


