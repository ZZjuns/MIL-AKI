from statsmodels.nonparametric.kernels_asymmetric import cdf_kernel_asym
from torch import nn


# 定义 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_channels):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.channel_attention = nn.Sequential(
            nn.Linear(input_size, int(input_size/2)),
            nn.Tanh(),
            nn.Linear(int(input_size/2), n_channels),
            nn.Sigmoid(),
        )

        self.fc1 = nn.Linear(hidden_size * 2, n_channels)  # 乘以2因为是双向 LSTM
        self.fc2 = nn.Linear(hidden_size, n_channels)  # 乘以2因为是双向 LSTM

    def forward(self, x):
        # (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        x1 = self.fc1(lstm_out)

        channel_attention = self.channel_attention(x)
        x = self.fc2(x)
        x2 = x * channel_attention

        return x1+x2, channel_attention


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 乘以2因为是双向 LSTM

    def forward(self, x):
        # (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义继承自 BiLSTM 的 BiLSTMModel
# class BiLSTMModel(BiLSTM):
#     def __init__(self, input_size, hidden_size, output_size):
#         # 调用父类构造函数
#         super(BiLSTMModel, self).__init__(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size * 2, output_size)  # 乘以2因为是双向 LSTM
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
#         return out
