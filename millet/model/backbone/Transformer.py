import torch
from torch import nn


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 48, d_model))  # 位置编码
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.embedding(src) + self.pos_encoder  # 添加位置编码
        transformer_out = self.transformer_encoder(src)
        # 聚合所有时间步的输出
        pooled_out = transformer_out.mean(dim=1)  # 按时间步（seq_len 维度）取平均
        out = self.fc(pooled_out)  # 通过全连接层输出预测
        # out = self.fc(transformer_out[:, -1, :])  # 取最后一个时间步的输出
        return out


class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(Transformer, self).__init__()

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 48, d_model))  # Maximum sequence length = 500

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # Add positional encoding
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]

        # Permute to shape (seq_len, batch_size, d_model) for transformer input
        x = x.permute(1, 0, 2)

        # Apply Transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=0)  # Shape: (batch_size, d_model)

        # Classification
        out = self.classifier(x)  # Shape: (batch_size, num_classes)
        return out
