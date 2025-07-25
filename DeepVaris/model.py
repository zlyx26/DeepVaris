import torch.nn as nn

class Classification_cnn1(nn.Module):
    def __init__(self, feature, k, class_num):
        super(Classification_cnn1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * k * int(feature), 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Classification_cnn2(nn.Module):
    def __init__(self, feature, k, class_num):
        super(Classification_cnn2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(k, 3),
                stride=1,
                padding=(0, 1)
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 1 * int(feature), 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Regression_cnn1(nn.Module):
    def __init__(self, feature, k):
        super(Regression_cnn1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(k, 3),
                stride=1,
                padding=(0, 1),
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 1 * int(feature), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

class Regression_cnn2(nn.Module):
    def __init__(self, feature, k):
        super(Regression_cnn2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(k, 3),
                stride=1,
                padding=(0, 1),
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 1 * int(feature), 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x
    


class Classification_dnn(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Classification_dnn, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(20, class_num)
        )

    def forward(self, x):
        return self.fc(x)



class Regression_dnn(nn.Module):
    def __init__(self, input_dim):
        super(Regression_dnn, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x





class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(self.relu(self.bn1(residual)))
        out = self.conv2(self.relu(self.bn2(out)))
        residual = residual + out
        out = self.conv3(self.relu(self.bn3(residual)))
        out = self.conv4(self.relu(self.bn4(out)))
        out = out + residual
        return out



class Mask_Network1(nn.Module):
    def __init__(self, block_num, feature_num, k):
        super(Mask_Network1, self).__init__()
        self.block_num = block_num
        self.first_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, dilation=1)
        self.residual_blocks = self._make_residual_blocks(block_num)
        self.finall_conv = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0)
        self.convs = nn.ModuleList()
        for _ in range(int(block_num * 5) + 1):
            conv_layer = nn.Conv2d(32, 32, kernel_size=(k, 1), padding=0)
            self.convs.append(conv_layer)
        self.fc1 = nn.Linear(feature_num, 1024)
        self.fc2 = nn.Linear(1024, feature_num)
        self.relu = nn.ReLU()

    def _make_residual_blocks(self, block_num):
        layers = []
        dilation_settings = [1] * block_num + [2] * block_num + [4] * block_num + [2] * block_num + [1] * block_num
        for dilation in dilation_settings:
            layers.append(ResidualBlock1(32, 32, dilation=dilation))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.first_conv(x)
        residual = self.convs[0](x)
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            residual += self.convs[i + 1](x)
        x = self.relu(self.finall_conv(self.relu(residual)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 1, 1, self.fc2.out_features)
        return x


class Mask_Network2(nn.Module):
    def __init__(self, block_num, feature_num, k):
        super(Mask_Network2, self).__init__()
        self.block_num = block_num
        self.first_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, dilation=1)
        self.residual_blocks = self._make_residual_blocks(block_num)
        self.finall_conv = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0)
        self.convs = nn.ModuleList()
        for _ in range(int(block_num * 5) + 1):
            conv_layer = nn.Conv2d(32, 32, kernel_size=(k, 1), padding=0)
            self.convs.append(conv_layer)
        self.fc1 = nn.Linear(feature_num, 2048)
        self.fc2 = nn.Linear(2048, feature_num)
        self.relu = nn.ReLU()

    def _make_residual_blocks(self, block_num):
        layers = []
        dilation_settings = [1] * block_num + [2] * block_num + [4] * block_num + [2] * block_num + [1] * block_num
        for dilation in dilation_settings:
            layers.append(ResidualBlock1(32, 32, dilation=dilation))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.first_conv(x)
        residual = self.convs[0](x)
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            residual += self.convs[i + 1](x)
        x = self.relu(self.finall_conv(self.relu(residual)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 1, 1, self.fc2.out_features)
        return x






class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(self.relu(self.bn1(residual)))
        residual = residual + out
        return residual



class Mask_Network3(nn.Module):
    def __init__(self, block_num, feature_num, k):
        super(Mask_Network3, self).__init__()
        self.block_num = block_num
        self.first_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1, dilation=1)
        self.residual_blocks = self._make_residual_blocks(block_num)
        self.finall_conv = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)
        self.convs = nn.ModuleList()
        for _ in range(int(block_num * 2) + 1):  # 减少卷积层数量
            conv_layer = nn.Conv2d(16, 16, kernel_size=(k, 1), padding=0)
            self.convs.append(conv_layer)
        self.fc1 = nn.Linear(feature_num, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, feature_num)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def _make_residual_blocks(self, block_num):
        layers = []
        dilation_settings = [1] * block_num + [2] * block_num
        for dilation in dilation_settings:
            layers.append(ResidualBlock2(16, 16, dilation=dilation))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.first_conv(x)
        residual = self.convs[0](x)
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            residual += self.convs[i + 1](x)
        x = self.relu(self.finall_conv(self.relu(residual)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), 1, 1, self.fc3.out_features)
        return x


class Mask_Network4(nn.Module):
    def __init__(self, block_num, feature_num, k):
        super(Mask_Network4, self).__init__()
        self.block_num = block_num
        self.first_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1, dilation=1)
        self.residual_blocks = self._make_residual_blocks(block_num)
        self.finall_conv = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)
        self.convs = nn.ModuleList()
        for _ in range(int(block_num * 2) + 1):  # 减少卷积层数量
            conv_layer = nn.Conv2d(16, 16, kernel_size=(k, 1), padding=0)
            self.convs.append(conv_layer)
        self.fc1 = nn.Linear(feature_num, 1024)
        self.fc2 = nn.Linear(1024, feature_num)
        self.relu = nn.ReLU()

    def _make_residual_blocks(self, block_num):
        layers = []
        dilation_settings = [1] * block_num + [2] * block_num
        for dilation in dilation_settings:
            layers.append(ResidualBlock2(16, 16, dilation=dilation))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.first_conv(x)
        residual = self.convs[0](x)
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            residual += self.convs[i + 1](x)
        x = self.relu(self.finall_conv(self.relu(residual)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 1, 1, self.fc2.out_features)
        return x
