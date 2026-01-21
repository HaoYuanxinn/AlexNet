import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.ReLU = nn.ReLU()
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4,)
        self.Pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.Pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.Conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.Conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.Pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(6*6*256, 4096)
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 10)
        self.drop = nn.Dropout(p=0.5)


    def forward(self, x):
        x = self.ReLU(self.Conv1(x))
        x = self.Pool1(x)
        x = self.ReLU(self.Conv2(x))
        x = self.Pool2(x)
        x = self.ReLU(self.Conv3(x))
        x = self.ReLU(self.Conv4(x))
        x = self.ReLU(self.Conv5(x))
        x = self.Pool3(x)

        x = self.flatten(x)
        x = self.ReLU(self.f1(x))
        x = self.drop(x)
        x = self.ReLU(self.f2(x))
        x = self.drop(x)
        x = self.f3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))