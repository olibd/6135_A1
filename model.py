import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 4)),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4)),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(4, 4)),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(4, 4)),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=(4, 4)),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=8192, kernel_size=(4, 4)),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=8192, out_channels=16384, kernel_size=(4, 4)),
            nn.ReLU()
        )

        """self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=16384, out_channels=32768, kernel_size=(4, 4)),
            nn.ReLU()
        )

        #392 704 parameters
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=32768, out_channels=32768, kernel_size=(4, 4)),
            nn.ReLU()
        )"""

        #720 384 params
        self.fullyConnected = nn.Linear(128, 10)

    def forward(self, input):
        out = self.layer1(input)
        #out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        #out = self.layer5(out)
        #out = self.layer6(out)
        #out = self.layer7(out)
        #out = self.layer8(out)
        #out = self.layer9(out)
        return self.fullyConnected(out)