import torch

class CNN(torch.nn.Module):
    def __init__(self, area):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, w, h, 1)
        #    Conv     -> (?, w, h, 32)
        #    Pool     -> (?, w/2, h/2, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, w/2, h/2, 32)
        #    Conv      ->(?, w/2, h/2, 64)
        #    Pool      ->(?, w/4, h/4, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 w/4 x h/4 x 64 inputs -> 6 outputs
        self.fc = torch.nn.Linear(int(area.width / 4)
                                  * int(area.height / 4) * 64, 6, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out