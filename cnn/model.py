import torch

class CNN(torch.nn.Module):
    def __init__(self, area):
        super(CNN, self).__init__()

        # L1 : (W, H, 1) > (W/2, H/2, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L2 : (W/2, H/2, 32) > (W/4, H/4, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        # L3 : (W/4, H/4, 64) > (W/8, H/8, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        # L4 : (W/8, H/8, 128) > (W/16, H/16, 128)
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # FC : (W/16 * H/16 * 128) > (W/16 * H/16 * 128) > 6
        input_and_hidden = int(area.width / 16) * int(area.height / 16) * 128
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_and_hidden, input_and_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(input_and_hidden, 6, bias=True)
        )

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        torch.nn.init.xavier_uniform_(self.fc[2].weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        out = torch.nn.functional.log_softmax(out, dim=1)
        return out