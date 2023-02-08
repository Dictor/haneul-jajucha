import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import cv2
import numpy as np
import model
from util import JajuchaCamera, JajuchaCV, default_perspective

class JajuchaDataset(dsets.ImageFolder):
    def __init__(self, root, perspective_area, is_valid_file=None, ):
        super(JajuchaDataset, self).__init__(
            root=root, is_valid_file=is_valid_file)
        self.perspective_area = perspective_area
        mtx = np.array([[309.07332417, 0., 319.4646727],
                [0., 309.49421445, 225.88178544],
                [0., 0., 1.]], dtype=np.float32)
        self.camera = JajuchaCamera(mtx, 81,
                      np.array([[0.5, 0., 250.],
                                [0., -0.5, 350.],
                                [0., 0., 1.]], dtype=np.float32),
                      np.array([[2., 0., -500.],
                                [0., -2., 700.],
                                [0., 0., 1.]], dtype=np.float32))
    def __getitem__(self, index):
        path, target = self.samples[index]
        stream = open(path.encode("utf-8"), "rb")
        bytes = bytearray(stream.read())
        arr = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        img_perspective = JajuchaCV.cv_perspective(
            img, self.perspective_area.left_top, self.perspective_area.right_top, self.perspective_area.left_bottom, self.perspective_area.right_bottom)
        img_canny = JajuchaCV.cv_canny(img_perspective)
        sample = transforms.ToTensor()(img_canny)
        return sample, target

device = "cpu"
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.manual_seed(777)
#if device == 'cuda':
#    torch.cuda.manual_seed_all(777)
#elif device == 'tml':
#    import torch_directml
#    device = torch_directml.device()

learning_rate = 0.001
training_epochs = 15
batch_size = 100

root_dir = input("dataset root dir? =")
train_dataset = JajuchaDataset(
    root_dir + "\\train", default_perspective)
test_dataset = JajuchaDataset(
    root_dir + "\\test", default_perspective)
loader_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)
loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,)

model = model.CNN(default_perspective).to(device)
# 비용 함수에 소프트맥스 함수 포함되어져 있음.
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(loader_train)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in loader_train:  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

with torch.no_grad():
    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        
        for img, lbl in loader_test:
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            _, pred = output.max(dim=1)
            corr += torch.sum(pred.eq(lbl)).item()
        
        acc = corr / len(loader_test.dataset)
        print("Accuracy: {}".format(acc))

torch.save(model.state_dict(), "model")
