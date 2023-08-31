import torch
from net import MyLeNet5
from torch.autograd import Variable   # 图像处理
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# 数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的模型，将模型数据转到GPU
model = MyLeNet5().to(device)


model.load_state_dict(torch.load("D:/xiangmu/Lenet/save_model/best_model.pth"))

# 获取结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 把tensor转化为图片，方便可视化
show = ToPILImage()

# 进入验证
for i in range(5):
    x, y = test_dataset[i][0], test_dataset[i][1]
    show(x).show()

    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():                                    # 张量没有梯度
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", actual:"{actual}"')