import torch
from torch import nn                              # 导入网络
from net import MyLeNet5                          # 导入写好的神经网络
from torch.optim import lr_scheduler              # 导入优化器
from torchvision import datasets, transforms      # 导入数据集
import os                                         # 文件夹相关

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

# 定义一个损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器(根据参数进行反向传播)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔10轮，变为原来的0.1(防止抖动大)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):           # X是图片，y是标签
        # 前向传播
        X, y = X.to(device), y.to(device)                 # 将数据放到GPU中
        output = model(X)
        cur_loss = loss_fn(output, y)                     # 输出值（output）与真实数据（y)进行交叉熵的运算
        _, pred = torch.max(output, axis=1)               # torch.max()输出最大的结果与索引

        cur_acc = torch.sum(y == pred)/output.shape[0]    # output.shape[0]输出16组批次的图片，
                                                          # y==pred真实数据和预测相等为1，不相等为0.16组数据进行累加，算出一轮的精确度
        # 反向传播
        optimizer.zero_grad()                             # 优化器清零
        cur_loss.backward()
        optimizer.step()                                  # 梯度更新

        loss += cur_loss.item()                           # 每一批次的loss值累加
        current += cur_acc.item()                         # 精度累加
        n = n + 1                                         # 批次数
    print("train_loss" + str(loss/n))                     # /n:求平均值
    print("train_loss" + str(current/n))

# 模型验证（不经过反向传播）
def val(dataloader, model, loss_fn):
    model.eval()                                          # 模型验证
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch,(X, y) in enumerate(dataloader):
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred)/output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("val_loss" + str(loss/n))
        print("val_loss" + str(current/n))

        return current/n

# 开始训练
epoch = 50
min_acc = 0
for t in range(epoch):
    print(f'epoch{t+1}\n----------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    # 保存最好的模型权重
    if a > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(), 'save_model/best_model.pth')
    # 保存最后的权重文件
    if t == epoch - 1:
        torch.save(model.state_dict(),"save_model/last_model.pth")
print('Done')

