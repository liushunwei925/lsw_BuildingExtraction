import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset import prepare_dataloader
from config import *
from getmodel import get_model
import torch.nn.functional as F
import time  # 导入time模块

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 定义训练函数
def train_model(model, train_loader, val_loader, num_epochs, lr, save_every=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(savel_model_path):
        os.makedirs(savel_model_path)
    if not os.path.exists(savel_model_path):
        os.makedirs(savel_model_path)
    seed_everything(42)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []

    # 开始记录整个训练过程的时间
    start_time = time.time()


    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # 记录每个epoch的开始时间

        model.train()
        train_loss = 0.0
        step = 0
        for images, labels in train_loader:
            step+=1
            images, labels = images.to(device),labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):  # 如果模型输出为元组
                loss_main = criterion(outputs[0], labels)  # 主要输出作为第一个元素
                loss_aux = criterion(outputs[1], labels)  # 辅助输出作为第二个元素
                loss = loss_main + 0.4 * loss_aux  # 根据需要调整权重
            else:
                loss = criterion(outputs, labels)


            #outputs = model(images)   #其他模型正确使用
            #loss = criterion(outputs, labels)   #其他模型正确使用
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            print("Epoch[%d] step[%d/%d]->loss:%.4f" %
                  (epoch+1,step, len(train_loader), loss.item()))


        train_loss = train_loss / len(train_loader)
        train_loss_history.append(train_loss)

        # 在验证集上进行评估
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        val_loss_history.append(val_loss)

        epoch_end_time = time.time()  # 记录每个epoch的结束时间
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}，Epoch Duration: {epoch_duration:.2f} seconds")
        # 保存模型
        #if (epoch + 1)  == ne:
        torch.save(model.state_dict(), f"{savel_model_path}/last_model.pt")

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

    # 绘制损失和精度变化图
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_loss_history, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{savel_model_path}/loss.jpg')
    plt.show()

if __name__ == "__main__":

    # 准备数据加载器
    train_loader = prepare_dataloader(train_img_dir,train_lab_dir, batch_size=bs, shuffle=True, num_workers=num_workers)
    val_loader = prepare_dataloader(val_img_dir,val_lab_dir, batch_size=bs, shuffle=False, num_workers=num_workers)
    model = get_model(modelname)
    train_model(model, train_loader, val_loader, ne, lr)
