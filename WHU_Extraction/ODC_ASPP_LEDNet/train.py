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
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 定义训练函数
def train_model(model, train_loader, val_loader, num_epochs, lr):
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



    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        step = 0
        for images, labels in train_loader:
            step+=1
            images, labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
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

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f"{savel_model_path}/last_model.pt")

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
