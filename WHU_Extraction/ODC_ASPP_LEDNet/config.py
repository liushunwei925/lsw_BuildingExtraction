# 模型参数
nc = 2 # num_class
bs = 15 # batch_size
ne = 100 # num_epochs
lr = 0.001 #学习率
num_workers = 4
loss_ = "ce" # (bce)
optimizer_ = "adam" # 优化器

# 模型配置
dataname = "whubuilding" #数据集名称
modelname = "lednet" # 选择模型
savel_model_path = f"./savel_model/{modelname}_{dataname}_{loss_}_{optimizer_}_ne{ne}_bs{bs}"


# 数据集路径,最后要加反斜杠
data_dir=r'.\data'
train_img_dir = data_dir+r"\train/image"
train_lab_dir = data_dir+r"\train/label"
val_img_dir = data_dir+r"\val/image"
val_lab_dir = data_dir+r"\val/label"
test_img_dir = data_dir+r"\test/image"
test_lab_dir = data_dir+r"\test/label"
