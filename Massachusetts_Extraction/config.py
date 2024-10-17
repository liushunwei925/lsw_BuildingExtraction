# 模型参数
nc = 2 # num_class
bs = 16 # batch_size
ne = 100 # num_epochs
lr = 0.001 #学习率
num_workers = 4
loss_ = "bce" # (bce)
optimizer_ = "adam" # 优化器

# 模型配置
dataname = "MS_zengqiang" #数据集名称
modelname = "OD_SA_ASPP_LEDNet" # 选择模型
savel_model_path = f"./savel_model/{modelname}_{dataname}_{loss_}_{optimizer_}_ne{ne}_bs{bs}"


# 数据集路径,最后要加反斜杠
data_dir=r'H:\dataset\MS_augmented\datasets/'
train_img_dir = data_dir+r"\train/images"
train_lab_dir = data_dir+r"\train/labels"
val_img_dir = data_dir+r"\val/images"
val_lab_dir = data_dir+r"\val/labels"
test_img_dir = data_dir+r"\test/images"
test_lab_dir = data_dir+r"\test/labels"
