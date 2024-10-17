The models included in the code include five CNN-class networks: UNet, DeepLabV3+, UNet++, SegNet, LEDNet, and three Transformer networks: Topformer, Segformer, and UNetformer. And the LEDNet+ODConv, LEDNet+SAPA_ASPP, ODC_ASPP_LEDNet networks used for ablation experiments.

How to use:

Change the path of num_class, batch_size, num_epochs, and dataset in config.py.
Modify the model in getmodel.py.
Run train.py to train the model.
Run test.py to extract the test set and verify its accuracy.

The WHU building dataset and the Massachusetts building dataset used in this code are data enhancement operations, and the author may be contacted if necessary.
