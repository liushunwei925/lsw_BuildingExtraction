The models included in the code include five CNN-class networks: UNet, DeepLabV3+, UNet++, SegNet, LEDNet, and three Transformer networks: Topformer, Segformer, and UNetformer. And the LEDNet+ODConv, LEDNet+SAPA_ASPP, ODC_ASPP_LEDNet networks used for ablation experiments.

How to use:
1. Change the path of num_class, batch_size, num_epochs, and dataset in config.py.
2. Modify the model in getmodel.py.
3. Run train.py to train the model.
4. Run test.py to extract the test set and verify its accuracy.
