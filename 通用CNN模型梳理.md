
# AlexNet
AlexNet由Alex Krizhevsky等人在2012年论文`ImageNet Classification with Deep Convolutional Neural Networks`中提出，在当年的ImageNet图像识别竞赛中稳居第一（在ILSVRC-2012数据集上得到top-5 test error rate of 15.3%），首次让世界听说了深度学习的概念、见证了一个全新的领域的崛起。与此同时，基于SVM的传统识别方法的霸主地位就此分崩瓦解。  
AlexNet主要由8层网络结构组成，其中5层是卷积层，3层是全连接层。如下图：
<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/AlexNet.png"/>  
    <br>  
    <em align="center">AlexNet网络结构</em>
</div>  
AlexNet几个独特之处：
- 激活函数采用ReLU从而增加数据稀疏性，加快网络训练收敛
- Local Respond Normalization对某个位置的不同卷积输出进行normalization，据说取得更好的泛化能力，但是后来没什么人使用
- overlapping pooling，一般的池化是没有重叠部分的，重叠池化据说可以避免过拟合，但是后来没什么人用
详细结构说明：
卷积+最大池化+卷积+最大池化+卷积+卷积+卷积+最大池化+全连接（加Dropout）+全连接（加Dropout）+全连接输出（softmax）  
为了避免过拟合采用了Dropout和数据增强。在原始数据基础上，从（256,256）图像提取（224，224）的image translations和image reflections。在测试时候，对每个输入原始图像，截取左上、右上、中间、左下、右下等五个子图分别输入到网络识别，最终求五个输出的平均作为最终输出。

# VGG
<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/VGG.png"/>  
    <br>  
    <em align="center">VGG网络结构</em>
</div>  

# GoogLeNet
<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/GoogLeNet.png"/>  
    <br>  
    <em align="center">GoogLeNet网络结构</em>
</div>  

# ResNet
<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/ResNet.png"/>  
    <br>  
    <em align="center">ResNet网络结构</em>
</div>

<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/Residual_blocks.png"/>  
    <br>  
    <em align="center">Residual blocks网络结构</em>
</div>
