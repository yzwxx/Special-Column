
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
VGG net是牛津大学于2014年在论文`VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION`提出的，其最大的意义在于证明了在深度学习领域模型的performance是可以通过不断加深网络实现的，同时由于其强大的泛化能力，因此是平时用到的最常见的预训练网络。其网络结构如下：  
<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/VGG.png"/>  
    <br>  
    <em align="center">VGG网络结构</em>
</div>  

VGG net严格控制卷积核尺寸为（3,3），因为(3,3）刚好是能包含上下左右中的局部空间相关性的最小尺寸,对应的stride和padding都是1。通常striding为1时，对于（5,5）的卷积核其padding为2,（7,7）的卷积核其padding为3,以此类推。同时，对于池化，采用size和stride都是2的最大池化。值得注意的是，网络还会采用（1,1）的卷积核，可以看作只起到了维度变化的作用。由于整个网络中的卷积层的stride都是1,因此在卷积前后图像的尺寸保持不变，只有在池化层会产生尺度变化。网络的激活函数采用的是ReLU。  

VGG与其他CNN结构最大的差异在于：  
- 采用堆叠（3,3）的卷积核来代替（5,5）,（7,7）甚至更大尺度的卷积核。这一做法可以极大地节约参数数量，从而更快地训练网络；同时，用两层（3,3）代替（5,5）也引入了两次激活函数，相比一次激活函数，模型的识别能力更强。  
- 加入很多（1,1）的卷积层来增加模型非线性。
VGG net的参数总共大概520M。在GPU上采用mini-batch训练的时候需要注意内存占用。在训练过程中，当验证集的识别率不再提升后，应该对learning rate进行decay。同样，模型参数的初始化对训练的效果至关重要，通常会用预训练的结果作为初始值。

# GoogLeNet
GoogLeNet和VGGnet都在2014年参加了ImageNet竞赛，最终击败VGG位居第一。
<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/GoogLeNet.png"/>  
    <br>  
    <em align="center">GoogLeNet网络结构</em>
</div>  

# ResNet
微软亚洲研究院用152层的ResNet把2015年的ImageNet错误率降低到了3.6%，第一次超过了人类专家的识别水平，也把CNN的深度提升了一个数量级，远深于GoogLeNet的22层和VGGNet的19层。参考`Deep Residual Learning for Image Recognition`，ResNet的想法很简单，就是让一个卷积-ReLU-卷积模块去学习输入到残差的映射，而不是直接学习输入到输出的映射。这个模块被作者称为“残差模块”如图：  
<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/Residual_blocks.png"/>  
    <br>  
    <em align="center">两类常见的Residual blocks网络结构</em>
</div>  
（注意：如果残差块的输入输出维度不同，这意味着skip-connection不应该是直接add输入，而是对输入做Projection（或者补零）以后在和卷积层输出add到一起。）  

深度神经网络最常见的梯度弥散问题，已经被normalized初始化以及在模型中加入BatchNorm所解决。然而，随着模型的加深有时候反而模型会变差。这个问题被叫做degradation。残差网络就是专门为了解决degradation而被提出的。degradation的存在暗示我们，多层神经网络来拟合恒等函数效果并不理想，因此残差网络试图拟合残差函数而非恒等函数。  

残差网络结构受VGG net启发，主要采用多个（3,3）卷积层叠加的方式构造残差块。同时遵循一个原则：对于输出feature map尺寸相同的各个层所包含的filter个数相同，如果输出feature map相比输入长宽都缩小了一倍的话，那么filter个数double，以保持每层网络具有相同的time complexity。  
在残差网络的Downsampling中，每当filter增倍的时候，skip-connection就需要做projection来匹配维度，或者通过补零来扩充维度；事实证明，补零不会对结果造成很大影响，因而是一种最常用的增维方式。而对于filter个数保持不变的残差块中的skip-connection则是identity mapping，减少了参数数量，减少了训练时长。  

<div align="center">
    <img src="https://github.com/yzwxx/Special-Column/blob/master/images/ResNet.png"/>  
    <br>  
    <em align="center">ResNet网络结构</em>
</div>
对于更深层的网络，一种叫做bottleneck的结构更为合理，因为其参数个数更少，从而不会让深度网络的训练时长变得难以承受。这种结构的优势在于，通过设计两个（1,1）的卷积层来实现维度降低和维度恢复，使得夹在中间的（3,3）卷积层所处理的输入和输出都是低维度的，参数个数为$2\times 1\times 1\times high\times low+1\times 3\times 3\times low\times low$。相比上图中左侧的结构，其参数个数为$2\times 3\times 3\times high\times high$，要远大于bottleneck结构。对于这种结构，skip-connection只能使用identity mapping。


