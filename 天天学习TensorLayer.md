# week1:1st July ~ 7th July,2017

## TL对接其他库
首先要`import tensorlayer`和其他库文件，然后使用`LambdaLayer`完成对接。`LambdaLayer`是一个按照`Tensor-In-Tensor-Out`原则工作的pipeline，其使用方法如下：
```python
network = ...
network = LambdaLayer(network, fn=tf.nn.relu, name='relu') 
```
其中`fn`可以换成随意的其他输入为Tensor输出也为Tensor的函数，其中便可以使用诸如Keras，TF-slim等其他库函数对Tensor进行计算。

## 自定义层
具体流程如下：
```python
class MyDenseLayer(Layer):
    def __init__(
        self,
        layer = None,
        n_units = 100,
        act = tf.nn.relu,
        name ='simple_dense',
    ):
        # 校验名字是否已被使用（不变）
        Layer.__init__(self, name=name)

        # 本层输入是上层的输出（不变）
        self.inputs = layer.outputs

        # 输出信息（自定义部分）
        print("  MyDenseLayer %s: %d, %s" % (self.name, n_units, act))

        # 本层的功能实现（自定义部分）
        n_in = int(self.inputs._shape[-1])  # 获取上一层输出的数量
        with tf.variable_scope(name) as vs:
            # 新建参数
            W = tf.get_variable(name='W', shape=(n_in, n_units))
            b = tf.get_variable(name='b', shape=(n_units))
            # tensor操作
            self.outputs = act(tf.matmul(self.inputs, W) + b)

        # 获取之前层的参数（不变）
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # 更新层的参数（加入自定义部分）
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )
```
其中自定义部分功能可以通过`LambdaLayer`实现，代码更为简洁。（存疑：`LambdaLayer`如何实现对接已经变量管理，查看源码）

## 训练/测试切换
通常在应用了`DropoutLayer`的时候这个问题会被考虑到，因为网络在训练和测试时候的工作方式不同。通常我们需要在训练的时候把`DropoutLayer`的keep_prob这个变量update到feed_dict中，而在测试的时候用`tl.utils.dict_to_one`来update feed_dict。也就是说，在训练时keep_prob按照Dropout所指定的概率随机置零，在测试的时候keep_prob全部置1（Dropout失效）。  
很好用的一个trick是在Graph definition的时候对一个模型设置is_train=True,reuse=False用于训练，对同一个模型设置is_train=False，reuse=True用于测试和验证，通过reuse的设置我们的训练和测试都是针对同一个网络，而通过is_train的设置可以区别网络在训练和测试时行为。这样我们相当于一个模型定义了2个不同的graph，但是二者训练参数又是共享的。从而不需要再去update feed_dict，省去了很多麻烦。

## 获取网络某层trainable参数或activation
首先需要明确，使用network.all_params的确可以获取到net这个instance所包含的所有参数，但是这是一个list，也就是说只可以用index来索引其中各参数，实在不可操作。在Tensorflow中，每个variable都可以定义一个name来作为唯一标识。一般地，name外可能还会有很多scope，比如某一个model的scope为‘cnn’，其内部有一个`tl.layers.Conv2d`的name设为‘conv1’，那么该层的W的name就是‘cnn/conv1/W_conv2d’。  
如果我们知道一个参数的name，就可以用`tl.layers.get_variables_with_name`获取该参数的值。
```python
train_vars = tl.layers.get_variables_with_name('MLP', True, True)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_vars)
```
如果我们想得到的是某一层的activation，除了输出层可以通过network.outputs获取外，其他的需要用layer的名字获取：
layers = tl.layers.get_layers_with_name(network,"layer_name",True)
在对某一层输出做activation regularization时这一方法会经常用到。

## 使用pretrained CNN和ResNet
check：https://github.com/zsdonghao/tl-book/blob/master/%E5%AE%9E%E4%BE%8B%E4%B8%80.md

## batch of data
当batchsize的data size足以读入机器内存，可以使用`tl.iterate.minibatches`来shuffle并返回指定batchsize的（samples，labels）。当data size很大时，考虑在每个training step最开始处用`tl.prepro.threading_data`利用多线程读取数据。

## 使用TFRecord存储和读取数据

