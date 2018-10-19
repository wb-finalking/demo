# Pytorch

### 接加载预训练模型

如果我们使用的模型和原模型完全一样，那么我们可以直接加载别人训练好的模型：

```python
my_resnet = MyResNet(*args, **kwargs)
my_resnet.load_state_dict(torch.load("my_resnet.pth"))
```

当然这样的加载方法是基于PyTorch推荐的存储模型的方法：

```python
torch.save(my_resnet.state_dict(), "my_resnet.pth")
```

还有第二种加载方法：

```python
my_resnet = torch.load("my_resnet.pth")
```

### 加载部分预训练模型

其实大多数时候我们需要根据我们的任务调节我们的模型，所以很难保证模型和公开的模型完全一样，但是预训练模型的参数确实有助于提高训练的准确率，为了结合二者的优点，就需要我们加载部分预训练模型。

```python
pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
model_dict = model.state_dict()
# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
model.load_state_dict(model_dict)
```

因为需要剔除原模型中不匹配的键，也就是层的名字，所以我们的新模型改变了的层需要和原模型对应层的名字不一样，比如：resnet最后一层的名字是fc(PyTorch中)，那么我们修改过的resnet的最后一层就不能取这个名字，可以叫fc_

### 微改基础模型

PyTorch中的torchvision里已经有很多常用的模型了，可以直接调用：

- AlexNet
- VGG
- ResNet
- SqueezeNet
- DenseNet

```python
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
squeezenet = models.squeezenet1_0()
densenet = models.densenet_161()
```

但是对于我们的任务而言有些层并不是直接能用，需要我们微微改一下，比如，resnet最后的全连接层是分1000类，而我们只有21类；又比如，resnet第一层卷积接收的通道是3， 我们可能输入图片的通道是4，那么可以通过以下方法修改：

```python
resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(2048, 21)
```

### 简单预训练

模型已经改完了，接下来我们就进行简单预训练吧。 
我们先从torchvision中调用基本模型，加载预训练模型，然后，重点来了，**将其中的层直接替换为我们需要的层即可**：

```python
resnet = torchvision.models.resnet152(pretrained=True)
# 原本为1000类，改为10类
resnet.fc = torch.nn.Linear(2048, 10)
```

其中使用了pretrained参数，会直接加载预训练模型，内部实现和前文提到的加载预训练的方法一样。因为是先加载的预训练参数，相当于模型中已经有参数了，所以替换掉最后一层即可。

### CPU<->GPU模型互相加载

从官方文档中我们可以看到如下方法

```python
torch.load('tensors.pt')
# 把所有的张量加载到CPU中
torch.load('tensors.pt', map_location=lambda storage, loc: storage)
# 把所有的张量加载到GPU 1中
torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
# 把张量从GPU 1 移动到 GPU 0
torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
```

在cpu上加载预先训练好的GPU模型，有一种强制所有GPU张量在CPU中的方式：

```python
torch.load('my_file.pt', map_location=lambda storage, loc: storage)
```

上述代码只有在模型在一个`GPU`上训练时才起作用。如果我在多个`GPU`上训练我的模型，保存它，然后尝试在`CPU`上加载，我得到这个错误：`KeyError: 'unexpected key "module.conv1.weight" in state_dict'` 如何解决？

您可能已经使用模型保存了模型`nn.DataParallel`，该模型将模型存储在该模型中`module`，而现在您正试图加载模型`DataParallel`。您可以`nn.DataParallel`在网络中暂时添加一个加载目的，也可以加载权重文件，创建一个没有`module`前缀的新的有序字典，然后加载它。

参考：

```python
# original saved file with DataParallel
state_dict = torch.load('myfile.pth.tar')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
```

