<a name="qpQs8"></a>
## 项目概览
仓库有两个子目录，分别是`SegFormer`和`TRT`。<br />`SegFormer`包含SegFormer的官方实现。<br />（1）利用Nvlab提供的的预训练权重将`onnx`模型导出。<br />（2）通过`OnnxRunTime`库来测试`Onnx`模型的正确性。<br />（3）保存`Onnx`推理的输入输出，作为`TensorRT`推理结果的参照。
```
├── README.md
├── SegFormer
│   ├── checkpoint
│   │   ├── segformer.b1.1024x1024.city.160k.pth
│   │   └── segformer.b2.1024x1024.city.160k.pth
│   ├── onnx
│   │   ├── segformer.b1.1024x1024.city.160k.onnx
│   │   └── segformer.b2.1024x1024.city.160k.onnx
│   └── tools
│       ├── data_make.py     			 # make test data for different batch_size
│       ├── onnxruntime_test.py
│       ├── pytorch2onnx.py
            ...
└── TRT
    ├── data                 			   # data with baseline input and output
    ├── engine               			   # path_to_engine
    ├── LayerNormPlugin     			   # path_to_plugin_v1
    ├── LayerNormPlugin-V3.0-OneFlow-TRT8  # path_to_plugin_v2
    ├── log								   # path_to_log
    ├── python                             # path_to_python_file
    │   ├── ln_replace.py
    │   ├── testSegFormer.py
    │   └── trt_int8_quant.py
    └── script							   # path_to_script_file
        ├── build_fp16.sh
        ├── build_ln_best.sh
        ├── build_ln_fp16_patial.sh
        ├── build_ln_fp16_v2.sh
        └── build.sh
```
<a name="v2MlT"></a>
## 总述

- 原始模型的名称及链接<br />SegFormer<br />[https://github.com/NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)
- 优化效果（精度和加速比）<br />我们最终的可用SegFormer TRT Engine相比于OnnxRunTime，batch_size = 1 时可以实现**3.9x** 加速！并且与OnnxRunTime的结果相比，几乎无损精度。
- 在`Docker`里面代码编译、运行步骤的完整说明
<a name="WYyiK"></a>
## Docker
本项目使用<br />nvidia-docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/dev<br />新建`docker`容器并在此基础上进行工作，具体环境配置过程见[传送门](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/hackathon/setup.md)（初赛环境搭建链接）
<a name="trTox"></a>
### 环境搭建
创建`conda`环境
```bash
conda create -n segformer python=3.7
source activate segformer
```
运行脚本安装依赖包
```bash
cd TRT-Hackathon-2022-SegFormer/SegFormer
chmod u+x segformer.sh
./segformer.sh
```
<a name="rhDSf"></a>
### 预训练权重
下载SegFormer预训练权重，这里我们选择B1和B2两个规模的模型。
```bash
ls checkpoints/
segformer.b1.1024x1024.city.160k.pth
segformer.b2.1024x1024.city.160k.pth
```
<a name="xPKTF"></a>
### onnx模型导出
```bash
mkdir onnx/
python tools/pytorch2onnx.py local_configs/segformer/B1/segformer.b1.1024.1024.city.160k.py  --checkpoint checkpoints/segformer.b1.1024.1024.city.160k.pth --output-file onnx/segformer.b1.1024.1024.city.160k.onnx

python tools/pytorch2onnx.py local_configs/segformer/B2/segformer.b2.1024.1024.city.160k.py  --checkpoint checkpoints/segformer.b2.1024.1024.city.160k.pth --output-file onnx/segformer.b2.1024.1024.city.160k.onnx
```
<a name="T0qCe"></a>
### TRT 引擎构建  & 测试
见优化过程
<a name="CBsmd"></a>
## 原始模型
<a name="A2pr6"></a>
### 模型简介

- Segformer主要用于语义分割，语义分割是计算机视觉中的基本任务，在语义分割中我们需要将视觉输入分为不同的语义可解释类别，`「语义的可解释性」`即分类类别在真实世界中是有意义的。例如，我们可能需要区分图像中属于汽车的所有像素，并把这些像素涂成蓝色。<br />![](http://hiphotos.baidu.com/feed/pic/item/e824b899a9014c080d48b239067b02087bf4f43f.jpg#crop=0&crop=0&crop=1&crop=1&id=jqymr&originHeight=223&originWidth=494&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

语义分割在众多领域有着应用

-  `自动驾驶` <br />对交通场景的有效认知是自动驾驶中的关键一环，尤其是对道路可行域的识别和检测，对前方车辆行人的识别和轨迹预测，这些行为的预测准确性直接决定了自动驾驶汽车的安全性能，例如几年前一辆特斯拉L2级别的自动驾驶汽车由于将一辆白色大货车误识别为天空，导致车毁人亡的悲剧。同时相比于激光雷达的物体检测，使用RGB图像信息可以完成在雾、雪、沙尘暴等恶劣天气条件下的物体检测并且成本较低。而单纯的物体检测会丢失场景的相对位置信息，因此快速准确的图像语义分割将给自动驾驶对环境的感知带来极大的帮助。 
- `抠图`<br />使用语义分割技术，iOS 16 将允许用户从照片中提取主题，然后将该主题作为照片拖放到整个系统中，以便在消息、便笺、邮件等中使用<br />![](https://img1.mydrivers.com/img/20220607/s_5a764868487145a698e2ecf39b643f42.png#crop=0&crop=0&crop=1&crop=1&id=roXt8&originHeight=423&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) 
-  `医学图像分割`<br />语义分割网络可以根据医学图像的某种相似性特征(如亮度、颜色、纹理、面积、形状、位置、局部统计特征或频谱特征等)将医学图像划分为若干个互不相交的“连通”的区域的过程。<br />![](https://pica.zhimg.com/v2-67c0d0d25d57a496564934cdea18e958_1440w.jpg?source=172ae18b#crop=0&crop=0&crop=1&crop=1&id=Uy2pP&originHeight=388&originWidth=804&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) 
<a name="jI6yk"></a>
### 模型的整体结构

-  此为论文原图，但是与实际代码实现有些许出入<br />![](https://github.com/FrancescoSaverioZuppichini/SegFormer/raw/main/images/architecture.png#pic_center#crop=0&crop=0&crop=1&crop=1&id=wlrs5&originHeight=1426&originWidth=2736&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) 
-  实际上应该如下图，`overlap patch merging`的位置应该在每个block的最前端<br />![](https://github.com/FrancescoSaverioZuppichini/SegFormer/raw/main/images/architecture_fixed.png#pic_center#crop=0&crop=0&crop=1&crop=1&id=Kn9X8&originHeight=533&originWidth=476&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) 
<a name="FwpeP"></a>
### 模型特点
Segformer有以下特点：

1. 使用分层次的encoder结构，输出多尺度的特征，并在decoder中将其融合在一起。这类似于CNN里面将浅层特征图与深层特征图融合的做法，目的是使得高分辨率粗粒度的特征和低分辨率细粒度的特征能一起被捕捉到并优化分割结果
1. 抛弃了Transformer和SETR里面的position embedding，在infer的图片的大小和train的大小不一样时，不需要再对position vector做插值
1. 没有像SETR中那样复杂的decoder，他的attention结构主要集中在encoder

![](https://github.com/FrancescoSaverioZuppichini/SegFormer/raw/main/images/BlockCorrect.png#pic_center#crop=0&crop=0&crop=1&crop=1&id=aG14D&originHeight=416&originWidth=345&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />我们来看一下他的主模块，上图是Segformer-Encoder的一个transformer block结构

-  其中Efficient Self-Attention类似于普通的Self-Attention结构，但是使用sequence reduction以减少计算的复杂度，具体操作是引入了一个reduction ratio的参数，通过线性变换将K和V的维度都缩减R倍，因此减少了计算量。 
-  什么是Overlap Patch Merging呢，VIT中，patch merging的操作主要是2D卷积，即通过大核卷积的方式，改变patch_size和stride把特征图进行缩放，形成了特征层级结构。ViT中使用的patch merging过程结合的是非重叠的patches，因此会导致无法保留这些patches之间的局部联系。 
-  segformer没有位置编码，但仍然能够学习位置信息，就是依靠Mix-FFN，它通过直接在前馈网络(FFN)中使用3×3Conv传递位置信息 
<a name="al18h"></a>
### 模型优化的难点
（1）直接转换原项目导出的onnx会出错，需要做常量折叠<br />（2）开启不同的精度选项（tf32、fo16)的转换相对顺利，但是fp16 engine出现了较为严重的精度问题，需要引入LayerNormPlugin。<br />（3）怎么充分利用trt的图融合，并以最小的代价获得可观的加速
<a name="YXlsY"></a>
## 优化过程
<a name="Nf07c"></a>
### FP32 & FP16 engine build
<a name="c2xj9"></a>
#### trtexec
`trtrxrc`是一种无需开发自己的应用程序即可快速使用 `TensorRT` 的工具。`trtexec`工具有三个主要用途：

- 它对于在随机或用户提供的输入数据上对网络进行集中测试很有用。
- 它对于从模型生成序列化引擎很有用。
- 它对于从构建器生成序列化时序缓存很有用。

我们将从最简单的`trtexec`命令行开始构建`segFormer`的序列化引擎。
```cpp
trtexec \
    --onnx=/path_to_onnx/segformer.b2.1024x1024.city.160k.onnx \
    --minShapes=input:1x3x1024x1024 \
    --optShapes=input:4x3x1024x1024 \
    --maxShapes=input:8x3x1024x1024 \
    --workspace=23000 \
    --saveEngine=/path_to_engine/segFormer_fp32.plan \
    --verbose \
    > /path_to_log/segformer_fp32.txt
```
其中：<br />`--onnx`指定onnx模型的绝对路径。<br />`--minShapes`、`--optShapes`、`--maxShapes`分别指定模型输入的最小、最优、最大的Shape，我们在导出SegFormer的onnx模型是指定了batch_size维度为动态维度，所以在构建引擎是需要告诉TensorRT动态维度的范围以及最优的维度，好让TensorRT在选择算子实现时兼顾最大最小范围的形状的同时去优化最优形状的性能。<br />`--workspace`指定构建序列化引擎时的显存大小，建议尽可能的大。<br />`--verbose`指定日志等级为verbose，它会详细的将构建过程中的信息保存到日志中<br />将上述命令保存为`build.sh`，运行如下命令：
```cpp
chmod u+x build.sh # 赋予shell脚本权限
./build.sh
```

        如果一切顺利的话，我们只需等待构建完成就可以看到生成的序列化引擎文件`segFormer.plan`了，可惜事与愿违，我们收获了如下错误：

```cpp
[06/10/2022-16:28:05] [E] [TRT] ModelImporter.cpp:779: ERROR: ModelImporter.cpp:180 In function parseGraph:
[6] Invalid Node - Pad_2408
[shuffleNode.cpp::symbolicExecute::392] Error Code 4: Internal Error (Reshape_2397: IShuffleLayer applied to shape tensor must have 0 or 1 reshape dimensions: dimensions were [-1,2])
```
        错误提示`onnx`模型中`Reshape_2397`结点的`shape tensor`必须有0或1个`reshape dimensions`，而当前的`dimensions`是[-1, 2]。<br />        经过一番查阅，错误来源于和padding相关的节点，目前不支持2D的reshape dimensions。
> I think padding related node is causing error, we don’t support 2D shape tensors yet. We can try workaround constant-fold with polygraphy. After this we are able to successfully generate engine.


![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1655363952360-d3221f56-11cc-444e-938d-2d7071e0f6cf.png#clientId=u60006bb7-b992-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=919&id=rKOPs&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1838&originWidth=1758&originalType=binary&ratio=1&rotation=0&showTitle=false&size=245347&status=done&style=none&taskId=u79e6bc8b-3606-48a0-bf36-b063f413a5c&title=&width=879)
<a name="sjRTU"></a>
#### PolyGraphy     
   通过使用Netron工具观察onnx模型结构可以发现Pad结点有两个输入分支data和pads，出问题的结点Reshape_2397出现在pads这条分支里，注意到pads分支起始结点是一个ConstantOfShape结点，也就是说pads实际上是由常量形状推导出来的。        <br />        找到了问题的原因，那么接下来的事情就很简单了，我们使用Polygraphy工具对我们的onnx模型做常量折叠的图优化。
```cpp
polygraphy surgeon sanitize  /path_to_onnx/segformer.b2.1024x1024.city.160k.onnx --fold-constants --output /path_to_onnx/segformer.b2.1024x1024.city.160k_v1.onnx 
```
![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1655364025053-6fa03c50-6614-457f-9a67-7d9d4792c42d.png#clientId=u60006bb7-b992-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=921&id=Zm98A&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1842&originWidth=1950&originalType=binary&ratio=1&rotation=0&showTitle=false&size=175768&status=done&style=none&taskId=ue1e3ddf2-97d3-43a0-bd53-f237da76b4a&title=&width=975)<br />        此时我们观察修改之后的onnx模型结构，Pad结点只有data一个输入分支，pads通过Initializer的形式输入，完成我们所希望的常量折叠的效果。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1655364490081-880376cc-0e58-4712-899d-6fa758972b68.png#clientId=u60006bb7-b992-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=874&id=g3RCI&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1748&originWidth=1798&originalType=binary&ratio=1&rotation=0&showTitle=false&size=185771&status=done&style=none&taskId=ufe564c4c-1aa7-4dd9-95a8-13a08b778a5&title=&width=899)<br />    另外，对比优化前后的onnx模型结构，整个计算图清爽了许多，例如模型最开始的有两个连续的Slice结点，通过观察starts、ends、axes、strides参数可以发现，第一个Slice结点（start=0、ends=1024、axes=1、strides=1）和第二个Slice结点（start=0、ends=1024、axes=2、strides=1）实际上对输入没有做任何修改，这两次切片都是没有意义的，优化后的模型这两个Slice节点都被移除了。
<a name="VFUiB"></a>
#### FP32 engine build & test
        完成了图优化后，再次运行构建脚本便可以得到序列化引擎segFormer_fp32.plan。<br />        为了测试生成的序列化引擎的性能和精度，我们编写了testSegFormer.py脚本，该脚本会反序列化我们刚刚生成的engine文件，读取我们保存的输入输出数据文件（.npz）推理并与onnxruntime的结果做精度对比，输出不同batch_size下的latency、throughout、absolute diff。
```cpp
python testSegFormer.py
```
脚本打印如下信息：
```cpp
bs: Batch Size
lt: Latency (ms)
tp: throughput (img/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+--------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       r0| output check
----+--------+---------+---------+---------+-------------

   1,  76.388,1.309e+01,2.384e-05,0.000e+00, Good
   2, 147.725,1.354e+01,2.432e-05,0.000e+00, Good
   4, 279.271,1.432e+01,2.122e-05,0.000e+00, Good
   8, 592.089,1.351e+01,2.372e-05,0.000e+00, Good
```
我们构建的fp32 engine在batch_size=1，图片大小为1024x1024时76ms可以完成推理，并且相对误差（分类错误的像素点占总像素点的比例）在1e-5级别。
<a name="kMLxa"></a>
#### FP16 engine build & test
另外，为了进一步加速，我们可以在构建engine时加入--fp16的选项，TRT会在合适的层使用fp16精度计算。
```cpp
trtexec \
	--onnx=/path_to_onnx/segformer.b2.1024x1024.city.160k.onnx \
    --minShapes=input:1x3x1024x1024 \
    --optShapes=input:4x3x1024x1024 \
    --maxShapes=input:8x3x1024x1024 \
    --fp16 \
    --workspace=23000 \
    --saveEngine=/path_to_engine/segFormer_fp16.plan \
    --verbose \
    > /path_to_log/segformer_fp16.txt
```
得到segFormer_fp16.plan后，再次运行
```cpp
python testSegFormer.py
```
脚本打印如下信息：
```cpp
bs: Batch Size
lt: Latency (ms)
tp: throughput (img/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+--------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       r0| output check
----+--------+---------+---------+---------+-------------

   1,  41.255,2.424e+01,7.800e-03,0.000e+00, Bad
   2,  81.047,2.468e+01,7.324e-03,0.000e+00, Bad
   4, 138.001,2.899e+01,7.200e-03,0.000e+00, Bad
   8, 319.691,2.502e+01,8.387e-03,0.000e+00, Bad
```
我们构建的fp16 engine在batch_size=1，图片大小为1024x1024时41ms可以完成推理，但是相对误差（分类错误的像素点占总像素点的比例）在1e-2级别，精度出现较大的损失，我们后面要着手解决这个问题。<br />小结：我们利用Polygraphy做常量折叠后，通过trtexec分别构建了FP32 engine和FP16 engine，并测试了性能和精度。
<a name="pWKRC"></a>
### Nsight System Profile
```cpp
nsys profile -o segformer-fp32-moPlugin --force-overwrite true  trtexec --loadEngine=/path_to_engine/segFormer_fp32.plan --iterations=10 --idleTime=500 --duration=0 --useSpinWait 
```
本地打开profile文件，可以发现：
<a name="JRACk"></a>
#### GELU算子自动融合
<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656082928773-6437e0bb-354b-4cfb-8db2-f8dcbccdfce3.png#clientId=u7e45191a-b316-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=295&id=u26d0deff&margin=%5Bobject%20Object%5D&name=image.png&originHeight=742&originWidth=1526&originalType=binary&ratio=1&rotation=0&showTitle=false&size=60655&status=done&style=none&taskId=u1d3c8e35-45da-4113-b547-ab5fc3e04ea&title=&width=607)<br />图示的GELU的onnx算子在TRT中被自动融合成：
```
PWN(
	PWN(
		PWN(
			(PWN(
				543 + (Unnamed Layer* 259) [Shuffle],  
				Div_177),  
				Erf_178),
			 PWN(
				546 + (Unnamed Layer* 263) [Shuffle], 
				Add_180)), 
				Mul_181), 
			PWN(
				549 + (Unnamed Layer* 267) [Shuffle], 
				Mul_183))
```
<br />可以看出，GELU算子在TRT中融合成了一个层来实现，我们无须再做处理。<br />LayerNorm算子部分融合<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656083646596-eb632d9d-830e-472d-8c47-ee216e30779e.png#clientId=u7e45191a-b316-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=548&id=u9004eb93&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1096&originWidth=1548&originalType=binary&ratio=1&rotation=0&showTitle=false&size=97587&status=done&style=none&taskId=u564a0cca-e95f-4444-baec-0a8cf2aa71b&title=&width=774)<br />图示的LayerNorm的onnx算子在TRT中被部分融合成：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656083741047-2e18b317-7e84-4a1e-83e3-024d2a6e584b.png#clientId=u7e45191a-b316-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=131&id=u55680163&margin=%5Bobject%20Object%5D&name=image.png&originHeight=262&originWidth=2296&originalType=binary&ratio=1&rotation=0&showTitle=false&size=96760&status=done&style=none&taskId=u1e1f4b7f-1a23-4530-8d4b-5d8daa192f7&title=&width=1148)
```
ReduceMean
Sub
PWN(
	398 + (Unnamed Layer* 75) [Shuffle], 
	Pow_46)
ReduceMean
PWN(
	PWN(
		PWN(
			PWN(
				PWN(
					401 + (Unnamed Layer* 79) [Shuffle], 
					Add_49), 
				Sqrt_50), 
			Div_51), 
		Mul_52), 
Add_53)
```
<br />可以看到，图融合后TRT仍然使用了三个层来实现LayerNorm，这显然是不够高效的，并且在SegFormer中有53个LayerNorm算子，符合热点代码的特征，我们可以尝试写一个高效的Plugin来实现。
<a name="d8Hta"></a>
#### Attention层融合
<br />在查看TRT生成的engine中耗时的层可以发现，有16个ForeignNode开头的层耗时相对较多，而恰巧在SegFormer的Encoder中有16个Transformer Block，很容易联想是TRT自动融合了Transformer Block中的一部分算子，我们来一探究竟。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656084366047-056f8eab-620c-4dfc-aece-4f044d6c53b9.png#clientId=u7e45191a-b316-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=75&id=uc01b287d&margin=%5Bobject%20Object%5D&name=image.png&originHeight=150&originWidth=2286&originalType=binary&ratio=1&rotation=0&showTitle=false&size=51925&status=done&style=none&taskId=u88f61b32-fa6a-4a38-ad70-6c5a6e2b902&title=&width=1143)<br />以其中一个ForeignNode为例，我们定位到它在计算图中处于Conv_93和Conv_166中，对应到onnx中便是：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656084576869-3b409924-9b17-409e-9074-c9da0a4cae69.png#clientId=u7e45191a-b316-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=1463&id=u785fc5f2&margin=%5Bobject%20Object%5D&name=image.png&originHeight=2925&originWidth=1446&originalType=binary&ratio=1&rotation=0&showTitle=false&size=314325&status=done&style=none&taskId=ue7c66567-9920-488d-8ca4-255ce454352&title=&width=723)<br />onnx中的计算子图虽然看上去比较复杂，但是简单梳理下<br />（1）忽略形状相关的算子（TRT自带形状推理器，会在构建期推理每个层的输出Shape）<br />（2）该子图实际包含2个LayerNorm算子和1个Attention算子<br />由此我们可以看出TRT是有非常强大的图融合能力，我们在后续优化中要尽量避免破坏它的融合（除非融合的层有精度问题）。<br />小结：我们使用了Nsight System对构建的engine进行Profile，观察TRT自动图融合并找到了LayerNorm可以优化的地方。

<a name="cEgNt"></a>
### FP16 engine with LayerNormPlugin build
<a name="v9ePE"></a>
#### 生成LayerNormPlugin的动态链接库
```cpp
cd ./LayerNormPlugin-V3.0-OneFlow-TRT8
make
```
在该目录下会生成LayerNorm.so。
<a name="pdpse"></a>
#### LayerNorm算子替换
通过Onnx-GraphSurgen将原Onnx计算图中零散的算子替换成LayerNorm算子。
```cpp
python python/ln_replace.py
```
在onnx模型存放路径会生成segformer.b2.1024x1024.city.160k_replace_ln_v1.onnx
<a name="ECaVd"></a>
#### 重新构建
```cpp
trtexec \
	--onnx=/root/onnx/segformer.b2.1024x1024.city.160k_v1.onnx \
    --fp16 \
    --minShapes=input:1x3x1024x1024 \
    --optShapes=input:4x3x1024x1024 \
    --maxShapes=input:8x3x1024x1024 \
    --workspace=23000 \
    --saveEngine=/path_to_engine/segFormer_ln_fp16.plan \
    --verbose \
    --tacticSources=-CUDNN,+CUBLAS \
    --plugins=/path_to_plugin/LayerNorm.so \
    > /path_to_log/segformer_ln_fp16.txt
```
<a name="dnpvi"></a>
#### 测试
```cpp
bs: Batch Size
lt: Latency (ms)
tp: throughput (img/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+--------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       r0| output check
----+--------+---------+---------+---------+-------------

   1,  38.382,2.605e+01,2.384e-04,0.000e+00, Good
   2,  74.199,2.695e+01,2.084e-04,0.000e+00, Good
   4, 140.734,2.842e+01,2.003e-04,0.000e+00, Good
   8, 288.861,2.770e+01,2.078e-04,0.000e+00, Good
```
我们构建的fp16 engine with LayerNormPlugin在batch_size=1，图片大小为1024x1024时38ms可以完成推理，并且相对误差（分类错误的像素点占总像素点的比例）在1e-4级别，精度相比替换LayerNorm之前有了大幅提升，那么可以合理的猜测之前FP16 engine是为LayerNorm中精度敏感的算子（ReduceMean等）选择了低精度实现并出现了溢出，造成最后较大的误差。
<a name="ogeeo"></a>
### FP16 engine with partial LayerNormPlugin build 
<a name="s4A36"></a>
#### 部分替换
前面在profile的时候提到，TRT会将两个LayerNorm和1个Attention融合为一个ForeignNode实现加速，而我们在上一步中将网络中所有的LayerNorm替换为LayerNormPlugin，TRT不会合并我们写的PLugin，通过对比FP16-no-Plugin engine 和 FP16-Plugin engine结果如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656170947026-6ac151c0-3789-4d60-ac5b-3afce7255dab.png#clientId=u9a3b1c20-98e0-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=406&id=u0f4e9ff3&margin=%5Bobject%20Object%5D&name=image.png&originHeight=812&originWidth=1700&originalType=binary&ratio=1&rotation=0&showTitle=false&size=210470&status=done&style=none&taskId=u8cf742a8-4efc-46f0-8b90-6d8014bcc1f&title=&width=850)![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656171112502-57ab9b53-d52e-428c-953d-52aa1f469484.png#clientId=u9a3b1c20-98e0-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=570&id=u6902860f&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1140&originWidth=1718&originalType=binary&ratio=1&rotation=0&showTitle=false&size=323785&status=done&style=none&taskId=ud4f868a8-1d73-4f6f-abca-d4bf9177671&title=&width=859)<br />替换前耗时：2.189ms<br />替换后耗时：2.058ms<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656171434651-196d6067-1de3-4c3e-b878-b502bb9c0f06.png#clientId=u9a3b1c20-98e0-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=209&id=uc9b67f9e&margin=%5Bobject%20Object%5D&name=image.png&originHeight=418&originWidth=2290&originalType=binary&ratio=1&rotation=0&showTitle=false&size=136253&status=done&style=none&taskId=uc7aa3a5d-eee0-48f3-963e-163cf418623&title=&width=1145)<br />这么看下来似乎破坏TRT融合使用了较多个算子实现并没有带来性能下降，甚至略有提升，我们的担心是多余的。但是真的是这样吗？仔细对这16个ForeignNode替换前后的表现，会发现还是全部替换以后的总耗时还是增加的，只是相差的幅度比较小
<a name="gYP3S"></a>
#### 测试
替换ForeignNode之外的所有LayerNorm算子，测试下性能与精度：
```cpp
bs: Batch Size
lt: Latency (ms)
tp: throughput (img/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+--------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       r0| output check
----+--------+---------+---------+---------+-------------

   1,  37.103,2.695e+01,2.842e-04,0.000e+00, Good
   2,  74.404,2.688e+01,2.251e-04,0.000e+00, Good
   4, 146.111,2.738e+01,1.915e-04,0.000e+00, Good
   8, 291.979,2.740e+01,2.185e-04,0.000e+00, Good
```
从最后的结果也可以看出，部分替换LayerNorm算子不破坏TRT融合在速度上还是略胜一筹。我们构建的fp16 engine with partial LayerNormPlugin在batch_size=1，图片大小为1024x1024时37ms可以完成推理，并且相对误差（分类错误的像素点占总像素点的比例）在1e-4级别。
<a name="rDAXk"></a>
### int8 engine with partial LayerNormPlugin build 
<a name="YBabj"></a>
#### 制作校准集
此外我们还尝试了在构建时开启int8精度，首先制作校准集。
```cpp
python python/trt_int8_quant.py
```
等待校准集文件segformer_calibration.cache的生成（需要一定的时间，因为TRT组要统计中间层的激活范围来确定量化参数），
```cpp
TRT-8401-EntropyCalibration2
input: 3caa54fc
371: 3caa54fc
376: 3caa54fc
380: 3cbe52af
(Unnamed Layer* 64) [Shuffle]_output: 0
366: 0
395: 3cbe52af
404: 3d5c11b5
(Unnamed Layer* 82) [Shuffle]_output: 3c56713e
405: 3c209a08
(Unnamed Layer* 85) [Shuffle]_output: 3b1d2336
406: 3c11f9d8
415: 3d58799c
(Unnamed Layer* 89) [Shuffle]_output: 3c722dc3
416: 3d8e949a
(Unnamed Layer* 92) [Shuffle]_output: 3c37c009
...
```
<a name="mQybV"></a>
#### 重新构建
在构建选项中开启int8，并传递校准集路径。
```cpp
trtexec \
    --onnx=/root/onnx/segformer.b2.1024x1024.city.160k_v1.onnx \
    --fp16 \
    --minShapes=input:1x3x1024x1024 \
    --optShapes=input:4x3x1024x1024 \
    --maxShapes=input:8x3x1024x1024 \
    --workspace=23000 \
    --saveEngine=/path_to_engine/segFormer_partial_ln_fp16_int8.plan \
    --verbose \
    --tacticSources=-CUDNN,+CUBLAS \
    --plugins=/path_to_plugin/LayerNorm.so \
    --precisionConstraints=obey \
    --calib="/home/pengsky/TRT-Hackathon-2022-SegFormer/TRT/python/segformer_calibration_backup.cache" \
    > /path_to_log/segformer_partial_ln_fp16_int8.txt
```
<a name="j5GUm"></a>
#### 测试
```cpp
bs: Batch Size
lt: Latency (ms)
tp: throughput (img/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+--------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       r0| output check
----+--------+---------+---------+---------+-------------

   1,  34.223,2.922e+01,9.990e-01,3.000e+00, Bad
   2,  65.595,3.049e+01,9.986e-01,3.000e+00, Bad
   4, 124.675,3.208e+01,9.974e-01,3.000e+00, Bad
   8, 253.542,3.155e+01,9.914e-01,1.000e+00, Bad
```
我们构建的int8  engine with partial LayerNormPlugin在batch_size=1，图片大小为1024x1024时37ms可以完成推理，但是相对误差（分类错误的像素点占总像素点的比例）在1e-2级别，这个误差对语义分割来说是一个不可用的状态（几乎全错）。<br />后续可以尝试的改进：<br />（1）方法一：找到使用低精度的层，手动调整为高精度实现，重新构建并测试生成的engine精度，直到找到问题层<br />（2）方法二：使用Polygraphy debug工具，详情见[https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug)
<a name="bzYGD"></a>
### Others

- 若实际环境中只需要跑batch_size = 1的推理，那么可以将构建构建选项中的optShape改为1x3x1024x1024，经过我们的测试，这样改动后单batch_size可以实现1ms左右的提升。
- 构建选项中workspace在显存允许的前提下，尽可能的给大，这样可能可以生成性能更好的engine。
- 在使用nsys进行profile时，本地ui打开.nsny-rep文件时，想要观察Kernel耗时请选择CUDA HW处右键show in Events View，而不是TensorRT处！（这里的耗时只是kernel的启动时间）<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12976256/1656223117172-22e366fd-badb-444e-8050-db15c196b20b.png#clientId=u1eb0a5df-71e3-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=344&id=uc2dd3552&margin=%5Bobject%20Object%5D&name=image.png&originHeight=688&originWidth=676&originalType=binary&ratio=1&rotation=0&showTitle=false&size=139829&status=done&style=none&taskId=ue90116e4-124e-4678-a96b-eba40ec6559&title=&width=338)
- 在运行testSegFormer.py时，注意修改engine文件以及plugin文件路径。
- 我们测试数据来源于city数据集的va部分，通过运行onnxruntime推理过程得到输出，并将输入输出打包成npz文件作为reference。
<a name="amcVS"></a>
##   精度与加速效果
<a name="RUG6S"></a>
#### 精度 & 性能

- 精度 & 性能：不同batch size下性能加速效果以及和onnxruntime相比精度误差。

|  | Baseline |  | TRT-V1 |  | TRT-V2 |  | TRT-V3 |  | fTRT-V4 |  | TRT-V5 |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Lt (ms) | Err | Lt (ms) | Err | Lt (ms) | Err | Lt (ms) | Err | Lt (ms) | Err | Lt (ms) | Err |
| batch_size = 1 | 143.305 | 0 | 76.388 | 2.384e-05 | 41.255 | 7.800e-03 | 38.382 | 2.384e-04 |  37.103 | 2.842e-04 | 34.334 | 9.990e-01 |
| batch_size = 2 | 278.804 | 0 | 147.725 | 2.432e-05 | 81.047 | 7.324e-03 | 74.199 | 2.084e-04 |  74.404 | 2.251e-04 | 65.595 | 9.986e-01 |
| batch_size = 4 | 497.254 | 0 | 279.271 | 2.122e-05 | 138.001 | 7.200e-03 | 140.734 | 2.003e-04 | 146.111 | 1.915e-04 | 124.675 | 9.974e-01 |
| batch_size = 8 | 1208.132 | 0 | 592.089 | 2.372e-05 | 319.691 | 8.387e-03 | 288.861 | 2.078e-04 | 291.979 | 2.185e-04 | 253.542 | 9.914e-01 |

Baseline:  OnnxRunTime-fp32<br />TRT-V1:  fp32-no-plugin<br />TRT-V2:  fp16-no-plugin<br />TRT-V3:  fp16-with-plugin<br />TRT-V4:  fp16-partial-plugin<br />TRT-V5:  int8-partial-plugin<br />Lt：  latency<br />Err：相对误差（分类错误像素点数占总像素点的比例）
<a name="iAuNY"></a>
#### 软硬件信息

- 比赛使用的硬件以软件信息：<br />CPU：Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz<br />GPU：NVIDIA A10 24GB<br />TensorRT v8401<br />CUDA 11.6


