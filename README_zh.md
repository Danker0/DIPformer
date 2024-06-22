****DIPformer: A Deep Invert Patching Transformer for Battery State of Health Estimation on Voltage Relaxation****

## DIPformer介绍

**DIPformer**是我们提出的一个基于Transformer的SOH估计模型，它专门适用于使用松弛电压估计电池SOH的场景。在实际应用中，DIPformer允许直接提取并输入原始的松弛电压，同时能够抓住松弛电压之间的相关信息，即多变量相关性。实现了SOH估计任务的高效性和准确性。

1. DIPformer允许直接使用**原始的松弛电压**进行电池SOH的估计，得益于我们所提出的反向补丁嵌入(Inverted Patch Embedding)。直接使用原始的松弛电压可以确保在保留原始电压松弛曲线空间形状信息的前提下，无缝提取细粒度特征。避免了以往复杂的特征选取工程。我们的方法可以直接利用松弛电压数据中的丰富信息，从而获得更准确的SOH估计。

   <img width="2501" alt="FIG1111" src="https://github.com/Danker0/DIPformer/assets/116075674/8ca4be90-012f-4115-987e-4c43e940da3e">

2. 我们提出的DIPformer，其中**反向补丁嵌入**(**Inverted Patch Embedding**)沿着特征维度执行贴片(Patch)操作，将每个贴片(Patch)视为表示松弛电压之间关系的标记(Token)。随后，结合了多元关注(Multivariate Attention)来捕捉弛豫电压之间的复杂依赖关系。然后利用层归一化(LayerNorm)和前馈网络(Feed-Forward Network)中将局部特征与原始松弛电压的全局退化趋势相结合，增强了鲁棒性和准确性。

   <img width="1253" alt="PicPatchiii" src="https://github.com/Danker0/DIPformer/assets/116075674/ebe4aa5c-dfe4-4318-ba75-22bee80b9699">


## 如何去运行代码
1. 安装Python 3.9，或者直接在Anaconda环境下运行下面代码：
```
> pip install -r requirements.txt
```
2. 选择要训练的松弛电压数据集。原始的循环数据在[Here](https://zenodo.org/records/6405084)。我们将处理好的数据集放在 ./data/ 文件下面，其中

    **a. 原始松弛电压数据**：`Raw_RV_Data.csv`是直接提取的松弛电压数据，保留了细粒度特征，也是我们文章所使用的数据集。

    ![image](https://github.com/Danker0/DIPformer/assets/116075674/24749621-bde7-41fb-8d9a-1d2d353c4206)

   可以观察到每一个松弛电压采样点都是输入给估计模型的特征变量，这提高了数据分辨率也降低了特征选取工程的人工复杂度。但特征变量的增多会增加模型复杂度，也会忽略掉多变量相关性（松弛电压点之间的相关信息）。因此，我们的DIPformer被提出来解决这些问题。

    **b. 松弛电压的统计特征数据**：`Statistical_RV_Data.csv`是采用了现有研究的特征选择方法，即提取最大值、最小值、平均值、方差、偏差等作为特征变量输入给模型。

    ![image](https://github.com/Danker0/DIPformer/assets/116075674/246195bb-f06d-4d8c-8590-3cec06a59ebb)

   简单的特征可以降低模型的处理复杂度，但可能会丢失了松弛电压曲线的空间信息，失去细粒度特征会影响模型的整体估计精度。


    **c. 未分层采样的NCA、NCM、NCA+NCM电池数据**：`NCA(Unstratified)`，`NCM(Unstratified)`，`NCA+NCM(Unstratified)`是NiCoAi、NiCoMn、NiCoAi+NiCoMn电池未经过分层抽样的松弛电压数据

4. 运行`run.py`, 设置形参
   
   ```
   --task_name long_term_forecast --is_training 1 --model_id test --model DIPformer --data Raw_RV_Data
   ```
   
5. 结果将保存在`results`文件夹内，权重文件会保存在`checkpoints`文件中。如果需要进行对松弛电压估计SOH的测试，将测试集放入./data/ 文件下，将`--is_training 0` 即可。


## 引用
  
   如果你觉的我们的工作有帮助，我们会在论文发表后在 **这里** 贴上具体引用信息。

## 联系

   有任何问题或者建议随时都可以联系我，联系方式： **邮箱**：`1540214738@qq.com`； **微信**：`WYYD2020`
## 致谢

   This work was funded by the National Natural Science Foundation of China (NSFC, No. 61702320)

   This repo is constructed based on following:
   
   1. Time Series Library : [Time series library](https://github.com/thuml/Time-Series-Library)
   2. Voltage Relaxation Data : [Data source](https://doi.org/10.5281/zenodo.6405084.)
   3. Extraction Code : [Extraction Code](https://github.com/Yixiu-Wang/data-driven-capacity-estimation-from-voltage-relaxation)
   
   
