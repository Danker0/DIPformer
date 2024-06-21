DIPformer: A Deep Invert Patching Transformer for Battery State of Health Estimation on Voltage Relaxation

## 如何去运行代码
1. 安装Python 3.9，或者直接在Anaconda环境下运行下面代码：
```
> pip install -r requirements.txt
```
2. 选择要训练的松弛电压数据集。原始的循环数据在[Here](https://zenodo.org/records/6405084)。我们将处理好的数据集放在 ./data/ 文件下面，其中

    **a. 原始松弛电压数据**：`Raw_RV_Data.csv`是直接提取的松弛电压数据，保留了细粒度特征，也是我们文章所使用的数据集。

    ![image](https://github.com/Danker0/DIPformer/assets/116075674/24749621-bde7-41fb-8d9a-1d2d353c4206)

    **b. 松弛电压的统计特征数据**：`Statistical_RV_Data.csv`是采用了现有研究的特征选择方法，即提取最大值、最小值、平均值、方差、偏差等作为特征变量输入给模型。

    ![image](https://github.com/Danker0/DIPformer/assets/116075674/246195bb-f06d-4d8c-8590-3cec06a59ebb)


    **c. 未分层采样的NCA、NCM、NCA+NCM电池数据**：`NCA(Unstratified)`，`NCM(Unstratified)`，`NCA+NCM(Unstratified)`是NiCoAi、NiCoMn、NiCoAi+NiCoMn电池未经过分层抽样的松弛电压数据

3. 运行`run.py`, 设置形参
   ```
   --task_name long_term_forecast --is_training 1 --model_id test --model DIPformer --data Raw_RV_Data
   ```
   
5. 结果将保存在`results`文件夹内，权重文件会保存在`checkpoints`文件中。如果需要进行对松弛电压估计SOH的测试，将测试集放入./data/ 文件下，将`--is_training 0` 即可。
