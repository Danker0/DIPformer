**Read this in other languages: [English](README.md), [中文](README_zh.md).**

## Introduction to DIPformer

**DIPformer** is our proposed Transformer-based SOH (State of Health) estimation model, specifically designed for scenarios that use voltage relaxation to estimate battery SOH. In practical applications, DIPformer allows for the direct extraction and input of raw voltage relaxation, while capturing the correlations between these voltages, i.e., multivariate correlations. This achieves high efficiency and accuracy in SOH estimation tasks.

1. DIPformer allows for the direct use of **raw voltage relaxation data** for battery SOH estimation, thanks to our proposed Inverted Patch Embedding. Directly using raw voltage relaxation ensures the retention of the spatial shape information of the original voltage relaxation curve, seamlessly extracting fine-grained features. This avoids the complex feature selection engineering of the past. Our method can directly utilize the rich information in voltage relaxation data, resulting in more accurate SOH estimation.

   <img width="2501" alt="FIG1111" src="https://github.com/Danker0/DIPformer/assets/116075674/8ca4be90-012f-4115-987e-4c43e940da3e">

2. In our proposed DIPformer, **Inverted Patch Embedding** performs patch operations along the feature dimension, treating each patch as a token representing the relationship between voltage relaxation. Subsequently, our model combines patch-wise multivariate attention to capture the complex dependencies among the relaxation voltages. Then LayerNorm and Feed-forward network are applied to enhance the robustness and accuracy by integrating local features with global degradation trends on raw relaxation voltages.
   <img width="1253" alt="PicPatchiii" src="https://github.com/Danker0/DIPformer/assets/116075674/ebe4aa5c-dfe4-4318-ba75-22bee80b9699">


## How to Run the Code
1. Install Python 3.9, or run the following code directly in an Anaconda environment:

```
pip install -r requirements.txt
```

2. Select the relaxation voltage dataset to train and test. The original cycle data can be found [Here](https://zenodo.org/records/6405084). We have placed the processed datasets in the ./data/ folder, including:

    **a. Raw Relaxed Voltage Data**: `Raw_RV_Data.csv` is the directly extracted voltage relaxation data, retaining fine-grained features, and is the dataset used in our paper.

    ![image](https://github.com/Danker0/DIPformer/assets/116075674/24749621-bde7-41fb-8d9a-1d2d353c4206)

   Each sampling point of the relaxed voltage is input as a feature variable to the estimation model, improving data resolution and reducing the complexity of feature selection engineering. However, the increase in feature variables also increases model complexity and may overlook multivariate correlations (the correlation information between relaxed voltage points). Therefore, our DIPformer was proposed to address these issues.

    **b. Statistical Features of Relaxed Voltage Data**: `Statistical_RV_Data.csv` uses existing feature selection methods, extracting maximum, minimum, average, variance, skewness, etc., as feature variables for the model.

    ![image](https://github.com/Danker0/DIPformer/assets/116075674/246195bb-f06d-4d8c-8590-3cec06a59ebb)

   Simple features can reduce the processing complexity of the model, but may lose the spatial information of the relaxed voltage curve. Losing fine-grained features can affect the overall estimation accuracy of the model.


    **c. Unstratified NCA, NCM, NCA+NCM Battery Data**: `NCA(Unstratified)`, `NCM(Unstratified)`, `NCA+NCM(Unstratified)` are the unstratified relaxed voltage data of NiCoAi, NiCoMn, and NiCoAi+NiCoMn batteries.

4. Run `run.py` and set the parameters:
   
```
--task_name long_term_forecast --is_training 1 --model_id test --model DIPformer --data Raw_RV_Data
```


5. The results will be saved in the `results` folder, and the weight files will be saved in the `checkpoints` folder. If you need to test SOH estimation with relaxed voltage, place the test set in the ./data/ folder and set `--is_training 0`.


## Citation

If you find our work helpful, we will post the specific citation information **here** after the paper is published.

## Contact

If you have any questions or suggestions, feel free to contact me at: **Email**: `1540214738@qq.com`; **WeChat**: `WYYD2020`; **Phone**: `+86 13761019604`

## Acknowledgments

This work was funded by the National Natural Science Foundation of China (NSFC, No. 61702320).

This repository is constructed based on the following:

1. Time Series Library: [Time series library](https://github.com/thuml/Time-Series-Library)
2. Voltage Relaxation Data: [Data source](https://doi.org/10.5281/zenodo.6405084.)
3. Extraction Code: [Extraction Code](https://github.com/Yixiu-Wang/data-driven-capacity-estimation-from-voltage-relaxation)
