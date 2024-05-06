# Modeling Spatial Distribution of Snow Water Equivalent using Transfer Learning across Mountainous Basins

Accurately estimating snow water equivalent (SWE) is crucial for understanding the impacts of climate change, urbanization, and population growth on water resources. High operational costs of lidar observations limit the frequency and coverage of SWE estimates at high spatial resolutions, leading to significant data gaps. We address this challenge with a transfer learning framework that leverages abundant SWE data from California to enhance predictions in Colorado, where data are scarce. From 2016 to 2019, the disparity in SWE data collection between these states was stark: 94 snowpack maps were recorded in California's Sierra Nevada versus only 12 in Colorado's Rocky Mountains. We hypothesized that geographic predictors (e.g., elevation and snowfall) would exhibit similar effects on SWE across these landscapes. By conducting an explanatory factor analysis, we validated this hypothesis and refined our transfer learning model, which incorporated data from California to predict SWE in Colorado. When compared with using data from Colorado alone, transfer learning improved the mean R$^2$ value from 0.43 to 0.56, indicating a significant enhancement of over 30\% in predictive accuracy. Such advancements underscore the potential of our framework to mitigate lidar data limitations, offering a valuable tool for water resource management amidst changing environmental conditions. 

## Datasets

• Lidar data: we obtained lidar-derived SWE from the Airborne Snow Observatory (ASO) with a 50 m resolution (T. Painter, 2018). ASO data in Colorado consist of 12 maps across five basins from 2016 to 2019: Blue River (BR), Crested Butte (CB), Maroon/Castle Creek (CM), Gunnison-East River (GE), and Gunnison-Taylor River (GT) basins. ASO data in California are more frequent and consist of 94 maps across 13 basins, serving as a rich source for transfer learning. Following quality control measures, we considered 80 maps across 11 basins from 2013 to 2019: Cherry Eleanor (CE), Kings Canyon (KC), Kings North (KN), Lakes Basin (LB), LEE Vining Creek (LV), Merced River (MB), Rush Creek (RC), San Joaquin South Fork (SF), San Joaquin Main Fork (SJ), Tuolumne River (TB), Tuolumne + Cherry/Eleanor (TE).

• Parameter elevation Relationships on Independent Slopes Model (PRISM) data: We obtained gridded estimates of daily preciptation and temperature from the PRISM data (Daly et al., 2008) at a spatial resolution of 800 m. Using these datasets, we derived four meteorological variables that are strongly correlated to the spatial distribution of SWE (Mital et al., 2022): accumulated snow, sum positive degree days (PDD), accumulated precipitation, and mean seasonal temperature.

• Elevation maps from National Elevation Dataset (Gesch et al., 2018): These maps were used to extract topographic variables that influence snow melt and snow accumulation processes. The topographic variables are: slope, aspect, and elevation. 

<div align=center><image src="./Figures/spatial_extent.jpg"></div>
<p align=justify>
Figure 1: Spatial extent and frequency of lidar-derived SWE maps used in this study. Basin names are abbreviated for brevity. 
</p> 

| File | Description |
| ------------- | ------------- |
| `Base_models.zip` | The 5 trained Base models. The Base model used to train TL models is Base model 3. |
| `Colorado_ScaledLMs.zip` | 12 Local 1 models |
| `Unscaled_LM_models.zip` | 12 Local 2 models |
| `TL_1_models.zip` | 12 TL 1 models |
| `TL_2_3.zip` | 12 TL 2 and 12 TL 3 models |
  
## Transfer Learning and Benchmark Models
We adopted a feed-forward Artificial Neural Network (ANN) architecture, initially training a base model on the source data which corresponds to California's 80 SWE maps. Subsequently, we considered three different modeling approaches to adapt the base model to perform the target task of predicting SWE in Colorado (Figure \ref{TL1, 2, 3}). The first two approaches were: 

• Model TL1: This involved freezing the shallower layers (preventing their weights from changing) and retraining only the deeper layers.
• Model TL2: This involved freezing all the weights of the model, removing deeper layers, and adding new layers whose weights were trained on the target data.

The two approaches were picked because the deeper layers help capture the higher order complexities in the relationship between input features and the output (Larochelle232
et al., 2007, 2009), while the shallower layers generally capture coarser and simpler relationships. Complex relationships in the deeper layers of the base model may be more particular to the source task, while the information in the shallower layers may be more easily generalized to the target task. Additionally, it is important to mention that in our analysis, permutation feature importance applied to the transfer learning models revealed that elevation was not a predominant factor during training (section 3.2). This finding contrasted with our EFA results (section 3.1), which highlighted elevation as a critical variable in determining SWE values in Colorado. Therefore, we developed a third transfer learning approach where the importance of elevation was prescribed:

• Model TL3: Here, the input variables were not scaled. We observed that elevation has a broad range of variation compared to other predictor variables. Not scaling the input biased the ANN optimizer to give more importance to elevation during training. Otherwise, the approach was similar to TL1 and TL2.

<div align=center><image src="./Figures/TL_schematic0.jpeg"></div>
<p align=justify>
Figure 2: Schematic describing the different TL models considered in this study. For brevity, only one version of model TL3 is shown
</p> 

The performance of transfer learning models was benchmarked against local models trained only on data from Colorado. This helps to validate the added value of transfer learning in improving SWE prediction accuracy. Through this structured approach,we demonstrate a methodological framework that could be applied to other regions facing challenges of data limitation. We considered two versions of local models: Local 1 considers scaled input variables per the usual machine learning practice, while Local 2 prescribes importance to elevation in a manner similar to model TL3.

| File | Description |
| ------------- | ------------- |
| `Base_models.zip` | The 5 trained Base models. The Base model used in transfer learning is Base model 3. |
| `Colorado_ScaledLMs.zip` | 12 Local 1 models |
| `Unscaled_LM_models.zip` | 12 Local 2 models |
| `TL_1_models.zip` | 12 TL 1 models |
| `TL_2_3.zip` | 12 TL 2 and 12 TL 3 models |

## Code Description
All the codes are writen in Python 3.9.0. The deep learning models are implemented using deep learning framework Keras/TensorFlow. Bayesian Hyperparameter optimization was conducted using the opensource framework Ax, Adaptive Experimentation platform, following this tutorial: (https://www.justintodata.com/hyperparameter-tuning-with-python-keras-guide/). The hyperparameter search space included activation function, feature scaling techniques, optimization function, learning rate, number of hidden layers, number of neurons per layer, dropout rate, L1 and L2 regularization rates, and batch size. For TL models, additional hyperparameters were: number of frozen hidden layers, number of removed hidden layers, and number of added hidden layers. 

| File | Description |
| ------------- | ------------- |
| `California base models .ipynb/` | Jupyter Notebook used for hyperparameter optimization, training and testing of ANNs on California data to predict SWE in California. 5 models are trained each with a different training/validation split. The models are called California base models.|
| `Colorado local Models .ipynb.zip` | Jupyter Notebook used for hyperparameter optimization, training and testing of ANNs on Colorado data to predict SWE in Colorado. 24 models are trained, 12 models with scaled input features and 12 models without scaled input features. Each 12 models are trained and tested using the leave-one-out method since there are 12 SWE maps. The models are called Colorado local models.  |
| `TL1_models_and_permutation_feature_importance.ipynb` | Jupyter Notebook used for transfer learning according to approach 1 and applies permuation feature importance on TL2 models.  |
| `TL2_TL3_models.ipynb` | Jupyter Notebook used for transfer learning according to approaches 2 and 3. |
| `FA-winter/FA_summer.ipynb` | Jupyter Notebook used to apply explanatory factor analysis on California and Colorado winter/summer data. |
| `California_data_processing.ipynb/Colorado_data_processing.ipynb` | Jupyter Notebook used to process the raw data (ASO Lidar-maps, Elevation maps, and PRISM data) and generates the datasets (CSV files) for training ANNs. |
| `SWE_maps_plots.ipynb` | Jupyter Notebook used to plot Colorado true and TL predicted SWE maps (scatter plots). |

## Factor Analysis Results

## SWE Prediction Results

To evaluate model performance, we computed the coefficient of determination $R^2$ on the test data. $R^2$ is generally used in regression models to quantify the proportion of the variance in the dependent variable that is predictable by the independent variables \cite{chicco2021coefficient}. $R^2$ is calculated as follows:
\begin{equation}
 R^2 = 1 - \frac{\sum_{i = 1}^N (y_i - \hat{y}_i)^2}{\sum_{i = 1}^N (y_i - \bar{y})^2}
\end{equation}

## References


