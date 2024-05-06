# Modeling Spatial Distribution of Snow Water Equivalent using Transfer Learning across Mountainous Basins

Accurately estimating snow water equivalent (SWE) is crucial for understanding the impacts of climate change, urbanization, and population growth on water resources. High operational costs of lidar observations limit the frequency and coverage of SWE estimates at high spatial resolutions, leading to significant data gaps. We address this challenge with a transfer learning framework that leverages abundant SWE data from California to enhance predictions in Colorado, where data are scarce. From 2016 to 2019, the disparity in SWE data collection between these states was stark: 94 snowpack maps were recorded in California's Sierra Nevada versus only 12 in Colorado's Rocky Mountains. We hypothesized that geographic predictors (e.g., elevation and snowfall) would exhibit similar effects on SWE across these landscapes. By conducting an explanatory factor analysis, we validated this hypothesis and refined our transfer learning model, which incorporated data from California to predict SWE in Colorado. When compared with using data from Colorado alone, transfer learning improved the mean R$^2$ value from 0.43 to 0.56, indicating a significant enhancement of over 30\% in predictive accuracy. Such advancements underscore the potential of our framework to mitigate lidar data limitations, offering a valuable tool for water resource management amidst changing environmental conditions. 

## Datasets

• Lidar data: we obtained lidar-derived SWE from the Airborne Snow Observatory (ASO) with a 50 m resolution (T. Painter, 2018). ASO data in Colorado consist of 12 maps across five basins from 2016 to 2019: Blue River (BR), Crested Butte (CB), Maroon/Castle Creek (CM), Gunnison-East River (GE), and Gunnison-Taylor River (GT) basins. ASO data in California are more frequent and consist of 94 maps across 13 basins, serving as a rich source for transfer learning. Following quality control measures, we considered 80 maps across 11 basins from 2013 to 2019: Cherry Eleanor (CE), Kings Canyon (KC), Kings North (KN), Lakes Basin (LB), LEE Vining Creek (LV), Merced River (MB), Rush Creek (RC), San Joaquin South Fork (SF), San Joaquin Main Fork (SJ), Tuolumne River (TB), Tuolumne + Cherry/Eleanor (TE).

• Gridded meteorological datasets: We obtained gridded estimates of daily preciptation and temperature from the Parameter elevation Relationships on Independent Slopes Model (PRISM) data (Daly et al., 2008) at a spatial resolution of 800 m. Using these datasets, we derived four meteorological variables that are strongly correlated to the spatial distribution of SWE: accumulated snow, sum positive degree days (PDD), accumulated precipitation, and mean seasonal temperature.

• Elevation maps: We obtained elevation maps from National Elevation Dataset (Gesch et al., 2018). These maps were used to extract topographic variables that influence snow melt and snow accumulation processes: slope, aspect, and elevation. 

These datasets were rescaled to 800 m and reprojected to a consistent coordinate system for consistency.

<div align=center><image src="./Figures/spatial_extent.jpg"></div>
<p align=center>
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
We adopted a feed-forward Artificial Neural Network (ANN) architecture, initially training a base model on the source data which corresponds to California's 80 SWE maps. Subsequently, we considered three different modeling approaches to adapt the base model to perform the target task of predicting SWE in Colorado (Figure 2). The first two approaches were: 

• Model TL1: This involved freezing the shallower layers (preventing their weights from changing) and retraining only the deeper layers.
• Model TL2: This involved freezing all the weights of the model, removing deeper layers, and adding new layers whose weights were trained on the target data.

The two approaches were picked because the deeper layers help capture the higher order complexities in the relationship between input features and the output, while the shallower layers generally capture coarser and simpler relationships. Complex relationships in the deeper layers of the base model may be more particular to the source task, while the information in the shallower layers may be more easily generalized to the target task. Additionally, it is important to mention that in our analysis, permutation feature importance applied to the transfer learning models revealed that elevation was not a predominant factor during training. This finding contrasted with our EFA results, which highlighted elevation as a critical variable in determining SWE values in Colorado. Therefore, we developed a third transfer learning approach where the importance of elevation was prescribed:

• Model TL3: Here, the input variables were not scaled. We observed that elevation has a broad range of variation compared to other predictor variables. Not scaling the input biased the ANN optimizer to give more importance to elevation during training. Otherwise, the approach was similar to TL1 and TL2.

<div align=center><image src="./Figures/TL_schematic0.jpeg"></div>
<p align=center>
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
All the codes are writen in Python 3.9.0. The deep learning models are implemented using deep learning framework Keras/TensorFlow (Chollet et al., 2015) . Bayesian Hyperparameter optimization was conducted using the opensource framework Ax, Adaptive Experimentation platform (Bakshy et al., 2018), following this tutorial: (https://www.justintodata.com/hyperparameter-tuning-with-python-keras-guide/). The hyperparameter search space included activation function, feature scaling techniques, optimization function, learning rate, number of hidden layers, number of neurons per layer, dropout rate, L1 and L2 regularization rates, and batch size. For TL models, additional hyperparameters were: number of frozen hidden layers, number of removed hidden layers, and number of added hidden layers. Lastly, Factor Analysis was conducted using the package FactorAnalyzer in Python with varimax rotation (Biggs & Madnani, 2021). 

| File | Description |
| ------------- | ------------- |
| `California base models .ipynb/` | Jupyter Notebook used for hyperparameter optimization, training and testing of ANNs on California data to predict SWE in California. 5 models are trained each with a different training/validation split. The models are called California base models.|
| `Colorado local Models .ipynb.zip` | Jupyter Notebook used for hyperparameter optimization, training and testing of ANNs on Colorado data to predict SWE in Colorado. 24 models are trained, 12 models with scaled input features and 12 models without scaled input features. Each 12 models are trained and tested using the leave-one-out method since there are 12 SWE maps. The models are called Colorado local models.  |
| `TL1_models_and_permutation_feature_importance.ipynb` | Jupyter Notebook used for transfer learning according to approach 1 and applies permuation feature importance on TL2 models.  |
| `TL2_TL3_models.ipynb` | Jupyter Notebook used for transfer learning according to approaches 2 and 3. |
| `FA-winter/FA_summer.ipynb` | Jupyter Notebooks used to apply explanatory factor analysis on California and Colorado winter (March/April) and summer (June) data. |
| `California_data_processing.ipynb/Colorado_data_processing.ipynb` | Jupyter Notebook used to process the raw data (ASO Lidar-maps, Elevation maps, and PRISM data) and generates the datasets (CSV files) for training ANNs. |
| `SWE_maps_plots.ipynb` | Jupyter Notebook used to plot Colorado true and TL predicted SWE maps (scatter plots). |

## Factor Analysis Results

We developed four explanatory factor analysis (EFA) models: two EFA models describing Colorado datasets (one each for March-April and June), and two EFA models describing California datasets (one each for March-April and June). EFA models captured between 0.64 to 0.72 of the total variance in our dataset. All models were able to capture a large proportion of the variance in elevation, accumulated snow, accumulated precipitation, sum PDD (with the exception for March-April in Colorado), and $T_{mean}$. For SWE, the four EFA models were able to capture 0.75, 0.78, 0.74, and 0.57 of the variance.

To further investigate the regional and seasonal variability in how the predictor variables affect SWE, we resorted to factor loading plots shown in Figures 3 and 4. These plots illustrate the association between variables and latent factors. Each latent factor can be considered to be a proxy or latent representation of underlying physical phenomena that influence snowpack. Factors are arranged based on the amount of variance they capture from the data, listed in descending order. Three factors were found to fit Colorado datasets best, while two factors yielded the best fit for California datasets.

Colorado in March/April         |  California in March/April
:-------------------------:|:-------------------------:
![](/Figures/COL_FA_loadingplot_winter_equal_y.jpg) |  ![](/Figures/CA_FA_loadingplot_winter_equal_y.jpg )

<p align=center>
Figure 3: EFA Loading plots illustrating the principal factors for Colorado and California datasets in March/April. The x-axis represents the variable name, while the y-axis represents the variable loading.
</p> 

Colorado in June         |  California in June
:-------------------------:|:-------------------------:
![](/Figures/COL_FA_loadingplot_summer_equal_y.jpg) |  ![](/Figures/CA_FA_loadingplot_summer_equal_y.jpg )

<p align=center>
Figure 4: EFA Loading plots illustrating the principal factors for Colorado and California datasets in June. The x-axis represents the variable name, while the y-axis represents the variable loading.
</p> 

The analysis reveals consistent patterns across both Colorado and California, with elevation and accumulated snow consistently driving high SWE values throughout different months. Additionally, low temperatures are consistently related to high SWE values in both regions across various time periods. This indicates that transfer learning could be implemented to predict SWE in Colorado using data from California. However, disparities emerge in the influence of precipitation and temperature-driven processes, with Colorado showing a stronger dependence on precipitation and California exhibiting a more pronounced sensitivity to temperature-related factors, particularly in March-April. These point to differences in higher-order complexities in relationships between the predictors and SWE across Colorado and California. This is a key insight for implementing transfer learning whose results are presented next

## SWE Prediction Results

To evaluate model performance, we computed the coefficient of determination $R^2$ on the test data. $R^2$ is generally used in regression models to quantify the proportion of the variance in the dependent variable that is predictable by the independent variables. We also calculated the normalized bias $b$. Bias is an indicator of how much our mean predicted SWE values deviate from the mean true SWE values.
We see that TL3 had the best performance with the highest mean $R^2$ and second lowest standard deviation about the mean of $R^2$, and the lowest mean absolute value of bias between all model types. The standard deviation of TL models is less than that of the local models. Overall, transfer learning enhanced the accuracy and robustness of SWE predictions in Colorado.

<div align=center><image src="./Figures/SWE_results.png"></div>
<p align=center>
Table 1: $R^2$ values for modeling SWE. 
</p> 

Figure 5 presents the statistical distribution of feature importances for models TL2 and TL3. Overall, unscaling raised the importance of elevation and reduced the importance of Tmean. EFA loading plots indicated that an ML approach should replicate the influence of elevation and accumulated snow on SWE. A feature importance analysis revealed that elevation was given a low priority during model training (TL2). This motivated us to develop a strategy to manually prescribe importance to elevation during training. Since the purpose of feature scaling is to ensure that optimization of model weights is affected equally by each predictor variable, we chose to train a model without scaling. All variables other than elevation exhibited a similar (and smaller) range of variation compared to elevation, therefore unscaling served as a simple and effective strategy to prescribe more importance to elevation. In general, unscaling may not always yield optimal results and other feature engineering approaches may need to be considered.

TL 2      | TL 3
:-------------------------:|:-------------------------:
![](/Figures/BOXplot_TL2.jpg) |  ![](/Figures/BOXplot_TL3.jpg )

<p align=center>
Figure 5: Box plot visualization of feature importance of TL2 and TL3.
</p> 

## References
<a id="1">[1]</a> 
Painter, T. (2018). Aso l4 lidar snow water equivalent 50m utm grid, version 1, nasa national snow and ice data center distributed active archive center, boulder, colorado usa.

<a id="1">[2]</a> 
Daly, C., Halbleib, M., Smith, J. I., Gibson, W. P., Doggett, M. K., Taylor, G. H.,. . . Pasteris, P. P. (2008). Physiographically sensitive mapping of climatological temperature and precipitation across the conterminous united states. International Journal of Climatology: a Journal of the Royal Meteorologica Society, 28 (15), 2031–2064

<a id="1">[3]</a> 
Gesch, D. B., Evans, G. A., Oimen, M. J., & Arundel, S. (2018). The national elevation dataset. https://apps.nationalmap.gov. U.S. Geological Survey. (Accessed: 2022-04-15)

<a id="1">[4]</a> 
Biggs, J., & Madnani, N. (2021). factor analyzer. GitHub. Retrieved from https://github.com/EducationalTestingService/factor\ analyzer/blob/main/factor\ analyzer/factor\ analyzer.py

<a id="1">[5]</a> 
Bakshy, E., Dworkin, L., Karrer, B., Kashin, K., Letham, B., Murthy, A., & Singh,S. (2018). Ae: A domain-agnostic platform for adaptive experimentation.. Retrieved from https://api.semanticscholar.org/CorpusID:73557896

<a id="1">[6]</a> 
Chollet, F., et al. (2015). Keras. GitHub. Retrieved from https://github.com/fchollet/keras

<a id="1">[7]</a> 
Mital, U., Dwivedi, D.,  ̈Ozgen Xian, I., Brown, J. B., & Steefel, C. I. (2022, October). Modeling Spatial Distribution of Snow Water Equivalent by Combinin Meteorological and Satellite Data with Lidar Maps. Artificial Intelligence for the Earth Systems, 1 (4), e220010. Retrieved 2022-12-08, from https://journals.ametsoc.org/view/journals/aies/1/4/AIES-D-22-0010.1.xml 
doi: 10.1175/AIES-D-22-0010.1
