# Modeling Spatial Distribution of Snow Water Equivalent using Transfer Learning across Mountainous Basins

Accurately estimating snow water equivalent (SWE) is crucial for understanding the impacts of climate change, urbanization, and population growth on water resources. High operational costs of lidar observations limit the frequency and coverage of SWE estimates at high spatial resolutions, leading to significant data gaps. We address this challenge with a transfer learning framework that leverages abundant SWE data from California to enhance predictions in Colorado, where data are scarce. From 2016 to 2019, the disparity in SWE data collection between these states was stark: 94 snowpack maps were recorded in California's Sierra Nevada versus only 12 in Colorado's Rocky Mountains. We hypothesized that geographic predictors (e.g., elevation and snowfall) would exhibit similar effects on SWE across these landscapes. By conducting an explanatory factor analysis, we validated this hypothesis and refined our transfer learning model, which incorporated data from California to predict SWE in Colorado. When compared with using data from Colorado alone, transfer learning improved the mean R$^2$ value from 0.43 to 0.56, indicating a significant enhancement of over 30\% in predictive accuracy. Such advancements underscore the potential of our framework to mitigate lidar data limitations, offering a valuable tool for water resource management amidst changing environmental conditions. 

## Code Description
All the codes are writen in Python 3.9.0. The deep learning models are implemented using deep learning framework Keras/TensorFlow. Bayesian Hyperparameter optimization was condcuted using the opensource framework Ax, Adaptive Experimentation platform, following this tutorial: https://www.justintodata.com/hyperparameter-tuning-with-python-keras-guide/. 

| File | Description |
| ------------- | ------------- |
| `data_processing/data_preprocess_snapshot_only.ipynb` | Jupyter Notebook used to capture images from the video stream at designated frequency. |
| `data_processing/data_preprocess_pv.ipynb` | Jupyter Notebook used to process the raw PV power generation history.  |
| `data_processing/data_nowcast.ipynb` | Jupyter Notebook used to down-sample the image frames, filter out the invalid frames and match images with the concurrent PV data, and partition model development and testing sets.  |
| `data_processing/data_forecast.ipynb` | Jupyter Notebook used to generate valid samples for the forecast task. |
| `models/SUNSET_nowcast.ipynb` | Jupyter Notebook used to create the SUNSET nowcast model to correlate PV output to contemporaneous images of the sky, including model training, validation and testing. |
| `models/SUNSET_forecast.ipynb` | Jupyter Notebook used to create the SUNSET forecast model to predict 15-min ahead minutely-averaged PV output, including model training, validation and testing. |
| `models/Relative_op_func.py` | Helper functions for calculating theoretical PV power output under clear sky condition and the clear sky index. |
