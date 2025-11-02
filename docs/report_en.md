# SUNS Assignment 2
Michal Balogh, xbaloghm1@stuba.sk
October 2025

# 1. Data Preparation

Data preparation and processing for further tasks are in the file `bike_rental.ipynb`, in section 1. Data Preparation. I loaded the data using the pandas library into a dataframe. The dataset contains 8741 rows and 12 columns.

First, I removed the `instant` column, as it is just a row identifier and carries no information. I also removed the `date` column, as it would be poorly encoded for the models we will use (365 unique values). Besides that, we have time information from the month, weekday, and hour columns. We don't need to preserve the year because all data is from one year - 2012.

In the next step, I checked for missing and duplicate values. In the dataset, 11 values were missing in the holiday column (0.13%). Since this is a very small part of the data, I removed these rows. I also removed 8 duplicate rows.

## 1.1 Removal of data not belonging to the specified range
According to the specification, the dataset contains data with the following ranges for continuous attributes:
- temperature: -40,40
- humidity: 0,100
- windspeed: 0,110

And for discrete attributes:
- month: 1,12
- hour: 0,23
- weekday: 0,6
- holiday: 0,1
- workingday: 0,1
- count: >=0

According to these rules, I removed 22 rows that had negative values in the `humidity` column.

## 1.2 Encoding of categorical attributes
In the dataset, we had 12 columns, of which we removed 2. Of the remaining 10 columns, 9 are numeric and only 1 is categorical - `weather`. I encoded it using `label encoding`, as it has only 4 unique values and the order between them makes sense - from the best to the worst weather.
Encoding:
- clear: 0
- cloudy: 1
- light rain/snow: 2
- heavy rain/snow: 3

## 1.3 Outlier analysis
For outlier analysis, I used the 1.5*IQR rule for continuous attributes, as it doesn't make sense for others. Only the `windspeed` column has outliers, 175 values (2.01%). I removed these values. The boxplot with outliers is in the image outliers_windspeed.

![outliers_windspeed](figures/outliers_windspeed.png)

## 1.4 Final dataset dimensions
After data processing, I have a final dataset with dimensions 8525 rows and 10 columns.

# 2. EDA

First, I analyzed the average number of rented bikes by hour and month. The graph is in the image eda_rentals_hour_and_month.
![eda_rentals_hour_and_month](figures/eda_rentals_hour_and_month.png)
From the graph, we can see that bikes are rented more in warm months (season) - from April to September. The highest rentals are in September.
Regarding hours, people rent bikes the most in the morning around 8 o'clock, and then in the afternoon around 17-18 o'clock. These are times when people go to work and from work.

Another graph shows the average number of rented bikes in a given hour on working days (pink) and on weekends and holidays (blue). The graph is in the image eda_rentals_pattern.
![eda_rentals_pattern](figures/eda_rentals_pattern.png)
From the graph, we can see that on working days there is a sharp increase in bike rentals in the morning around 7-9 o'clock and then in the afternoon around 16-18 o'clock. On the other hand, on weekends and holidays, bike rentals are distributed more evenly throughout the day, with peaks around 11-15 o'clock, when people use bikes for recreation.

The graph eda_weather shows the distribution of weather in the dataset. In the graph on the left is the average and median number of rentals for individual weather categories.
![eda_weather](figures/eda_weather.png)
The most bikes are rented during clear and cloudy weather.

The graph eda_heatmap shows the average number of rented bikes by hour and month in a heatmap.
![eda_heatmap](figures/eda_heatmap.png)
The clearest peak on the heatmap is in the months August, September, and October around 17-18 o'clock. Other clear peaks are in the months March to July, also around 17-18 o'clock. Besides that, also in March around 8 o'clock in the morning.

The summary of rental peaks during working and non-working days is in the table eda_rentals_table.
![eda_rentals_table](figures/eda_rentals_table.png)

# 3. Splitting data into training and testing

I split the data into training and testing in a ratio of 8:2 using the `train_test_split` function from the `sklearn.model_selection` library.

I scaled the data using `StandardScaler`.

# 4. Model Training

Model training is in the file `bike_rental.ipynb`, in section 4. Model Training. I used three models: Decision Tree Regressor, Random Forest Regressor, and Support Vector Machine from the `sklearn` library.

## 4.1 Decision Tree
To find the best depth of the decision tree, I tried values from 1 to 100 in a loop. I found the best value at depth 10. The model achieved R2 0.889 on test data.

The table with results for tree depth 1 to 10 is in the image tree_max_depth_table.
![tree_max_depth_table](figures/tree_max_depth_table.png)

Since depth 10 is already poorly visualized, I chose depth 3 for tree visualization. The visualization of the decision tree is in the image tree_viz.

![tree_viz](figures/tree_viz.png)

In the tree, we can see that the most important attribute for decision making is the `hour` attribute. It is in the root of the tree and also in four other nodes - a total of 5 out of 7 decision nodes. Next is the important attribute `temperature` and `workingday`.

This tree with depth 3 achieved R2 0.525 on test data.

## 4.2 Random Forest

For Random Forest, I tried different numbers of trees in the forest (n_estimators) and decided on the value 10, which although did not achieve the best results, was fast enough and increasing the number of trees no longer improved the results much.

The model achieved R2 0.925 on test data with 10 trees in the forest. With 100 trees in the forest, the model achieved R2 0.934, which is an improvement of only 0.009 compared to 10 trees.

The features by which Random Forest decided are shown in the image importance_of_input_features.
![importance_of_input_features.png](figures/importance_of_input_features.png)

It was confirmed that the most important attribute is `hour`, then `temperature` and `workingday`. The `hour` attribute provides approximately 70% importance in the model's decision making, then `temperature` approximately 13% and `workingday` around 6%. The remaining attributes already have similarly low importance.

## 4.3 Support Vector Machine (SVM)

For support vector machine, I tried different kernels and achieved the best results with the 'rbf' kernel (radial basis function). I set the C parameter (regularization) to 100, which achieved better results than lower values.

The model achieved an R2 score of 0.554.

## 4.4 Comparison and evaluation of models
The comparison of all three models (Decision Tree, Random Forest, and SVM) is listed in the table below. The models were evaluated using R2, RMSE, MSE metrics on training and test data.

![model_comparison_table](figures/model_comparison_table.png)

Random Forest achieved the best results out of all three models:
- Test R2 score: 0.925, meaning the model explains 92.5% of the variability in the data
- Test RMSE: 57.00, representing the average prediction deviation
- High training R2 (0.989) indicates slight overfitting, but the difference from test R2 is not dramatic

Decision Tree with optimal depth 10 achieved:
- Test R2 score: 0.889, which is 0.036 worse than Random Forest
- Test RMSE: 69.17
- Significant difference between training (0.950) and test R2 indicates slight overfitting

Support Vector Machine (SVM) achieved the worst results:
- Test R2 score: 0.554, significantly lower than tree models
- Test RMSE: 138.77, more than double that of Random Forest
- Similar results on training and test data (0.579 vs 0.554) indicate that the model is not overfitted, but just cannot capture the relationships between the data well with the given hyperparameter settings

#### Residual analysis

Residual plots (differences between actual and predicted values) for all three models are shown below:

Decision Tree:
![tree_residuals](figures/tree_residuals.png)

Random Forest:
![forest_residuals](figures/forest_residuals.png)

SVM:
![svm_residuals](figures/svm_residuals.png)

From the residual graphs, it can be observed that Random Forest has the smallest dispersion of residuals and best approximates the ideal state (residuals around zero). The residual distribution is symmetric around zero (approximately normally distributed), which is a sign of a good model.

Decision Tree has slightly larger dispersion, but still acceptable. The residual distribution is slightly skewed to the left.

SVM already has significant errors with large residual dispersion. In the residual distribution graph, the data is skewed to the right, meaning the model tends to underestimate some high values (predictions are too low compared to reality).
In other words - the model cannot capture extreme high values of the target variable well.

#### Comparison of R2 score on train and test set

Visual comparison of R2 score for all models for training and test data is in the graph r2_comparison.

![r2_comparison](figures/r2_comparison.png)

From the graph, it is visible that there is not such a big difference between training and test data. From this, it follows that the models are not overfitted.

#### Predictions and actual values

Graphs comparing predicted and actual values for the test set for all models are in the graphs below.

Decision Tree:
![tree_predictions](figures/tree_predictions.png)

Random Forest:
![forest_predictions](figures/forest_predictions.png)

SVM:
![svm_predictions](figures/svm_predictions.png)

Ideal predictions should lie on the red line (y = x). Most points stay close to the diagonal, especially at lower values, so the model performs quite well for smaller real values. For SVM at higher values (e.g., above 400-500), the points start to scatter below the line, which again confirms that the model underestimates higher values.

# 5. Dimensionality Reduction

In this part, I visualized the data in 3D space using two methods: 3D scatter plot, where I selected 3 features, and PCA (Principal Component Analysis).

## 5.1 3D Scatter Plot

As the 3 features that I plotted in the 3D graph, I chose `hour`, `temperature`, and `weather`. The first two are the most important features according to the feature importance analysis from the Random Forest model. The third feature is `weather`, which I chose because it is easier to interpret. The `weather` attribute has 4 unique values, represented by numbers 0-3 after label encoding.

The graph is in the image 3d_plot.
![3d_plot](figures/3d_plot.png)

The `weather` attribute divided the data into nice 4 clusters. In the last cluster - `heavy rain/snow` (3) there is only one dot - during very unfavorable weather, bike rentals are very low. On the other hand, in `clear` weather (0), there is the largest number of points, and from the other two axes we can deduce that the most rentals are between 14-16 o'clock and at temperatures around 20-30 degrees.

## 5.2 PCA

After applying PCA to the scaled training data, I obtained the graph in the image pca_3d.
![pca_3d](figures/pca_3d.png)

Most points are in a dense area, meaning these data have similar properties. There are apparently no clear clusters. Dots with high count values (yellow) are concentrated relatively close in clusters of the graph, which may indicate that there are combinations of conditions (weather, day of the week, hour...) that lead to high bike activity. Outliers outside the main cluster may represent specific situations, such as extreme weather or holidays.

After reducing the dimension to three main PCA components, I analyzed the weights (loadings) of the original variables on these components:
| Component | Explained variability | Main weights (loadings)                     | Interpretation                      |
|-----------|-----------------------|---------------------------------------------|-------------------------------------|
| PC1       | 19.2 %                | +humidity, +weather, –temperature, –windspeed | Weather (unfavorable vs. favorable) |
| PC2       | 14.9 %                | +holiday, –workingday, –weekday             | Type of day (weekend/holiday vs. working day) |
| PC3       | 13.2 %                | +month, +temperature, –windspeed            | Season (summer vs. winter)          |
| Total     | 47.3 % variability    | –                                           | Almost half the information in the data |

# 6. Training on subset of features

In this part, I trained the best model - Random Forest on a reduced set of features. I reduced the set of features using three methods:
- according to the correlation matrix
- according to feature importance from Random Forest
- using PCA

## 6.1 Correlation matrix

The correlation matrix of features is in the image corr_mat.
![corr_mat](figures/corr_mat.png)

I selected features that correlated with the target variable `count` more than the set threshold 0.1 as new features for model training. There were 5 such features: `hour`, `temperature`, `humidity`, `weather`, and `windspeed`.

The model achieved R2 0.694 on test data. Test RMSE worsened to 114.86.

## 6.2 Feature importance from Random Forest

The feature importance graph from Random Forest was already in section 4.2, in the image importance_of_input_features.

With cumulative sum, I wanted to cover at least 90% of feature importance. I achieved this only with 3 features: `hour`, `temperature`, and `workingday`. This means that these 3 features account for 90% of the model's decision making.

The model achieved R2 0.839 on test data. Test RMSE worsened to 83.30.

## 6.3 PCA

For PCA, I chose a threshold of 90% explained variability. I achieved this value with eight components. Eight components covered 95.3% of data variability.
The cumulative sum of explained variability is in the image cumsum_expl_var.
![cumsum_expl_var](figures/cumsum_expl_var.png)

The model achieved R2 0.583 on test data. Test RMSE worsened to 134.14.

## 6.4 Comparison of results on reduced feature set

The comparison of Random Forest model results on reduced feature set is in the table feature_selection_methods_table.
![feature_selection_methods_table](figures/feature_selection_methods_table.png)

The best results were achieved on the original feature set. Reducing the feature set caused worsening of results in all cases. The results worsened the least when selecting features according to feature importance from Random Forest. The R2 metric worsened by 0.086 and RMSE worsened by 26.30.

From the results, it follows that all features in the dataset bring some value to the model and none of them should be removed unless there is a serious reason.

The comparison of R2 metric on training and test set for individual feature reductions is in the graph features_barplot. Next to it is a graph showing how many features were used in each method.
![features_barplot](figures/features_barplot.png)

The residual graph for individual methods and residual distribution is in the image features_residuals.
![features_residuals](figures/features_residuals.png)

From the graph, it is visible that the model trained on the original feature set has the smallest residual dispersion. Models on reduced sets have larger dispersion and their residuals are less symmetric around zero. Feature selection according to feature importance from Random Forest has the smallest dispersion among reduced sets, but significantly underestimates high values. This is also visible in the residual distribution graph, where the residuals are skewed to the right.

# 7. Data Clustering

I clustered the data using KMeans into 6 clusters. In the 3D graph, the clusters are displayed in colors. On the axes, I put 3 continuous attributes: `humidity`, `windspeed`, and `temperature`. The graph is in the image clusters.

![clusters](figures/clusters.png)

We don't see clearly separated clusters, which indicates that the data is quite evenly distributed in space and is not naturally clustered.

I then trained the best model, Random Forest, on individual clusters and compared the predictions with the original model. The table with results and weighted average is in the image clusters_vs_original_table.

![clusters_vs_original_table](figures/clusters_vs_original_table.png)

For most clusters, the model achieved better results than the original model. Only in clusters 3 and 5 was the result worse. This may be because these clusters contain quite few data, which causes overfitting.

The weighted average of all clusters achieved R2 0.921, which is worse than the original model with R2 0.925, but test RMSE decreased from 57.00 to 51.99, which is a significant improvement.

A more detailed analysis of individual clusters is in the table clusters_analysis_table.
![clusters_analysis_table](figures/clusters_analysis_table.png)

# 8. Neural Network

Since it is regression, the output of the neural network is one continuous value - the predicted number of bike rentals.

The network architecture contains:
- Input layer with 9 neurons (9 features after data processing)
- 3 hidden layers with 128, 64, and 32 neurons
- Output layer with 1 neuron (prediction of continuous value)
- ReLU activation function in each hidden layer

I used Adam as the optimizer. I chose Mean Squared Error (MSE) as the loss function, since it is regression.

The hyperparameters for the network are set as follows:
+--------------------------+----------------+
| Hyperparameter           | Value          |
+--------------------------+----------------+
| Learning Rate            | 1e-3           |
| Batch Size               | 256            |
| Epochs                   | 500            |
| Early Stopping Patience  | 30             |
+--------------------------+----------------+

The training progress of the network and the development of R2 score is in the image training_curves.
![training_curves](figures/training_curves.png)

Training stopped after 406 epochs thanks to early stopping, when the R2 score on the validation set stopped improving.
The model achieved R2 0.9379 and RMSE 53.87 on test data.

![nn_residuals](figures/nn_residuals.png)

The residual graph shows that the residuals are symmetric around zero and most points are close to zero, which indicates a good model. The residual distribution is approximately normal, with slight skewness to the right, which again indicates that the model tends to underestimate very high values.
