# MLP-MCDA
Framework for ranking prediction based on Multi-Layer Perceptron (MLP) regressor model and historical datasets evaluated by experts using Multi-Criteria Decision Analysis (MCDA) methods in Python.

The `main_ann.py` file includes:

- Application of machine learning models from `scikit-learn` Python library:

	- `MLPRegressor`
	- `LinearRegression`
	
- And other methods:

	- `GridSearchCV`
	- `cross_val_score`
	- `r2_score`
	- `train_test_split`
	
- This framework uses the TOPSIS method from `pyrepo-mcda` Python package. You can install it via the pip command: 

```
pip install pyrepo-mcda
```

- And Gini coefficient-based weighting method from `crispyn` Python package. You can install it via the pip command:

```
pip install crispyn
```

- Preparation of training and test datasets with feature values.
- Generation of the target variable representing MCDA score.
- Splitting dataset to train and test.
- Selection of the best hyper-parameters for MLP regressor model using `GridSearchCV`.
- Training and testing MLP regressor model in prediction rankings.
- Comparing `MLPRegressor model` with `LinearRegression` model.
- Determining the correlation between rankings.
- Results visualizations using column, line, scatter, and heat map.
