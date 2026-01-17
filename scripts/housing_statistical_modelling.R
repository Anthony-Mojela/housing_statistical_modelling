############################################################ 
#Linear Regression                    
############################################################

# ----------------------------------------------------------------------------- #
# 0. Load Libraries -----------------------------------------------------------
# ----------------------------------------------------------------------------- # 

library(h2o)
library(MLmetrics)
library(recipes)
library(dplyr)
library(caTools)
library(caret)        
library(ggplot2)
library(lmtest)       # For autocorrelation test
library(gridExtra)    # For arranging plots
library(tidyr)
library(rpart) # to fit a reg tree (not in h2o)
library(rpart.plot)
# ----------------------------------------------------------------------------- #
# 1. Setup & User Parameters -------------------------------------------------- #


seed <- 123            # Random seed for reproducibility 
train_frac <- 0.8      # Proportion of data used for training 
metric <- "rmse"       # Evaluation metric: "rmse", "mae", "r2", "mse", etc.
folds <- 5             # Number of folds for cross-validation CV=5.

# 2. Load & Inspect Data ------------------------------------------------------
# ----------------------------------------------------------------------------- #
df <- read.csv("Housing.csv")
str(df)
# Convert all character variables to factors
df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], as.factor)

# Check structure again to confirm conversion
str(df)

# ----------------------------------------------------------------------------- #
# 3. Explore Target Distribution ---------------------------------------------- 
# ----------------------------------------------------------------------------- #

# For this demonstration, we are interested in predicting the housing market value(price). Let's first assess normality (a requirement for OLS):

hist <- ggplot(df, aes(x = price)) +
  geom_histogram(bins = 10, fill = "skyblue", color = "black") +
  theme_minimal() +
  ggtitle("Histogram of Target Variable (price)")

hist # some deviation from normality so let's transform it:

# apply transformation to make more normal
df$tranformed_target <- log(df$price)  # natural logarithm or try sqrt()

hist_transformed <- ggplot(df, aes(x = tranformed_target)) +
  geom_histogram(bins = 10, fill = "#C1FFC1", color = "black") +
  theme_minimal() +
  ggtitle("Histogram of Target Variable (log price)")

hist_transformed # showing less deviation from normality

# specify target and features

# Target: streams, Features: all others - remove if not so
# remove original target:

df_final <- df %>% select(-price) # change this to transformed_target if not being used

target <- "tranformed_target" 

features <- setdiff(names(df_final), target)

# ----------------------------------------------------------------------------- #
# 4. Train/Test Split --------------------------------------------------------- 
# ----------------------------------------------------------------------------- #

set.seed(seed)  
split=sample.split(df_final[[target]],SplitRatio = train_frac) 

training_set=subset(df_final,split==TRUE) 
test_set=subset(df_final,split==FALSE)


# ----------------------------------------------------------------------------- #
# 5. Data preprocessing (outside H2O) -----------------------------------------
# ----------------------------------------------------------------------------- #

# Note: for this example, there are no categorical predictors, but we will include it in the process anyway.

# as.formula(paste(target, "~ .")) is a generic or dynamic way to create a formula in R, where the target variable name is stored in a variable (called 'target" specified above) instead of being hard-coded.

# 6. define the model so that the function knows what is the target (this defines the recipe)
rec <- recipe(as.formula(paste(target, "~ .")), data = training_set) %>%
  step_normalize(all_numeric_predictors()) %>%  # this must be done first
  step_dummy(all_nominal_predictors(), one_hot = FALSE) # the last category is dropped

# to avoid perfect linearity, one of the categories of the variable during the encoding is dropped. This speeds up the training and improves the stability of the ML model. This is done by setting one_hot = FALSE. 

# 7.. Prep the recipe using the training data
rec_prep <- prep(rec, training = training_set)

# . Apply (bake) the prepped recipe on the scaled training set 
train_processed <- bake(rec_prep, new_data = NULL)

# 8. Apply (bake) the same transformations to the scaled test set
test_processed <- bake(rec_prep, new_data = test_set)

summary(train_processed)
summary(test_processed)

# we do not need to scale the target for these methods

# ----------------------------------------------------------------------------- #
# 9. OLS Regression (with inference) ------------------------------------------
# ----------------------------------------------------------------------------- #

ols_model <- lm(as.formula(paste(target, "~ .")), data = train_processed)
summary(ols_model)  # Shows coefficients, p-values

# Residual diagnostics
ols_residuals <- residuals(ols_model)
ols_fitted <- fitted(ols_model)

# Autocorrelation check (Durbin-Watson)
dwtest(ols_model)

# recall: The Durbin-Watson statistic ranges roughly from 0 to 4
# Around 2 means no autocorrelation (residuals are independent)
# Values < 2 suggest positive autocorrelation.
# Values > 2 suggest negative autocorrelation
# The null hypothesis is that there is no positive/negative autocorrelation in the residuals.
# The alternative hypothesis is that there is positive/negative autocorrelation.

# 10. Residual plots
resid_plot <- ggplot(data.frame(fitted = ols_fitted, resid = ols_residuals),
                     aes(x = fitted, y = resid)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  theme_minimal() +
  ggtitle("Residuals vs Fitted")

qq_resid_plot <- ggplot(data.frame(resid = ols_residuals),
                        aes(sample = resid)) +
  stat_qq() +
  stat_qq_line(col = "red") +
  theme_minimal() +
  ggtitle("QQ Plot of Residuals")

grid.arrange(resid_plot, qq_resid_plot, ncol = 2)


# Performance of OLS model
preds_ols_train <- predict(ols_model, newdata = train_processed)
preds_ols_test  <- predict(ols_model, newdata = test_processed)


## Performance on training set:

# NOTE: train_processed[[target]] is also a generic code instead of using hard-coded targets

# the following functions require the predicted values followed by the actual values
MAE(preds_ols_train,train_processed[[target]])   # MAE - Mean Absolute Error 
RMSE(preds_ols_train,train_processed[[target]])  # RMSE - Root Mean Square Error
R2_Score(preds_ols_train,train_processed[[target]]) # R2_Score (unadjusted though)

## Performance on test set:

MAE(preds_ols_test,test_processed[[target]])
RMSE(preds_ols_test,test_processed[[target]])
R2_Score(preds_ols_test,test_processed[[target]])


# ----------------------------------------------------------------------------- #

h2o.init()

# Convert scaled data to H2O dataframe
# Convert scaled data to H2O dataframe
train_h2o <- as.h2o(train_processed)
test_h2o  <- as.h2o(test_processed)

target <- "tranformed_target"
features <- setdiff(colnames(train_processed), target)


hyper_params <- list(
  lambda = 10^seq(-3, 5, length = 20)  # lambda values (regularization strength)
) 

ridge_grid <- h2o.grid(
  algorithm = "glm",
  grid_id = "ridge_grid",
  x = features,
  y = target, 
  training_frame = train_h2o,
  family = "gaussian",
  alpha = 0,  # ridge
  nfolds = folds,
  keep_cross_validation_predictions = TRUE,
  standardize = FALSE, # this has been done outside of H2O
  seed = seed,
  hyper_params = hyper_params,
  search_criteria = list(strategy = "Cartesian")
)

# Step 2: Get the best lambda based on chosen metric (e.g., RMSE)
grid_perf_ridge <- h2o.getGrid(grid_id = "ridge_grid", 
                               sort_by = metric, 
                               decreasing = TRUE)
print(grid_perf_ridge)

# Step 3: Extract the tuned hyperparameter(s)
best_model_id_ridge <- grid_perf_ridge@model_ids[[1]]
best_ridge_model <- h2o.getModel(best_model_id_ridge)
best_lambda_ridge <- best_ridge_model@parameters$lambda

# Step 4: Refit a single Ridge model with the best lambda 
ridge_final <- h2o.glm(
  x = features,
  y = target,
  training_frame = train_h2o,
  family = "gaussian",
  alpha = 0,
  lambda = best_lambda_ridge,
  standardize = FALSE,
  seed = seed
)

# Save predictions
preds_ridge_train <- h2o.predict(ridge_final, train_h2o)
preds_ridge_test <- h2o.predict(ridge_final, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_ridge_train <- as.vector(as.data.frame(preds_ridge_train)$predict)
preds_ridge_test <- as.vector(as.data.frame(preds_ridge_test)$predict)

# Look at performance:
MAE(preds_ridge_train,train_processed[[target]])  
RMSE(preds_ridge_train,train_processed[[target]])  
R2_Score(preds_ridge_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_ridge_test,test_processed[[target]])
RMSE(preds_ridge_test,test_processed[[target]])
R2_Score(preds_ridge_test,test_processed[[target]])


## Lasso: alpha = 1 -----------------------------------------------------------


# Step 1: Grid search for lambda with alpha=1 (Lasso)
lasso_grid <- h2o.grid(
  algorithm = "glm",
  grid_id = "lasso_grid",
  x = features,
  y = target, 
  training_frame = train_h2o,
  family = "gaussian",
  alpha = 1,  # lasso
  nfolds = folds,
  keep_cross_validation_predictions = TRUE,
  standardize = FALSE,
  seed = seed,
  hyper_params = hyper_params,
  search_criteria = list(strategy = "Cartesian")
)

# Step 2: Get the best lambda based on chosen metric (e.g., RMSE)
grid_perf_lasso <- h2o.getGrid(grid_id = "lasso_grid", 
                               sort_by = metric, 
                               decreasing = FALSE)
print(grid_perf_lasso)

# Step 3: Extract the tuned hyperparameter(s)
best_model_id_lasso <- grid_perf_lasso@model_ids[[1]]
best_lasso_model <- h2o.getModel(best_model_id_lasso)
best_lambda_lasso <- best_lasso_model@parameters$lambda

# Step 4: Refit a single Lasso model with the best lambda 
lasso_final <- h2o.glm(
  x = features,
  y = target,
  training_frame = train_h2o,
  family = "gaussian",
  alpha = 1,
  lambda = best_lambda_lasso,
  standardize = FALSE,
  seed = seed
)

# Save predictions
preds_lasso_train <- h2o.predict(lasso_final, train_h2o)
preds_lasso_test <- h2o.predict(lasso_final, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_lasso_train <- as.vector(as.data.frame(preds_lasso_train)$predict)
preds_lasso_test <- as.vector(as.data.frame(preds_lasso_test)$predict)

# Look at performance:
MAE(preds_lasso_train,train_processed[[target]])  
RMSE(preds_lasso_train,train_processed[[target]])  
R2_Score(preds_lasso_train,train_processed[[target]])

## Performance on test set:

MAE(preds_lasso_test,test_processed[[target]])
RMSE(preds_lasso_test,test_processed[[target]])
R2_Score(preds_lasso_test,test_processed[[target]])


## Elastic net (combination of ridge and lasso) --------------------------------

# define a new grid search to include alpha

hyper_params_elastic <- list(
  lambda = 10^seq(-3, 5, length = 20),
  alpha = seq(0, 1, by = 0.1)
) 

# Step 1: Grid search for lambda and alpha
elastic_grid <- h2o.grid(
  algorithm = "glm",
  grid_id = "elastic_grid",
  x = features,
  y = target, 
  training_frame = train_h2o,
  family = "gaussian",
  nfolds = folds,
  keep_cross_validation_predictions = TRUE,
  standardize = FALSE,
  seed = seed,
  hyper_params = hyper_params_elastic,
  search_criteria = list(strategy = "Cartesian")
)

# Step 2: Get the best lambda and alpha based on chosen metric (e.g., RMSE)
grid_perf_elastic <- h2o.getGrid(grid_id = "elastic_grid", 
                                 sort_by = metric, 
                                 decreasing = FALSE)
print(grid_perf_elastic)

# Step 3: Extract the tuned hyperparameter(s)
best_model_id_elastic <- grid_perf_elastic@model_ids[[1]]
best_elastic_model <- h2o.getModel(best_model_id_elastic)
best_lambda_elastic<- best_elastic_model@parameters$lambda
best_alpha_elastic <- best_elastic_model@parameters$alpha


# Step 4: Refit a single elastic model with the best lambda and alpha
elastic_final <- h2o.glm(
  x = features,
  y = target,
  training_frame = train_h2o,
  family = "gaussian",
  lambda = best_lambda_elastic,
  alpha = best_alpha_elastic,
  standardize = FALSE,
  seed = seed
)

# Save predictions
preds_elastic_train <- h2o.predict(elastic_final, train_h2o)
preds_elastic_test <- h2o.predict(elastic_final, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_elastic_train <- as.vector(as.data.frame(preds_elastic_train)$predict)
preds_elastic_test <- as.vector(as.data.frame(preds_elastic_test)$predict)

# Look at performance:
MAE(preds_elastic_train,train_processed[[target]])  
RMSE(preds_elastic_train,train_processed[[target]])  
R2_Score(preds_elastic_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_elastic_test,test_processed[[target]])
RMSE(preds_elastic_test,test_processed[[target]])
R2_Score(preds_elastic_test,test_processed[[target]])


# 10. Create a data frame of the results --------------------------------------
# ----------------------------------------------------------------------------- #

models <- c("ols", "elastic", "ridge", "lasso")
metrics <- c("MAE", "RMSE", "R2")

results <- data.frame()

for (model in models) {
  # Construct prediction object names dynamically
  preds_train <- get(paste0("preds_", model, "_train"))
  preds_test  <- get(paste0("preds_", model, "_test"))
  
  # Calculate metrics for train
  mae_train  <- round(MAE(preds_train, train_processed[[target]]), 4)
  rmse_train <- round(RMSE(preds_train, train_processed[[target]]), 4)
  r2_train   <- round(R2_Score(preds_train, train_processed[[target]]), 4)
  
  # Calculate metrics for test
  mae_test  <- round(MAE(preds_test, test_processed[[target]]), 4)
  rmse_test <- round(RMSE(preds_test, test_processed[[target]]), 4)
  r2_test   <- round(R2_Score(preds_test, test_processed[[target]]), 4)
  
  # Bind results into a dataframe
  temp <- data.frame(
    Model = model,
    Dataset = c("Train", "Test"),
    MAE = c(mae_train, mae_test),
    RMSE = c(rmse_train, rmse_test),
    R2 = c(r2_train, r2_test)
  )
  
  results <- rbind(results, temp)
}

print(results)

results_long <- results %>%
  pivot_longer(cols = c(MAE, RMSE, R2), names_to = "Metric", values_to = "Value")

results_long$Dataset <- factor(results_long$Dataset, levels = c("Train", "Test"))

# Plot
ggplot(results_long, aes(x = Model, y = Value, fill = Dataset)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(title = "Model Performance Metrics by Dataset",
       y = "Metric Value",
       x = "Model") +
  theme_minimal() +
  scale_fill_manual(values = c("Train" = "#66CDAA", "Test" = "#FF6A6A"))



#6. Regression tree in H2O --------------------------------------------
  # ----------------------------------------------------------------------------- #
  
  # Hyperparameter grid for pruning/tuning
  # (max_depth = tree depth, min_rows = minimum observations per leaf)
  hyper_params_tree <- list(
    max_depth = seq(1, 10, 1),
    min_rows  = c(1, 5, 10)
  )

# Define search criteria:
search_criteria <- list(
  strategy = "Cartesian" 
)


grid_tree <- h2o.grid(
  algorithm = "gbm",
  grid_id = "reg_tree_grid", # this is just the ID we are giving the search grid
  x = features,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params_tree,
  search_criteria = search_criteria,
  ntrees = 1, # ntrees=1 for single tree
  sample_rate = 1,  # no row subsampling
  col_sample_rate = 1,  # no column subsampling
  seed = seed
)

# Get the best model from the grid (based on specified metric) - note: if we want to minimize the metric then we use decreasing = FALSE, if we want to maximize the metric, then we set decreasing = TRUE

sorted_grid_tree <- h2o.getGrid("reg_tree_grid", 
                                sort_by = metric, 
                                decreasing = FALSE)

print(sorted_grid_tree)

best_model_id_tree <- sorted_grid_tree@model_ids[[1]]

best_model_tree <- h2o.getModel(best_model_id_tree)

#Extract the tuned hyperparameter(s)
tuned_max_depth <- best_model_tree@allparameters$max_depth
tuned_min_rows <- best_model_tree@allparameters$min_rows

# refit the regression tree on training set using tuned hyperparameters:

reg_tree_h2o <- h2o.gbm(
  x = features,
  y = target,
  training_frame = train_h2o,
  ntrees = 1,                
  max_depth = tuned_max_depth, 
  min_rows = tuned_min_rows,    
  sample_rate = 1,           # use all rows
  col_sample_rate = 1,       # use all predictors
  seed = seed
)

# Save predictions
preds_tree_h2o_train <- h2o.predict(reg_tree_h2o, train_h2o)
preds_tree_h2o_test <- h2o.predict(reg_tree_h2o, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_tree_h2o_train <- as.vector(as.data.frame(preds_tree_h2o_train)$predict)
preds_tree_h2o_test <- as.vector(as.data.frame(preds_tree_h2o_test)$predict)

# Look at performance:
MAE(preds_tree_h2o_train,train_processed[[target]])  
RMSE(preds_tree_h2o_train,train_processed[[target]])  
R2_Score(preds_tree_h2o_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_tree_h2o_test,test_processed[[target]])
RMSE(preds_tree_h2o_test,test_processed[[target]])
R2_Score(preds_tree_h2o_test,test_processed[[target]])


# ----------------------------------------------------------------------------- #
# 7. Regression tree in rpart --------------------------------------------
# ----------------------------------------------------------------------------- #

set.seed(seed) #always need to set the seed again before fitting any model for reproducibility of results, as rpart automatically runs cross-validation in the background. 

# Note: rpart itself doesn’t have built-in grid search

tree_rpart <- rpart(as.formula(paste(target, "~ .")),  
                    data = training_set,  # fit to the unprocessed training set
                    method  = "anova", # for regression tree
                    xval = folds, # xval argument = number of folds for CV
                    control = rpart.control(
                      #cp = 0.01,             # complexity parameter for pruning
                      minsplit = 20,         # minimum observations to attempt a split
                      maxdepth = 5           # maximum depth
                    )
)


# In H2O, trees are not pruned post-hoc. Instead, H2O controls tree growth while building the tree using: max_depth and min_rows. rpart however grows a large tree, then prunes back using cp and CV.

#By default, rpart function runs 10-fold cross validation (or k-folds where xval = k has been specified in the rpart function. Default stopping criterion is that a node needs to contain a minimum of 20 observations in order for further splitting to be permitted.

tree_rpart # run this to see information about the fitted tree
summary(tree_rpart) #you can run this too, but it provides TOO MUCH information!

# We can obtain the table and plot the error of the tree according to the different values of cp (cost complexity parameter):

# Value of cp is inversely related to the complexity of the tree.
# cp is tuned by examining how SSE on the training set diminishes with increasing complexity.  
# Eventually diminishing returns will set in i.e. it doesn't make sense to grow the tree any further

plotcp(tree_rpart)
tree_rpart$cptable

# rel error (Relative Error): This is the training error at a given level of pruning, relative to the root node
# xerror (Cross-validated Error): This is the estimated prediction error obtained via k-fold cross-validation
# xstd (Standard Deviation of xerror): This is the standard deviation of the cross-validated error across folds.

# print the tuned cp:
tree_rpart$cptable[which.min(tree_rpart$cptable[,"xerror"]), "CP"]

# ----------------------------------------------------------------------------- #
# 8. Visualizing the rpart tree --------------------------------------------
# ----------------------------------------------------------------------------- # 

# We can visualize our model with rpart.plot

rpart.plot(tree_rpart)

rpart.plot(tree_rpart, yesno=1,type=2,fallen.leaves = FALSE) # add additional options to change the appearance.

# see http://www.milbo.org/rpart-plot/prp.pdf for more options to customize the plot

# ----------------------------------------------------------------------------- #
# 9. Performance metrics for rpart tree --------------------------------------------
# ----------------------------------------------------------------------------- #

preds_tree_rpart_train = predict(tree_rpart,newdata=training_set) 
preds_tree_rpart_test = predict(tree_rpart,newdata=test_set) 

# Look at performance:
MAE(preds_tree_rpart_train,training_set[[target]])  
RMSE(preds_tree_rpart_train,training_set[[target]])  
R2_Score(preds_tree_rpart_train,training_set[[target]]) 

## Performance on test set:

MAE(preds_tree_rpart_test,test_set[[target]])
RMSE(preds_tree_rpart_test,test_set[[target]])
R2_Score(preds_tree_rpart_test,test_set[[target]])

# variable importance

varImp(tree_rpart) # we use the VarImp function to extract the overall variable importance.  



# 5. SVR -----------------------------------------
# ----------------------------------------------------------------------------- #

# we first set up our controls for cross validation:

metric <- "RMSE" 

control <-  trainControl(method = "cv", 
                         number = folds,  
                         savePredictions = TRUE)

#### SVR Linear ####

# Recall: For a linear SVR (just as in the case for the SVM), the only hyperparameter is C (the regularization parameter that controls the trade-off between maximizing the margin and minimizing the loss by introducing slack variables. 

# Recall from STAT606, we can specify a grid search via 3 different vectors:

grid1 <- expand.grid(C = c(0.75, 0.9, 1)) # specify a vector of possible values

grid2 <- expand.grid(C = seq(0, 2, length = 20)) # 20 values from 0 to 20

grid3 <- expand.grid(C = seq(0, 2, by = 0.1)) # values from 0 to 2 in increments of 0.1

set.seed(seed)

SVR_linear <- train(as.formula(paste(target, "~ .")),    
                    data = train_processed, 
                    method = "svmLinear", # for linear SVM
                    metric= metric,
                    trControl = control,
                    tuneGrid = grid1   # change between grid1, grid2 and grid3 depending on preference
)

# Print the best tuning parameter C that maximizes the model's performance

SVR_linear$bestTune

#### Performance for SVR linear ####

preds_SVR_linear_train = predict(SVR_linear,newdata=train_processed) 
preds_SVR_linear_test = predict(SVR_linear,newdata=test_processed) 

# Look at performance:
MAE(preds_SVR_linear_train,train_processed[[target]])  
RMSE(preds_SVR_linear_train,train_processed[[target]])  
R2_Score(preds_SVR_linear_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_SVR_linear_test,test_processed[[target]])
RMSE(preds_SVR_linear_test,test_processed[[target]])
R2_Score(preds_SVR_linear_test,test_processed[[target]])

#### SVR Radial ####

# Recall: For the Radial basis function (RBF) SVR, there is an additional hyperparameter: Gamma, (called sigma here in the train function). 

# Use the expand.grid to specify the search space	
grid_radial_1 <- expand.grid(sigma = c(.01, .015, 0.2),
                             C = c(0.75, 0.9, 1, 1.1, 1.25))

grid_radial_2 <- expand.grid(sigma = seq(1, 3, length = 10),
                             C = 10^6)


set.seed(seed) 

SVR_radial <- train(as.formula(paste(target, "~ .")), 
                    data = train_processed,
                    method = "svmRadial", # Radial kernel
                    metric=metric,
                    trControl=control,
                    tuneGrid = grid_radial_1) # change as required

# Extract tuned hyperparameters:

SVR_radial$bestTune

#### Performance for SVR radial ####

preds_SVR_radial_train = predict(SVR_radial,newdata=train_processed) 
preds_SVR_radial_test = predict(SVR_radial,newdata=test_processed) 

# Look at performance:
MAE(preds_SVR_radial_train,train_processed[[target]])  
RMSE(preds_SVR_radial_train,train_processed[[target]])  
R2_Score(preds_SVR_radial_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_SVR_radial_test,test_processed[[target]])
RMSE(preds_SVR_radial_test,test_processed[[target]])
R2_Score(preds_SVR_radial_test,test_processed[[target]])

#### SVR Poly ####

# The polynomial SVR has degree (d) and scale hyperparameters, in addition to C

grid_poly <- expand.grid(
  degree = c(1, 2, 3),   # polynomial degree
  scale  = c(0.01, 0.05, 0.1),  # kernel scale (gamma)
  C      = c(1, 5, 10)          # regularization
)

set.seed(seed)
SVR_poly <- train(as.formula(paste(target, "~ .")), 
                  data = train_processed, 
                  method = "svmPoly",
                  metric= metric,
                  trControl = control,
                  tuneGrid = grid_poly
)

# Extract tuned hyperparameters:

SVR_poly$bestTune

#### Performance for SVR poly ####

preds_SVR_poly_train = predict(SVR_poly,newdata=train_processed) 
preds_SVR_poly_test = predict(SVR_poly,newdata=test_processed) 

# Look at performance:
MAE(preds_SVR_poly_train,train_processed[[target]])  
RMSE(preds_SVR_poly_train,train_processed[[target]])  
R2_Score(preds_SVR_poly_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_SVR_poly_test,test_processed[[target]])
RMSE(preds_SVR_poly_test,test_processed[[target]])
R2_Score(preds_SVR_poly_test,test_processed[[target]])

# ----------------------------------------------------------------------------- #
# 6. KNN -----------------------------------------
# ----------------------------------------------------------------------------- #

# For the KNN, we fit the same train function, but change the grid search as the method.

grid_knn <- expand.grid(k = seq(3, 15, 2))  # try odd values between 3 and 15


set.seed(seed)

KNN <- train(as.formula(paste(target, "~ .")),    
             data = train_processed, 
             method = "knn", 
             metric= metric,
             trControl = control, # use the same control function from SVR
             tuneGrid = grid_knn   
)


# Extract tuned hyperparameters:

KNN$bestTune

#### Performance for KNN ####

preds_KNN_train = predict(KNN,newdata=train_processed) 
preds_KNN_test = predict(KNN,newdata=test_processed) 

# Look at performance:
MAE(preds_KNN_train,train_processed[[target]])  
RMSE(preds_KNN_train,train_processed[[target]])  
R2_Score(preds_KNN_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_KNN_test,test_processed[[target]])
RMSE(preds_KNN_test,test_processed[[target]])
R2_Score(preds_KNN_test,test_processed[[target]])

----------------------------------------------------------------------------- #
  # 7. NN in H2o -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

#### Initialize h2o ####

h2o.init()

# Convert scaled data to H2O dataframe
train_h2o <- as.h2o(train_processed)
test_h2o  <- as.h2o(test_processed)

#### Train the NN #####

# We fit the NN for regression in the exact same way as what we did for classification in STAT606. For this demonstration, we will also include regularization parameters added to the neural network (to aid in reducing overfitting). This add a penalty to the original loss function for the NN.

# Set hyperparameters for grid search
hyper_params_NN <- list(
  hidden = list(c(5, 5), c(10, 10), c(5, 10, 5)),  # 2 hidden layers with 5 nodes each, then 2 with 10 nodes each, then 3 with 5, 10 and 5 nodes.
  activation = c("Rectifier", "Tanh", "Maxout"),  # activation functions (maxout is the max of the input for the node, which is the weighted outputs from the previous node plus the bias)
  epochs = c(10, 20),  # number of epochs 
  rate = c(0.001, 0.01),  # learning rate 
  l1 = c(0, 1e-5), # lasso
  l2 = c(0, 1e-5) # ridge
)


# Perform grid search for hyperparameter tuning
grid_search_NN <- h2o.grid(
  algorithm = "deeplearning", 
  grid_id = "nn_grid",
  hyper_params = hyper_params_NN,
  x = features,
  y = target,
  standardize = FALSE, # this has already been done
  training_frame = train_h2o,
  search_criteria = list(strategy = "Cartesian"),
  adaptive_rate = FALSE, # turn on and off
  nfolds = folds,
  stopping_rounds = 0, # turn off early stopping
  seed = seed,
  reproducible = TRUE # turns off multi-threading for reproducibility, does make it slower
)


# View grid sorted by metric
grid_results_NN <- h2o.getGrid(grid_id = "nn_grid", 
                               sort_by = metric, 
                               decreasing = FALSE)
print(grid_results_NN)

# extract best model
best_model_NN <- h2o.getModel(grid_results_NN@model_ids[[1]])

best_params_NN <- best_model_NN@allparameters
print(best_params_NN)

NN <- h2o.deeplearning(
  x = features,
  y = target,
  training_frame = train_h2o,  # Combine training + validation if needed
  hidden = best_params_NN$hidden,
  activation = best_params_NN$activation,
  rate = best_params_NN$rate,
  adaptive_rate = FALSE,
  l1 = best_params_NN$l1,
  l2 = best_params_NN$l2,
  epochs = best_params_NN$epochs, # defaults to 10 if not included in hyperparams
  seed = seed,
  reproducible = TRUE
) 

#### Performance for NN ####

# Save predictions
preds_NN_train <- h2o.predict(NN, train_h2o)
preds_NN_test <- h2o.predict(NN, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_NN_train <- as.vector(as.data.frame(preds_NN_train)$predict)
preds_NN_test <- as.vector(as.data.frame(preds_NN_test)$predict)

# Look at performance:
MAE(preds_NN_train,train_processed[[target]])  
RMSE(preds_NN_train,train_processed[[target]])  
R2_Score(preds_NN_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_NN_test,test_processed[[target]])
RMSE(preds_NN_test,test_processed[[target]])
R2_Score(preds_NN_test,test_processed[[target]])


#8. Combine and compare all results  --------------------------------------
  # ----------------------------------------------------------------------------- #
  
  models <- c("SVR_linear", "SVR_radial", "SVR_poly", "KNN", "NN")
metrics <- c("MAE", "RMSE", "R2")

results <- data.frame()

for (model in models) {
  # Construct prediction object names dynamically
  preds_train <- get(paste0("preds_", model, "_train"))
  preds_test  <- get(paste0("preds_", model, "_test"))
  
  # Calculate metrics for train
  mae_train  <- round(MAE(preds_train, train_processed[[target]]), 4)
  rmse_train <- round(RMSE(preds_train, train_processed[[target]]), 4)
  r2_train   <- round(R2_Score(preds_train, train_processed[[target]]), 4)
  
  # Calculate metrics for test
  mae_test  <- round(MAE(preds_test, test_processed[[target]]), 4)
  rmse_test <- round(RMSE(preds_test, test_processed[[target]]), 4)
  r2_test   <- round(R2_Score(preds_test, test_processed[[target]]), 4)
  
  # Bind results into a dataframe
  temp <- data.frame(
    Model = model,
    Dataset = c("Train", "Test"),
    MAE = c(mae_train, mae_test),
    RMSE = c(rmse_train, rmse_test),
    R2 = c(r2_train, r2_test)
  )
  
  results <- rbind(results, temp)
}

print(results)

# graph the results

# change results to long form first
results_long <- results %>%
  pivot_longer(cols = c(MAE, RMSE, R2), names_to = "Metric", values_to = "Value")

results_long$Dataset <- factor(results_long$Dataset, levels = c("Train", "Test"))

# Plot
ggplot(results_long, aes(x = Model, y = Value, fill = Dataset)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(title = "Model Performance Metrics by Dataset",
       y = "Metric Value",
       x = "Model") +
  theme_minimal() +
  scale_fill_manual(values = c("Train" = "#66CDAA", "Test" = "#FF6A6A"))

# ----------------------------------------------------------------------------- #
# 9. Shutdown H2O ------------------------------------------------------------
# ----------------------------------------------------------------------------- #

# If a mistake was made along the way or H2O model needed to be run again, shut H2o down and start again.

h2o.shutdown(prompt = FALSE)

