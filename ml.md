---
title: "Practical Machine Learning - Course Project"
author: "GM"
date: "August 8, 2020"
output: 
  html_document:
    keep_md: true

---



# Executive summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.

# Data

Download:

```r
train_URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

train <- read.csv(url(train_URL),na.strings=c("","NA"))
test <- read.csv(url(test_URL),na.strings=c("","NA"))
```

# Exploratory Analysis

Prepare enviroment to be used:

```r
library(caret) # ML library
```

```
## Warning: package 'caret' was built under R version 4.0.2
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(parallel) # Paraller processing library to speed up training 
library(doParallel) # Paraller processing library to speed up training 
```

```
## Warning: package 'doParallel' was built under R version 4.0.2
```

```
## Loading required package: foreach
```

```
## Warning: package 'foreach' was built under R version 4.0.2
```

```
## Loading required package: iterators
```

```
## Warning: package 'iterators' was built under R version 4.0.2
```

```r
library(tictoc) # Timer to calculate how long it takes to compute code
```


```r
dim(train)
```

```
## [1] 19622   160
```
160 variables is quite a lot. Lets look at them.


```r
str(train)
str(test)
```

A lot of NA and not needed first two columns for model.

Remove index and user column:

```r
train<-train[, -c(1:2)]
test<-test[, -c(1:2)]
```

Check full cases (all values present in row):

```r
sum(complete.cases(train))/dim(train)[1]
```

```
## [1] 0.02069106
```

```r
sum(complete.cases(test))/dim(test)[1]
```

```
## [1] 0
```
Only about 2% full cases is too small to build model. 

Remove columns with NA and check full cases again:

```r
zero_sums<-colSums(is.na(train)) == 0
train <- train[, zero_sums] 
test <- test[, zero_sums] 
sum(complete.cases(train))/dim(train)[1]
```

```
## [1] 1
```

```r
dim(train)
```

```
## [1] 19622    58
```

```r
sum(complete.cases(test))/dim(test)[1]
```

```
## [1] 1
```

```r
dim(test)
```

```
## [1] 20 58
```
Looks like we have 19622 full cases in train and 20 in test.

We are left only with 57 variables.

Split at 80% to training and crossvalidation parts and set seed:

```r
set.seed(56842)
train_partitioned  <- createDataPartition(train$classe, p=0.8, list=FALSE)
training_set <- train[train_partitioned, ]
crossvalidation_set  <- train[-train_partitioned, ]
```

# Prediction Model

Building random forest prediction model using caret package and turning on parallel proccesing.

Make CPU cluter

```r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
```

Fit model:

```r
tic()
fitControl  <- trainControl(method = "cv",number = 5,allowParallel = TRUE)
random_forest_model <- train(classe ~ ., data=training_set, method="rf",trControl=fitControl )
stopCluster(cluster)
toc()
```

```
## 373.51 sec elapsed
```

```r
saveRDS(random_forest_model, file = "rf_model.rds")
```

Look at model:

```r
random_forest_model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 38
## 
##         OOB estimate of  error rate: 0.06%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4464    0    0    0    0 0.0000000000
## B    1 3036    1    0    0 0.0006583278
## C    0    3 2734    1    0 0.0014609204
## D    0    0    1 2571    1 0.0007773028
## E    0    0    0    1 2885 0.0003465003
```


```r
random_forest_model$resample
```

```
##    Accuracy     Kappa Resample
## 1 0.9993629 0.9991942    Fold1
## 2 0.9993629 0.9991941    Fold3
## 3 0.9990449 0.9987919    Fold2
## 4 0.9993633 0.9991947    Fold5
## 5 0.9987257 0.9983881    Fold4
```


```r
confusionMatrix.train(random_forest_model)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.0  0.0  0.0  0.0
##          B  0.0 19.3  0.0  0.0  0.0
##          C  0.0  0.0 17.4  0.0  0.0
##          D  0.0  0.0  0.0 16.4  0.0
##          E  0.0  0.0  0.0  0.0 18.4
##                             
##  Accuracy (average) : 0.9992
```
Model with Accuracy (average) : 0.9992 is realy nice. Lets check with test set:


```r
tic()
predict_random_forest_model <- predict(random_forest_model, newdata=test)
predict_random_forest_model
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
toc()
```

```
## 0.01 sec elapsed
```

# Results

Quiz gives 100%. Nice.

It takes about 0.001-0.0025 to make prediction using model for one case. 

Model with Accuracy (average) 0.9992 using Random Forest with 5 fold cross-validation.
