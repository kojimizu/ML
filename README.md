# Machine Learning Archive  
Bookdown is a R package, enabling users to publish books. 
Bookdown.org has an archive storing published books based on the bookdown package. 
The below is an simple introduction of useful books for statistical modeling and proeprocessing/feature engineering.

## ML Overview
### 1. Applied Machine Learning
Applied ML workshop by Max Kuhn from useR2018 Conference
Max Kuhn has talked about Applied Machine Learning in RStudio::Conf 2017. The material is stored in [this repository](https://github.com/kojimizu/rstudio-conf-2018). The latest repository is [here](https://github.com/topepo/rstudio-conf-2019).  

### 2. Hands on Machine Learning with R
Hands-on Machine Learning with R: An applied book covering the fundamentals of machine learning with R.  
This book covers multiple models 1) gm, 2) regularized glm, 3) random forest and 4) gradient boosting methods.  
From 2018 to 2019 January  
https://bradleyboehmke.github.io/hands-on-machine-learning-with-r/   
https://github.com/bradleyboehmke/hands-on-machine-learning-with-r   

### 3. UC Business analyrics by R
This material from the below page covers basic R techniques, descrptive analytics, and predictive analytics.  
Link: University of Cincinatti - http://uc-r.github.io  

- Predictive analyitics: 
    - ML(Linear, Naive Bayes, Regularized Regression, MARS, Regression Tree, Random Forests, GBM, Discriminant Analysis, SVM)

## Modeling Resources
### 1. Caret
__Caret package introduction by Max Kuhn (Bookdown)__
A package for ML modeling with pre-processing techniques 
```{R}
# Avaiable from CRAN
install.packages("caret")
```
From Sep 2018 - Oct 2018  
http://topepo.github.io/caret/  

### 2. Tidymodels  
Tidymodels is a multi-package containing modern tidyverse-based packages. I understand no text covers tidymodels as of now, and I have prepared a material covering each package's vignettes.  

```{R}
# Avaiable from CRAN
install.packages("tidymodels")
```
- broom:  tidy, augment, glance for model interpretation support
- rsample:  data resampling 
- recipes:  pre-processing functions (similar to caret)
- parsnip:  modeling functions 


## Feature Engineering Resouces 
### 1. Feature Engineering and Selection (FES)  
A Bookdown page prepared by Max Kuhn  
http://www.feat.engineering/  
Reference: https://github.com/kojimizu/FES  

### 2. Feature Engineering Book
by Alice Zheng  




