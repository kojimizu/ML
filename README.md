
<font size="10"></font>

# Machine Learning Archive  
This repository archives useful R-related books and repositories for machine learning. 
The recent development comes from `tidyverse` and `tidymodels`, and following cites are useful for catch-up.  

<Tidyverse training materials>

- 🎁 Tidy data science workshop :https://tidy-ds.wjakethompson.com/       
- 🎁 Remaster `tidyverse`: https://github.com/rstudio-education/remaster-the-tidyverse     
- 🎁  An Antarctic Tour of the Tidyverse (R-Ladies Chicago): https://silvia.rbind.io/talk/2020-08-31-tour-of-the-tidyverse/  

![image](https://user-images.githubusercontent.com/29205710/111499004-d7737400-8785-11eb-9519-d9d6afb407e2.png)


For latest papers, browse the state-of-the art is useful for finding the paper and code on GitHub.  
https://paperswithcode.com/sota  

## ML Overview
| Material | Link/Repository  | 
|--------- | ---------------- | 
| Applied ML | https://github.com/kojimizu/rstudio-conf-2018 |
| Hands-on ML with R |  http://bit.ly/HOML_with_R   |
| UC Business Analytics with R | http://uc-r.github.io |
| Data Analysis and Prediction Algorithms with R |  https://rafalab.github.io/dsbook/ |

### Applied Machine Learning
Applied ML workshop by Max Kuhn from useR/rstudio-conf Conference

- rstudio-conf-2019 repository: https://github.com/topepo/rstudio-conf-2019
- rstudio-conf-2018 repository: https://github.com/kojimizu/rstudio-conf-2018

### 📚 Hands on Machine Learning with R
Hands-on Machine Learning with R: An applied book covering the fundamentals of machine learning with R.  
This book covers multiple models 1) gm, 2) regularized glm, 3) random forest and 4) gradient boosting methods.  
From 2018 to 2019 January  
📚 Book: http://bit.ly/HOML_with_R  
📦 Repo: https://github.com/bradleyboehmke/hands-on-machine-learning-with-r   

### 📚 UC Business analyrics by R 
This material from the below page covers basic R techniques, descrptive analytics, and predictive analytics.  
📚 Book: University of Cincinatti - http://uc-r.github.io  

- Predictive analytics / Supervised learning: Linear, Naive Bayes, Reularized Regression (Ridge, LASSO), MARS, Regression Tree, Random Forest, GBM, Discriminant Analysis, SVM
- Descriptive analytics / Unsupervised learning (K-means, PCA, Hierarchical clustering), Text mining, classical methods 
- Time series analysis: Exponential smoothing, MA/AR
- DL: Regression DNN, Classification DNN

--- updated until here on 11/02/2021

### 📚 Introduction to data science   
Modeling algorithm by R. 
📚 Book: https://rafalab.github.io/dsbook/  
📦 Repo: https://github.com/rafalab/dsbook/tree/master/docs  

## ML Interpretability  
### Intepretable Machine Learning  
📚 Book: https://christophm.github.io/interpretable-ml-book/

### Limitations of Interpretable Machine Learning Methods  
📚 Book: https://compstat-lmu.github.io/iml_methods_limitations/  

## Modeling Resources
### 📦 Caret
__Caret package introduction by Max Kuhn (Bookdown)__
A package for ML modeling with pre-processing techniques 
📚 Book: http://topepo.github.io/caret/  
📦 Repo: https://github.com/topepo/caret  

```{R}
# Avaiable from CRAN
install.packages("caret")
```

### 📦 Tidymodels  
Tidymodels is a multi-package containing modern tidyverse-based packages. 
Tidymodel usecase is prepared by the dev team: https://www.tidymodels.org/learn/

📦Learn: https://www.tidymodels.org/learn/
📦 Repo: https://github.com/tidymodels/tidymodels

- **broom**:  tidy, augment, glance for model interpretation support
- **rsample**:  data resampling 
- **recipes**:  pre-processing functions (similar to caret)
- **parsnip**:  modeling functions 

Useful blog posts
recipes: http://www.rebeccabarter.com/blog/2019-06-06_pre_processing/  
tidymodels: https://rviews.rstudio.com/2019/06/19/a-gentle-intro-to-tidymodels/  
Tutorial by RStudio: https://github.com/tidymodels/aml-training

## Feature Engineering Resouces 
### 📚 Feature Engineering and Selection (FES)  
A Bookdown page prepared by Max Kuhn  
📚 Book: http://www.feat.engineering/  
📦 Repo: https://github.com/kojimizu/FES  

### 📚 Feature Engineering Book
by Alice Zheng  

=========================================
# Natural Language Processing (NLP)

## Supervised machine learning for text analysis 
The latest book on 2020/07 by Julia Silge and EMIL HVITFELDT  

📚 Book: https://smltar.com/language.html  
📦 Repo: https://github.com/EmilHvitfeldt/smltar  

=========================================
# Modern R 
## Modern commands
### 📚 R for Data Science
Basic commands with tidyverse is summarised.
📚 Book: https://r4ds.had.co.nz/  
The excercise solution can be found on Bookdown as well.

### 📚 Modern R with Tidyverse  
Bookdown material is available [here](https://b-rodrigues.github.io/modern_R/), covering:

### Datacamp courses
- 🎥 [Machine Learning in the Tidyverse](https://www.datacamp.com/courses/machine-learning-in-the-tidyverse).

## Purrr material 
### RStudio 
https://purrr.tidyverse.org/  

### Happy Purrrr 
- 🎥 [A tutorial video](https://resources.rstudio.com/wistia-rstudio-conf-2017/happy-r-users-purrr-tutorial-charlotte-wickham) by Charotte Wickhkm. THe DataCamp courses are useful for code practice.
- 🎥 [Functional progamming with purrr](https://www.datacamp.com/courses/foundations-of-functional-programming-with-purrr)  by DataCamp  
### Case study of Purrr
Lessons and Examples by Jenny Brian: 
📚 Web: https://jennybc.github.io/purrr-tutorial/  
- Background basics
- Core purrr lessons
- Worked examples 

=========================================
## Visualization
### 📚 Data visualition - A practical introduction  
by Kieran Healy.   
📚 Book: http://socviz.co/index.html.

### 📚 R Graphics cookbook
by Winston Chang
📚 Book: https://r-graphics.org/  
Another useful website is Cookbook for R: covering other topics online: http://www.cookbook-r.com/

### Geo-visualization
- Site: https://www.jasondavies.com/  

### ggplot tutorial  
https://cedricscherer.netlify.com/2019/08/05/a-ggplot2-tutorial-for-beautiful-plotting-in-r/#toc  

### Data viz packages

#### Rayshader: 
3D plot including `ggplot2` viz: https://www.tylermw.com/  

#### Uber's H3
Hexigon-shaped viz:  
site: http://estrellita.hatenablog.com/entry/2019/05/01/235406  

#### Mapdeck
Blogpost: http://estrellita.hatenablog.com/entry/2018/09/25/230000_3    

=========================================
## Stats by Uni Class / Workshop 

### Stat 
Lecture by Jenny Brian:  

### Stanford CS229
Machine learning course by Andrew Ng from Stanford University  
🎥Video and 📚Material : http://cs229.stanford.edu/syllabus.html  

### Statistical Rethinking  
Bayes stats using `brms`, `ggplot2` and `tidyverse`
- 📚 Bookdown: https://bookdown.org/ajkurz/Statistical_Rethinking_recoded/  
- HP: https://xcelab.net/rm/statistical-rethinking/    

### Deep learning with TF and Keras in R  
Workshop for Rstudio::2020 by Bradley Boemeike.  
Repo: https://github.com/kojimizu/dl-keras-tf  
