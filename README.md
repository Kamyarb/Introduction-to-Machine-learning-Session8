# Introduction-to-Machine-learning-Session8

هدف: آشنایی با درخت های تصمیم گیری و جنگل تصادفی Random Forest & Decision Tree

آشنایی با تئوری های آماری مربوط به Decision Tree
حل مثال عملی با Decision Tree و بهینه سازی Hyper parameter های مدل
آشنایی با روش های Ensemble Learning و معرفی جنگل تصادفی
آشنایی با سایر الگوریتم‌های خردجمعی مانند Xgboost و Adaboost
مقایسه جنگل های تصادفی و درخت تصمیم روی دیتای واقعی
ریلیز تمرین سری سه

# Decision Tree

* A supervised learning technique that can be used for discrete and continuous output.
* Visually engaging and easy to interpret.
* Excellent model for someone transitioning into the world of data science.
* Foundational to learning some very powerful techniques.
* Are prone towards high-variance.

<img src="https://i.postimg.cc/mg99cVx5/decision-tree2.png" width="700"/>


## Basic Interpretation

Decision Trees are made up of interconnected nodes, which act as a series of questions / test conditions (e.g., is the passenger male or female?)

* **Rule of thumb #1:** The higher up the tree a variable is, the most important it is
* **Rule of thumb #2:** The more frequently a variable appears, the more important it is

<img src="https://i.postimg.cc/GhF8dtvW/decision-tree.png" width="700"/>

### Training process of a Decision Tree

On each step or node of a decision tree, used for classification, we try to form a condition on the features to separate all the labels or classes contained in the dataset to the **fullest purity**.


### ⛔ Where to Stop Splitting

We worked on splitting the data based on features. If we have a huge number of features and we want to use all the features to build the tree, the tree will become too large. That may cause overfitting and long computation time.

To handle this:
1) there is a `max_depth` parameter in the decision tree function of the scikit-learn library. If the max depth parameter is large, the tree will be larger. So, we can choose the depth of the tree.

2) There is another parameter, `max_features`. The name of the parameter says what it does. You can specify the maximum number of features you want to use for your tree.


Here are what the columns represent:
* credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
* purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
* int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
* installment: The monthly installments owed by the borrower if the loan is funded.
* log.annual.inc: The natural log of the self-reported annual income of the borrower.
* dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
* fico: The FICO credit score of the borrower.
* days.with.cr.line: The number of days the borrower has had a credit line.
* revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
* revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
* inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
* delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
* pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).




## Ensemble learning

Ensemble simply means **combining multiple models**. Thus a collection of models is used to make predictions rather than an individual model.
The two main classes of ensemble learning methods are **bagging** and **boosting**.

*  **Bagging**– It creates a different training subset from sample training data with replacement & the final output is based on majority voting. For example,  Random Forest.

*  **Boosting**– It combines weak learners into strong learners by creating sequential models such that the final model has the highest accuracy. For example,  ADA BOOST, XG BOOST

<img src="https://i.postimg.cc/c4w9WR9d/ensemble.png" width="700"/>
<img src="https://i.postimg.cc/zBy7cKHZ/soft-vs-hard-voting.png" width="700"/>


## Ensemble learning

Ensemble simply means **combining multiple models**. Thus a collection of models is used to make predictions rather than an individual model.
The two main classes of ensemble learning methods are **bagging** and **boosting**.

*  **Bagging**– It creates a different training subset from sample training data with replacement & the final output is based on majority voting. For example,  Random Forest.

*  **Boosting**– It combines weak learners into strong learners by creating sequential models such that the final model has the highest accuracy. For example,  ADA BOOST, XG BOOST

<img src="https://i.postimg.cc/c4w9WR9d/ensemble.png" width="700"/>
<img src="https://i.postimg.cc/zBy7cKHZ/soft-vs-hard-voting.png" width="700"/>

### Why does this work?
* Different models may be good at different 'parts' of data (even if they underfit)
* Individual mistakes can be 'averaged out' (especially if models overfit)

### Which models should be combined?
* Bias-variance analysis teaches us that we have two options:
    * If model underfits (high bias, low variance): combine with other low-variance models
        * Need to be different: 'experts' on different parts of the data
        * Bias reduction. Can be done with **_Boosting_**
    * If model overfits (low bias, high variance): combine with other low-bias models
        * Need to be different: individual mistakes must be different
        * Variance reduction. Can be done with **_Bagging_**
* Models must be uncorrelated but good enough (otherwise the ensemble is worse)


## Bagging (Bootstrap Aggregating)

* Obtain different models by **training the _same_ model on _different training samples_**
    * Reduce overfitting by averaging out individual predictions (variance reduction)
* In practice: take $I$ bootstrap samples of your data, train a model on each bootstrap
   * Higher $I$: more models, more smoothing (but slower training and prediction)    
* Base models should be unstable: different training samples yield different models
    * E.g. very deep decision trees, or even randomized decision trees 
    * Deep Neural Networks can also benefit from bagging (deep ensembles)
* Prediction by averaging predictions of base models
    * Soft voting for classification (possibly weighted)
    * Mean value for regression
* Can produce uncertainty estimates as well
    * By combining class probabilities of individual models (or variances for regression) 
    
<img src="https://i.postimg.cc/SKCN7tVw/Bagging-Bootstrap-Aggregation.gif" />


### In practice

* Different implementations can be used. E.g. in scikit-learn:
    * `BaggingClassifier`: Choose your own base model and sampling procedure
    * `RandomForestClassifier`: Default implementation, many options

* Most important parameters:
    * `n_estimators` (>100, higher is better, but diminishing returns)
        * Will start to underfit (bias error component increases slightly)
    * `max_features`
        * Defaults: $sqrt(p)$ for classification, $log2(p)$ for regression
        * Set smaller to reduce space/time requirements
    * parameters of trees, e.g. `max_depth`, `min_samples_split`,...
        * Prepruning useful to reduce model size, but don't overdo it

* Easy to parallelize (set `n_jobs` to -1)
* Fix `random_state` (bootstrap samples) for reproducibility 

# 💥 Let’s compare


* **base_estimator** - This represents the algorithm used as the base/weak learners. We will use the DecisionTreeClassifier algorithm as our weak/base learners.

* **n_estimators** - This represents the number of weak learners used. We will use 100 decision trees to build the bagging model.

* **max_samples** - The maximum number of data that is sampled from the training set. We use 80% of the training dataset for resampling.

* **bootstrap** - Allows for resampling of the training dataset without replacement.

* **oob_score** - Used to compute the model’s accuracy score after training. The OOB_score is computed as the number of correctly predicted rows from the out-of-bag sample.

* **random_state** - Allows us to reproduce the same dataset samples.

## Random Forests
* Uses randomized trees to make models even less correlated (more unstable)
* At every split, only consider max_features features, randomly selected
* Extremely randomized trees: considers 1 random threshold for random set of features (faster)

<img src="https://i.postimg.cc/SNPS0pvB/Random-Forest.gif" width="800" />

>**❗ NOTE:** Increasing the number of models (trees) decreases variance (less overfitting)


## 6. Training the Random Forest model

* n_estimators = number of trees in the foreset
* max_features = max number of features considered for splitting a node
* max_depth = max number of levels in each decision tree
* min_samples_split = min number of data points placed in a node before the node is split
* min_samples_leaf = min number of data points allowed in a leaf node
* bootstrap = method for sampling data points (with or without replacement)


## Boosting

The idea is to fit models iteratively such that the training of model at a given step depends on the models fitted at the previous steps.

Boosting is an ensemble learning method that combines a set of weak learners into a strong learner to minimize training errors. In boosting, a random sample of data is selected, fitted with a model and then trained sequentially—that is, **each model tries to compensate for the weaknesses of its predecessor.** With each iteration, the weak rules from each individual classifier are combined to form one, strong prediction rule.  For example, ADA BOOST, XG BOOST



<img src="https://i.postimg.cc/5yHrLDSz/Boosting.png" width="600"/>


## ADA BOOST

<img src="https://i.postimg.cc/KjLdCd32/ada.jpg" width="600"/>

Random Forest and AdaBoost are two popular machine learning algorithms. 

Both algorithms can be used for <span style="color:red">classification</span> and <span style="color:red">regression</span> tasks.

Both Random Forest and AdaBoost algorithm is based on the creation of a Forest of trees. 

* **Random Forest** is an ensemble learning algorithm that is created using a bunch of decision trees that make use of different variables or features and makes use of bagging techniques for data samples.

* **AdaBoost** is also an ensemble learning algorithm that is created using a bunch of what is called a decision stump. Decision stumps are nothing but decision trees with one node and two leaves.

<img src="https://i.postimg.cc/9QXnB4bV/trees.png" width="600"/>


## ADA BOOST

<img src="https://i.postimg.cc/KjLdCd32/ada.jpg" width="600"/>

Random Forest and AdaBoost are two popular machine learning algorithms. 

Both algorithms can be used for <span style="color:red">classification</span> and <span style="color:red">regression</span> tasks.

Both Random Forest and AdaBoost algorithm is based on the creation of a Forest of trees. 

* **Random Forest** is an ensemble learning algorithm that is created using a bunch of decision trees that make use of different variables or features and makes use of bagging techniques for data samples.

* **AdaBoost** is also an ensemble learning algorithm that is created using a bunch of what is called a decision stump. Decision stumps are nothing but decision trees with one node and two leaves.

<img src="https://i.postimg.cc/9QXnB4bV/trees.png" width="600"/>

The AdaBoost algorithm can be said to make decisions using a bunch of decision stumps.

**The tree is then tweaked iteratively to focus on areas where it predicts incorrectly.**

As a result, AdaBoost typically provides more accurate predictions than Random Forest. However, AdaBoost is also more sensitive to over-fitting than Random Forest. Here are different posts on Random forest and AdaBoost.

<img src="https://i.postimg.cc/Ghd7cBZY/vote.png" width="600"/>

## XGBoost
 `XGBoost`  is a scalable and highly accurate implementation of gradient boosting that pushes the limits of computing power for boosted tree algorithms, being built largely for energizing machine learning model performance and computational speed.
`Supervised machine learning` uses algorithms to train a model to find patterns in a dataset with labels and features and then uses the trained model to predict the labels on a new dataset’s features.

<img src="https://i.postimg.cc/d348Q0ym/evolution.png" width="600"/>

![image](https://user-images.githubusercontent.com/100142624/181243424-2c805d8e-01ae-4c30-83c1-12e060d1b65b.png)

![image](https://user-images.githubusercontent.com/100142624/181243608-26ddee9e-863b-4eca-b232-b62749ba6db8.png)


<img src="https://i.postimg.cc/nrcDWtPs/comparison.jpg" width="600"/>

The list of benefits and attributes of XGBoost is extensive, and includes the following:

A large and growing list of data scientists globally that are actively contributing to XGBoost open source development
Usage on a wide range of applications, including solving problems in regression, classification, ranking, and user-defined prediction challenges
A library that’s highly portable and currently runs on OS X, Windows, and Linux platforms
Cloud integration that supports AWS, Azure, Yarn clusters, and other ecosystems
Active production use in multiple organizations across various vertical market areas
A library that was built from the ground up to be efficient, flexible, and portable
