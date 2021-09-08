![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png)
# Capstone Project: Real or Fake Jobs

## Executive Summary
The datasets provided on kaggle include 18 variables (both structured and unstructured) that predicts whether a job listing is fraudulent or not. In this project, we cleaned the data by backfilling missing data and grouping similar categories of data together before one-hot-encoding categorical variables. We also pre-processed the text variables by removing links, non-alphabetic characters, stopwords and lemmatization. We removed variables which are highly correlated (>0.7) with each other and variables which we think are not significant in predicting whether a job listing is fraudulent based on our EDA findings (e.g. telecommuting and employment_type). We performed vectorization on the textual data and included it together with the structured data via a column transformer. As the classification problem is highly imbalanced, we performed Synthetic Minority Oversampling Technique (SMOTE) before fitting our models. We ran seven models - (i) Logistic Regression; (ii) Support Vector Classifier; (iii) Extreme Gradient Boost; (iv) Multinomial Naive Bayes; (v) K-Nearest Neighbors; (vi) Random Forest; and (vii) AdaBoost with both the count vectorizer and the TFIDF vectorizer at a 70%-30% train-test split. As this is an imbalanced classification problem, accuracy would not be a good metric for evaluation. In this case, we compared both the ROC AUC and recall scores and eventually selected the Logistic Regression with TFIDF Vectorizer as it had a high ROC AUC score of 0.946 and the highest recall score of 0.801 out of all the models. We used the BERTopic model for topic modelling and found that fraudulent jobs were generally from engineering or healthcare while genuine jobs were from more specialised areas such as computer programming and creative designing.

## Problem Statement
Given the increasing number of job scams, especially during the COVID-19 pandemic, we aim to train several classification models including Logistic Regression, Random Forest and Naive Bayes before selecting the best classifier to predict whether jobs are real or fake to prevent job-seekers from falling into job scams. The classifier will be incorporated to job portals such that if a job ad is predicted to be fraudulent or fake, the listing will not be published on the portal. A successful model would be one with a high ROC AUC score (>0.9) and high recall score. We want to minimise the False Negatives as the impact of incorrectly predicting a fraudulent job as a genuine might lead to job-seekers falling prey to job scams and put them in a worse situation that what they are already facing.

## Background
The COVID-19 pandemic has caused unemployment rate to increase due to the economic slowdown. As a result, many are left unemployed and vulnerable, becoming easy targets for job scammers. The Better Business Bureau has reported more than 13,000 job listing scams in North America alone between December 2019 to April 2020. An annual [Flexjob survey](https://www.flexjobs.com/blog/post/how-to-find-a-real-online-job-and-avoid-the-scams-v2/) found that more than 80% of job seekers report being on guard or very concerned about scams on other job boards. According to the same survey, almost 20% of job seekers have been a victim of a job scam (up from 13% in 2016), with 22% of job seekers knowing somebody who has been victim of a job scam.

There are some generic indicators of job listings which are fraudulent such as:
- Grammatical or spelling errors
- High paying jobs with low requirements
- Urgency to fill the position
- Requiring sensitive info/upfront payment 

However, there may be other factors which are not so direct in determining that the job listing is fradulent. Furthermore, it is easy for anyone to publish job ads on job portals or messaging apps today, regardless of whether the job is genuine or fraudulent.

## Data Dictionary

|Feature            |Type     |Description|
|-------------------|---------|-----------| 
|company_profile    |*integer*|Whether the Job Posting comes with a Company Profile (1 if yes; 0 if no)|
|telecommuting      |*integer*|Whether the Job offers Telecommuting Option (1 if yes; 0 if no)|
|has_company_logo   |*integer*|Whether the Job Posting comes with a Company Logo (1 if yes; 0 if no)|
|has_questions      |*integer*|Whether there are Interview Questions (1 if yes; 0 if no)|
|employment_type    |*object* |Type of Employment|
|required_experience|*object* |Minimum Expereience required for the Job|
|required_education |*object* |Minimum Education Level required for the Job|
|industry           |*object* |Industry of the Job|
|industry_grp       |*object* |Grouped Industry of the Job |
|text_lemma         |*object* |Lemmatized form of combined info on Job Title, Job Description and Requirements|
|word_length        |*integer* |Number of Words in text_lemma|
|char_length        |*integer* |Number of Characters in text_lemma|
|fraudulent         |*integer*|**Target Variable** - Whether a Job Posting is fraudulent (1 if yes; 0 if no)|

## Model Evaluation
A summary of the models' scores is appended in the table below. 

There seem to be overfitting across all models as the train scores are higher than the test scores. Generally, the performance of the models have also improved after SMOTE is performed. The top 2 models with the highest test ROC AUC score is the Random Forest with cvec at 0.951, followed by the Random Forest with tvec at 0.950. However, the recall scores for both models are below 0.5 and this is not ideal. If the recall score is low, it means that the model is not good at predicting fraudulent (positives) jobs but the objective of the classifier is to correctly identify fraudulent (positives) jobs. The next best model is the Logistic Regression with tvec at 0.946. It also have the highest recall score of 0.801 across all other models, which means that it can predict the positive classes correctly 80% of the time. Hence, the model that we selected is the **Logistic Regression with tvec (SMOTE)**.

| S/N | Model | Train Accuracy | Test Accuracy | Train ROC AUC | Test ROC AUC | Train Recall | Test Recall | 
|:---:|:---|:---|:---|:---|:---|:---|:---|
| 1 | Logistic Regression with cvec       | 1.000 | 0.966 | 1.000 | 0.925 | 1.000 | 0.664 |
| 2 | **Logistic Regression with tvec**   | 0.961 | 0.948 | 0.995 |**0.946**| 0.991 | **0.801** |
| 3 | Support Vector Classifier with cvec | 1.000 | 0.971 | 1.000 | 0.890 | 1.000 | 0.637 |
| 4 | Support Vector Classifier with tvec | 0.999 | 0.978 | 1.000 | 0.897 | 1.000 | 0.630 |
| 5 | Extreme Gradient Boost with cvec    | 1.000 | 0.974 | 1.000 | 0.917 | 1.000 | 0.582 |
| 6 | **Extreme Gradient Boost with tvec**| 1.000 | 0.978 | 1.000 |**0.945**| 1.000 | 0.610 |
| 7 | Multinomial Naive Bayes with cvec   | 0.962 | 0.953 | 0.979 | 0.923 | 0.876 | 0.664 |
| 8 | Multinomial Naive Bayes with tvec   | 0.922 | 0.922 | 0.946 | 0.919 | 0.776 | 0.712 |
| 9 | K Nearest Neighbors with cvec       | 0.927 | 0.892 | 1.000 | 0.877 | 1.000 | 0.760 |
|10 | K Nearest Neighbors with tvec       | 0.928 | 0.896 | 1.000 | 0.862 | 1.000 | 0.760 |
|11 | **Random Forest with cvec**         | 1.000 | 0.975 | 1.000 |**0.951**| 1.000 | 0.459 |
|12 | **Random Forest with tvec**         | 1.000 | 0.977 | 1.000 |**0.950**| 1.000 | 0.466 |
|13 | AdaBoost with cvec                  | 0.990 | 0.969 | 0.998 | 0.939 | 0.818 | 0.568 |
|14 | AdaBoost with tvec                  | 0.999 | 0.972 | 1.000 | 0.906 | 0.976 | 0.582 |

## Conclusions
The Logistic Regression with TFIDF Vectorizer is able to accurately distinguish between the fraudulent (fake) and non-fraudulent (real) jobs, with a high ROC AUC score of 0.946 and recall score of 0.801. If we deploy this model to job portals, it will be able to correctly predict the fraudulent jobs 80% of the time and prevent fraudulent jobs from being published on job portals. With the reduction in fraudulent job listings, this largely reduced the risk of job-seekers falling prey to job scams. However, there is also a trade-off between recall and precision. The model has a high recall score (0.801) but a low precision score (0.429). This means that the model may predict more real job ads as fake resulting in genuine jobs not being able to be published on job portals, causing them to have a difficult time finding candidates to fill their positions. These employers can then choose to amend their job ads or use other channels to source for candidates such as via internal referrals.

To take this project further, we could explore the following:
- Getting updated jobs data in recent years (e.g. from 2019 when COVID-19 happened)
- Getting additional data on fraudulent jobs as and when there are reports of such cases
- Collect more features such as platform the job ad was posted on, and the nature of the job
- Further fine-tune hyperparameters, understand context before removing stopwords 
- Try other oversampling techniques

## Recommendations
While we are able to train a classifier to aid in filtering fraudulent job ads from being published on job portals, job-seekers should also be cautious when applying for jobs. If a job posting does not come with a company logo or profile, job-seekers should do some research on the company and ask around or check for online reviews before applying. 

Based on the top features of the model and topic modelling, there seem to be many jobs on engineering or admin which are fraudulent. Job-seekers are advised to be wary of 'too-good-to-be-true' jobs. If the job is high paying and technical but has low requirements, job-seekers should pay extra attention as scrutinize the job ad further. 

Job-seekers should also look for jobs of their interest based on more specialised skills such as computing/programming instead of generic roles requiring basic skills like microsoft office. Even if the recruitment process may be more tedious, at least it puts job-seekers at a safer spot with lower risk of falling prey for job scams. 

As the classifier is not 100% accurate or robust, job-seekers or any users who come across suspicious job ads such as those requesting for sensitive information or payment should report these listings so that action can be taken against the original poster.

## References
1. http://emscad.samos.aegean.gr/ 
2. https://www.american.edu/careercenter/fraudulentjobs.cfm 
3. https://www.flexjobs.com/blog/post/how-to-find-a-real-online-job-and-avoid-the-scams-v2/
4. https://insights.omnia-health.com/hospital-management/employment-fraud-rates-increase-30-cent-during-pandemic
5. https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
6. https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing#scrollTo=ScBUgXn06IK6
