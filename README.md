# Credit Risk Classification: Module 20

## Objective

The goal for the module was to create a supervised learning logistic regression model that can accurately identify and differentiate a high-risk loan from a healthy one, using a dataset of lending activity from peer-to-peer services. Two logistic regression models were generated for comparison: one trained with the original data and the other trained with a resampling of the data with the `RandomOverSampler` module from the imbalanced-learn library.

## Dependencies and Implementation
This is performed in Python using Jupyter notebooks and Pandas, along with `train_test_split`, `logistic_regression`, `balanced_accuracy_score`, `confusion_matrix`, and `classification_report` modules from the sklearn library, as well as the `RandomOverSampler` module from the imbalanced-learn library.

## Analysis Report

The lending data used for this analysis is entirely numerical with the following columns acting as the features:
- Loan Size
- Interest Rate
- Borrower Income
- Debt to Income Ratio
- Number of Accounts
- Derogatory Marks
- Total Debt

### Logistic Regression Model 1 (Original Data)
In this analysis, `loan_status = 0` qualifies as a healthy loan, while `loan_status = 1` denotes a high-risk loan. Using `value_counts()` on the `loan_status` column of the original data yields 75036 identified healthy loans and 2500 identified high-risk loans, which is a stark class imbalance. After splitting the data for training and testing, the original data was fit to a logistic regression model with a `random_state = 1`. This `random_state` would remain constant to test both models for consistency. The results on the original data logistic regression model are as follows:

- The Balanced Accuracy Score of 95.204%, while promising, is not entirely reliable due to class imbalance identifying healthy loans versus high-risk loans present in the original data.
- This model performs incredibly well in both predicting and identifying healthy loans, with a perfect precision score of 100% and a near-perfect recall of 99%.
- For high-risk loans, the model still performs very well, but less so when compared to identifying healthy loans identifying 91% of the high-risk loans correctly.
- The model takes a more cautious approach to predicting a high-risk loan, with only 85% predicted to be high-risk loans.
- Despite the 99% accuracy of the model in the classification report, the imbalance present in the classes of the original data lends to how the model performs better when predicting a healthy loans as opposed to predicting a high-risk loan.
- This model generated 56 False Negatives, meaning it incorrectly identified 56 high-risk loans as healthy ones. While this is a low number relative to the total number of results in the data, it's still accounts for 9.9% of the high-risk loans present in the data and this is by far the most costly and most signficant result to avoid.   

### Logistic Regression Model 2 (Resampled Data)
In order to alleviate the class imbalance present in the original data, a second model was generated after using the `RandomOverSampler` module. Using `value_counts()` on the `loan_status` column of the original data yields 56271 identified healthy loans and 56271 identified high-risk loans, a far-cry from the previous model. Running this resampled data through a second logistic regression model generated the following findings:

- The Balanced Accuracy Score improved greatly from 95.204% to 99.367%, indicating the model has a near-perfect ability to properly recall healthy and high-risk loans.
- The precision and recall of healthy loans remains the same as it was in the original data.
- The resampled data greatly improves the recall of high-risk loans, offering a near-perfect identification of what qualifies as a high-risk loan, up to 99%, as compared to 91% in the previous model. This is also evident in the False Negatives, where the resampled model had only 4 compared to the 56 present in the original model.
- The resampled model offers a very minor decrease in its precision of predicting high-risk loans (85% to 84%), but offers significantly increased performance at avoiding mislabeling a high-risk loan as a healthy loan.

### Summary

The resampled logistic regression model enhanced the performance of the area where the original model was lacking: identifying high-risk loans. While the original model performed incredibly well at predicting and identifying healthy loans, resampling the class imbalance enabled the model to maintain the accuracy it already had in that area and improved upon it, with only a very minor decrease in the precision of high-risk loans. The better performance on minimizing False Negatives, and the near perfect Balanced Accuracy Score highlight that this model excels at identifying the creditworthiness of potential borrowers. As the resampled data model offers the highest chance to avoid risking significant losses by mislabeling high-risk loans, I can confidently say it has my recommendation.
