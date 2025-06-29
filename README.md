# credit-risk-model
# Credit Scoring Business Understanding

## Basel II Accord's Influence
The Basel II Capital Accord emphasizes accurate risk measurement and capital adequacy, requiring banks to use interpretable and well-documented credit risk models. Interpretable models, such as Logistic Regression with Weight of Evidence (WOE), allow regulators to audit and validate risk assessments, ensuring compliance and transparency in loan decisions.

## Necessity of Proxy Variable
Since the dataset lacks a direct "default" label, a proxy variable based on Recency, Frequency, and Monetary (RFM) metrics is necessary to classify customers as high-risk or low-risk. However, this approach risks misclassification if the proxy does not accurately reflect default behavior, potentially leading to incorrect loan approvals or denials, impacting the bank’s financial stability.

## Model Trade-offs
Simple models like Logistic Regression with WOE are highly interpretable, facilitating regulatory compliance, but may have lower predictive power. Complex models like Gradient Boosting Machines offer superior performance but are less transparent, posing challenges in regulated financial contexts where explainability is critical.