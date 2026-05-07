# Get most recent checking code
!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git
from learntools.ml_explainability.ex4 import *
print("Setup Complete")


import pandas as pd
data = pd.read_csv('../input/hospital-readmissions/train.csv')
data.columns


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/hospital-readmissions/train.csv')

y = data.readmitted

base_features = [c for c in data.columns if c != "readmitted"]

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)


# Your code here
import eli5
from eli5.sklearn import PermutationImportance

imp = PermutationImportance(my_model, random_state=1)
imp.fit(val_X, val_y)
cols = list(val_X.columns)
eli5.show_weights(imp, feature_names=cols)


# Your Code Here
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

col = 'number_inpatient'
plot = PartialDependenceDisplay.from_estimator(my_model, val_X, [col])
plt.show()


# Your Code Here
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

col = 'time_in_hospital'
pdp = PartialDependenceDisplay.from_estimator(my_model, val_X, [col])
plt.show()


# Your Code Here
train_data = pd.concat([train_X, train_y], axis=1)
train_data.groupby('time_in_hospital')['readmitted'].mean().plot()
plt.show()


# Your Code Here
import shap

row = val_X.iloc[0].astype(float)
def patient_risk_factors(model, patient_data):
    exp = shap.TreeExplainer(model)
    vals = exp.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(exp.expected_value[1], vals[1], patient_data)
  
