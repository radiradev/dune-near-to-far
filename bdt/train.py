from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import uproot
import os
import pickle
import numpy as np

def load_data(data_path, near_variables=None, far_variables=None):
    tree = uproot.open(data_path)['nd_fd_reco']
    input = tree.arrays(near_variables, library='pd')
    target = tree.arrays(far_variables, library='pd')
    return input, target

root_dir = '/global/cfs/cdirs/dune/users/rradev/near_to_far'
data_path = os.path.join(root_dir, 'paired_ndfd.root')
load_data(data_path)

regressor = LGBMRegressor(boosting_type='gbdt', verbose=3)

near_det = [
    'eRecoP', 'eRecoN', 'eRecoPip', 
    'eRecoPim', 'eRecoPi0', 'eRecoOther', 
    'Ev_reco', 'Elep_reco', 'theta_reco'
]

far_det = [
    'nc_nu_E',
]

input, target = load_data(data_path, near_variables=near_det, far_variables=far_det)

X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state=42)
regressor.fit(X_train, y_train)
# save the model to disk
filename = 'lgbm_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

# evaluate the model
y_pred = regressor.predict(X_test)
print(mean_squared_error(y_test, y_pred))


import matplotlib.pyplot as plt

plt.rcParams.figure_figsize = (20, 10)
fig, axs = plt.subplots(1, 2)
y_test = y_test.to_numpy().flatten()
error = y_pred - y_test

axs[0].hist(error, bins=25)
axs[0].set_title('Prediction Error')

axs[1].scatter(y_test, y_pred, marker='o', alpha=0.1')
axs[1].set_title('Prediction vs True Value')
axs[1].set_xlabel('True Value')
axs[1].set_ylabel('Predictions')
plt.savefig('lgbm_error.png')