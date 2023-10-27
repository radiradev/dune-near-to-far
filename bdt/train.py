from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import uproot
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
target = np.log(np.clip(target, 0.03, None))
X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state=42)
regressor.fit(X_train, y_train)
# save the model to disk
filename = 'lgbm_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

# evaluate the model
y_pred = regressor.predict(X_test)
print("MSE in log space:", mean_squared_error(y_test, y_pred))
print("MSE in linear space:", mean_squared_error(np.exp(y_test), np.exp(y_pred)))

plt.rcParams['figure.figsize'] = [12, 6]
fig, axs = plt.subplots(1, 2)

y_test = y_test.to_numpy().flatten()
error = y_pred - y_test

axs[0].hist(error, bins=100, range=(-2, 2))
axs[0].set_title('Prediction Error log(energy)')
axs[0].set_xlabel('(True - Predicted)')


axs[1].hist2d(y_test, y_pred, bins=50, range=[[-2, 3], [-2, 3]], cmin=0)
# axs[1].scatter(y_test, y_pred, marker='o', alpha=0.5, s=1)
# # add line through middle
# x = np.linspace(-2, 3, 100)
# axs[1].plot(x, x, color='tab:red', linestyle='--')
axs[1].set_title('Prediction vs True Value')
axs[1].set_xlabel('True Energy log[GeV]')
axs[1].set_ylabel('Predicted Energy log[GeV]')
plt.savefig('lgbm_error.png')

plt.clf()
plt.rcParams['figure.figsize'] = [8, 6]
plt.hist(y_pred, bins=30, label='Predicted', histtype='step', linewidth=2, density=True, range=(-3, 3));
plt.hist(y_test, bins=30, label='True', histtype='step', linewidth=2, density=True, range=(-3, 3), color='tab:red');
plt.xlabel('Log Energy')
plt.ylabel('Count')
plt.legend()
plt.title('Energy Distribution')
plt.savefig('lgbm_energy_dist.png')