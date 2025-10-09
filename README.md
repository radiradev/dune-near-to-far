# Near to Far Translation
A generative model based on [min-gpt](https://github.com/karpathy/minGPT). It learns the conditional distribution over the `fd` recontructed variables given `nd` reconstructed variables:
$$p(x_{FD} | x_{ND})$$

We learn this autoregressively, i.e the transformer is trained to predict:
$$p(x_{FD} | x_{ND}) = \prod_i p(x_{i_{FD}}| x_{1_{FD}}, x_{2_{FD}}, ..., x_{i-1_{FD}}, x_{ND}) $$

We learn each `fd` dimension as Gaussian Mixture distribution that has been changed in some way. 
For the CVN scores we transform the distribution using a sigmoid to ensure it stays within the range $[0, 1]$.

\_ 
## Train a Model

To train a model do:
```
python gpy_train.py <data_path> <work_dir>
```
Where `data_path` is the training data csv file and `work_dir` is the path to a directory (that does not have to exist) where data related to the training will be written. Hyperparameters may be set from the command line with the `-o` flag, check `python gpt_train.py --help` to see how. To understand the hyperparameters and what values they may take, you need to read the code :).

The `gpt_train.py` script has many arguments related to resampling/reweighting the training dataset. A decent amount of the codebase also handles all this. In the end, I found this to not be useful so never use it. But it is all still there just incase. The idea is you may want your training set to mimic different underlying distributions such as an oscillated FD true neutrino energy spectrum.

## Evaluating a Model

Once you have trained a model, the `work_dir` for this training will contain the best (by validation loss) model checkpoint, the loss curve data, and the hyperparameters defining the experiment.

You can plot the loss curve using the `scripts/plot_lc.py` script.

You can produce evaluation plots (that will be written to the `work_dir`) using:
```
python gpt_sample.py <data_path> <work_dir>
```
There are a few optional arguments. You should probably always use `--resample_negative_preds` which will resample the Gaussian mixture if the result is negative. `--pdf_plots` plots the predicted Gaussian mixture for some number of events, can be interesting. It is useful to know how the model performs when the test dataset distribution is changed, this can be done for, say, FD oscillated true energy with  `--apply_sample_weights_from data/prism_nufit_target_fd_flux_norate`. The predicted distributions should match the truth well under any true neutrino energy distribution.

## Generating Paired Data for Training 

A paired dataset `.h5` file is required for training. This contains ND and FD reco variables from the paired data generation procedure (see `AlexWilkinsonnn/ndfd-pairs`). This `.h5` is converted into a `.csv` using the script `scripts/cut.py`. 

## Software Requirements
wtih `pip` (also possible with conda):

`pip install -r requirements.txt`
 

