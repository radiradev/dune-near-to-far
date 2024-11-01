import os, argparse, warnings
from collections.abc import MutableMapping
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import json
import yaml
import scipy.optimize
import scipy.stats

from gpt.utils import set_seed, setup_logging, CfgNode as CN
from gpt.model import GPT
from gpt.dataset import NewPairedData

import dunestyle.matplotlib as dunestyle

def get_config(work_dir):
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = work_dir

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    return C

# Magic stack overflow thingy to flatten nested dicts
def flatten(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def get_new_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [
        Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles
    ]
    return new_handles, labels

def gauss_fit_func(x, a, mu, sigma):
    return a * scipy.stats.norm.pdf(x, loc=mu, scale=sigma)

def diff_plot(bins, true, pred, xlabel, savename, frac=False, fit=True, clip=60):
    fig, ax = plt.subplots(1, 1, figsize=(8,6), layout="compressed")
    ax.vlines(
        0, ymin=0, ymax=1, linestyle="dashed", linewidth=1, transform=ax.get_xaxis_transform()
    )
    if clip is not None:
        pred = pred.copy()
        pred[pred > 60] = 60
    diffs = pred - true
    if frac:
        diffs = diffs / true
    ax.hist(diffs, bins=bins, histtype="step")
    if fit:
        nphist, npbins = np.histogram(diffs, bins=bins)
        bin_centres = npbins[:-1] + (npbins[1] - npbins[0])
        params, _ = scipy.optimize.curve_fit(gauss_fit_func, bin_centres, nphist, [100, 0, 10])
        fit_x = np.linspace(npbins[0], npbins[-1], 1000)
        fit_y = gauss_fit_func(fit_x, *params)
        ax.plot(fit_x, fit_y, c="r")
        ax.text(
            0.8, 0.9, r'$\mu =$' + f"{params[1]:.3f}\n" + r'$\sigma = $' + f"{params[2]:.3f}",
            ha="left", va="top", transform=ax.transAxes, fontsize=16
        )
    ax.set_ylabel("No. Events", fontsize=16, loc="top")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_xlim(left=bins[0], right=bins[-1])
    plt.savefig(os.path.join(args.work_dir, savename))
    plt.close()

def dist_plot(bins, true, pred, xlabel, savename, nd=None):
    fig, ax = plt.subplots(1, 1, figsize=(8,6), layout="compressed")
    if nd is not None:
        ax.hist(nd, bins=bins, histtype="step", label="ND", linestyle="dashed")
    ax.hist(true, bins=bins, histtype="step", label="True")
    ax.hist(pred, bins=bins, histtype="step", label="Pred")
    new_handles, labels = get_new_handles(ax)
    ax.legend(new_handles, labels, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=16, loc="right")
    ax.set_ylabel("No. Events", fontsize=16, loc="top")
    ax.set_xlim(left=bins[0], right=bins[-1])
    plt.savefig(os.path.join(args.work_dir, savename))
    plt.close()

def dist2d_plot(
    n_bins, range_bins, true_x, true_y, pred_x, pred_y, x_label, y_label, savename, logscale=False
):
    true_hist2d, bins_x, bins_y = np.histogram2d(true_x, true_y, bins=n_bins, range=range_bins)
    pred_hist2d, _, _ = np.histogram2d(pred_x, pred_y, bins=n_bins, range=range_bins)
    fig, ax = plt.subplots(1, 2, figsize=(14,6), layout="compressed")
    # extent = [bins_y[0], bins_y[-1], bins_x[0], bins_x[-1]]
    extent = [bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]]
    vmin = np.min([np.min(true_hist2d[true_hist2d != 0]), np.min(pred_hist2d[pred_hist2d != 0])])
    vmax = np.max([np.max(true_hist2d), np.max(pred_hist2d)])
    if logscale:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax[0].imshow(
        np.ma.masked_where(pred_hist2d == 0, pred_hist2d).T,
        origin="lower", interpolation="none", extent=extent, norm=norm, cmap="cividis", aspect="auto"
    )
    add_identity(ax[0], color="r", linestyle="dashed")
    ax[1].imshow(
        np.ma.masked_where(true_hist2d == 0, true_hist2d).T,
        origin="lower", interpolation="none", extent=extent, norm=norm, cmap="cividis", aspect="auto"
    )
    add_identity(ax[1], color="r", linestyle="dashed")
    cb = fig.colorbar(im, ax=[ax[0], ax[1]], orientation="vertical", location="right")
    cb.set_label("No. Events", fontsize=12)
    for a in ax.flatten():
        a.set_xlabel(x_label, fontsize=16, loc="right")
        a.set_ylabel(y_label, fontsize=16, loc="top")
        a.xaxis.set_tick_params(labelsize=12)
        a.yaxis.set_tick_params(labelsize=12)
    ax[0].set_title("Model Prediction", fontsize=18, pad=15)
    ax[1].set_title("Paired Dataset", fontsize=18, pad=15)
    plt.savefig(os.path.join(args.work_dir, savename))
    plt.close()

# Thanks stackoverflow
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def get_stats(true_df, pred_df):
    metrics = {}
    pred_df = pred_df.copy()
    pred_df.loc[pred_df["fd_numu_nu_E"] > 60, "fd_numu_nu_E"] = 60
    metrics["all_nuE_mae"] = float(
        np.mean(np.abs(true_df["fd_numu_nu_E"] - pred_df["fd_numu_nu_E"]))
    )
    metrics["min_max_nuE_pred"] = [
        float(np.min(pred_df["fd_numu_nu_E"])), float(np.max(pred_df["fd_numu_nu_E"]))
    ]
    metrics["min_max_nuE_true"] = [
        float(np.min(true_df["fd_numu_nu_E"])), float(np.max(true_df["fd_numu_nu_E"]))
    ]
    metrics["all_cvnnumu_mae"] = float(
        np.mean(np.abs(true_df["fd_numu_score"] - pred_df["fd_numu_score"]))
    )
    with open(os.path.join(args.work_dir, "metrics.yml"), "w") as f:
        yaml.dump(metrics, f)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device is {device}')

    config = get_config(args.work_dir)
    with open(os.path.join(args.work_dir, "config.json")) as f:
        exp_config = json.load(f)
    merge_args = [
        "--" + flat_arg + "=" + str(val)
        for flat_arg, val in flatten(exp_config).items()
            if "trainer" not in flat_arg and flat_arg != "system.work_dir"
    ]
    config.merge_from_args(merge_args)
    test_dataset = NewPairedData(data_path=args.data_path, train=False)
    config.model.block_size = test_dataset.get_block_size()
    config.model.scores_size = test_dataset.get_scores_length()
    config.model.far_reco_size = test_dataset.get_far_reco_length()

    model = GPT(config.model)
    model.load_state_dict(torch.load(os.path.join(config.system.work_dir, 'model.pt')))
    model = model.eval()

    print(len(test_dataset.near_reco), len(test_dataset.far_reco), len(test_dataset.cvn_scores))

    true_df = pd.read_csv(args.data_path)

    test_dataset = NewPairedData(data_path=args.data_path, train=False)

    def get_df(pred_x, true_x=None):
        col_names = test_dataset.near_reco + test_dataset.cvn_scores + test_dataset.far_reco
        assert len(col_names) == pred_x.shape[1]
        df = pd.DataFrame(pred_x, columns=col_names)
        df['class'] = 'predicted'
        df_true = pd.DataFrame(true_x, columns=col_names)

        if true_x is not None:
            assert len(col_names) == true_x.shape[1]
            df_true['class'] = 'true'
            df = pd.concat([df, df_true])
        return df

    batch_size = 2000
    num_iter = 90
    model = model.to(device)

    # shuffle it
    test_dataset.data = test_dataset.data[np.random.permutation(len(test_dataset.data))]
    pred_list = []
    for i in range(num_iter):
        idx = torch.tensor(test_dataset.data[:, :len(test_dataset.near_reco)], dtype=torch.float).to(device)[i*batch_size:(i+1)*batch_size]
        pred = model.generate(idx, device='cuda').cpu().numpy()
        pred_list.append(pred)
    pred = np.concatenate(pred_list)

    df = get_df(pred, test_dataset.data[:len(pred), :])

    # ignore future warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    sns.set_context('talk')
    bins = np.linspace(0.0, 1.0, 40)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    common_kwargs = dict(hue='class', stat='density', element='step', common_norm=True, bins=bins, fill=False, log_scale=True)
    sns.histplot(data=df, x='fd_numu_score', hue='class', stat='density', element='step', common_norm=True, bins=bins, ax=ax, fill=False, legend=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(args.work_dir, "plot_1.pdf"))
    plt.close()

    # My distribution and residual plots
    dist_plot(
        np.linspace(0.0, 1.0, 50), 
        df[df["class"] == "true"]["fd_numu_score"],
        df[df["class"] == "predicted"]["fd_numu_score"],
        "CVN numu Score",
        "cvn_dist_plot.pdf"
    )
    diff_plot(
        np.linspace(-0.25, 0.25, 100), 
        df[df["class"] == "true"]["fd_numu_score"],
        df[df["class"] == "predicted"]["fd_numu_score"],
        "(Pred - True) CVN numu Score",
        "cvn_diff_plot.pdf"
    )
    dist_plot(
        np.linspace(0.0, 16.0, 80), 
        df[df["class"] == "true"]["fd_numu_nu_E"],
        df[df["class"] == "predicted"]["fd_numu_nu_E"],
        r'$E_\nu^{\mathrm{reco}}$ (GeV)',
        "nuE_dist_plot.pdf",
        nd=df[df["class"] == "true"]["Ev_reco"]
    )
    diff_plot(
        np.linspace(-1.0, 1.0, 100), 
        df[df["class"] == "true"]["fd_numu_nu_E"],
        df[df["class"] == "predicted"]["fd_numu_nu_E"],
        r'(Pred - True) / True $E_\nu^{\mathrm{reco}}$',
        "nuE_diff_plot.pdf",
        frac=True
    )
    dist_plot(
        np.linspace(0.0, 16.0, 80), 
        df[df["class"] == "true"]["fd_numu_lep_E"],
        df[df["class"] == "predicted"]["fd_numu_lep_E"],
        r'$E_{\mathrm{lep}}^{\mathrm{reco}}$ (GeV)',
        "lepE_dist_plot.pdf",
        nd=df[df["class"] == "true"]["Elep_reco"]
    )
    diff_plot(
        np.linspace(-1.0, 1.0, 100), 
        df[df["class"] == "true"]["fd_numu_lep_E"],
        df[df["class"] == "predicted"]["fd_numu_lep_E"],
        r'(Pred - True) / True $E_{\mathrm{lep}}^{\mathrm{reco}}$',
        "lepE_diff_plot.pdf",
        frac=True
    )
    dist_plot(
        np.linspace(0.0, 10.0, 100), 
        df[df["class"] == "true"]["fd_numu_had_E"],
        df[df["class"] == "predicted"]["fd_numu_had_E"],
        r'$E_{\mathrm{had}}^{\mathrm{reco}}$ (GeV)',
        "hadE_dist_plot.pdf",
        nd=df[df["class"] == "true"]["Ev_reco"] - df[df["class"] == "true"]["Elep_reco"]
    )
    diff_plot(
        np.linspace(-1.0, 1.0, 100), 
        df[df["class"] == "true"]["fd_numu_had_E"],
        df[df["class"] == "predicted"]["fd_numu_had_E"],
        r'(Pred - True) / True $E_{\mathrm{had}}^{\mathrm{reco}}$',
        "hadE_diff_plot.pdf",
        frac=True
    )
    dist2d_plot(
        70, ((0, 14), (0, 14)),
        np.array(df[df["class"] == "true"]["Ev_reco"]),
        np.array(df[df["class"] == "true"]["fd_numu_nu_E"]),
        np.array(df[df["class"] == "predicted"]["Ev_reco"]),
        np.array(df[df["class"] == "predicted"]["fd_numu_nu_E"]),
        r'ND $E_\nu^{\mathrm{reco}}$ (GeV)',
        r'FD $E_\nu^{\mathrm{reco}}$ (GeV)',
        "ndfd_nuE_hist2d_true_pred.pdf"
    )
    dist2d_plot(
        (60, 70), ((0, 12), (0, 14)),
        np.array(df[df["class"] == "true"]["Elep_reco"]),
        np.array(df[df["class"] == "true"]["fd_numu_nu_E"]),
        np.array(df[df["class"] == "predicted"]["Elep_reco"]),
        np.array(df[df["class"] == "predicted"]["fd_numu_nu_E"]),
        r'ND $E_{\mathrm{lep}}^{\mathrm{reco}}$ (GeV)',
        r'FD $E_\nu^{\mathrm{reco}}$ (GeV)',
        "ndfd_lepE_hist2d_true_pred.pdf"
    )
    dist2d_plot(
        (40, 70), ((0, 3), (0, 14)),
        np.array(df[df["class"] == "true"]["eRecoP"]),
        np.array(df[df["class"] == "true"]["fd_numu_nu_E"]),
        np.array(df[df["class"] == "predicted"]["eRecoP"]),
        np.array(df[df["class"] == "predicted"]["fd_numu_nu_E"]),
        r'ND $E_{\mathrm{proton}}^{\mathrm{reco}}$ (GeV)',
        r'FD $E_\nu^{\mathrm{reco}}$ (GeV)',
        "ndfd_protonE_hist2d_true_pred.pdf",
        logscale=True
    )
    dist2d_plot(
        (40, 70), ((0, 3), (0, 14)),
        (
            np.array(df[df["class"] == "true"]["eRecoPip"]) +
            np.array(df[df["class"] == "true"]["eRecoPim"])
        ),
        np.array(df[df["class"] == "true"]["fd_numu_nu_E"]),
        (
            np.array(df[df["class"] == "predicted"]["eRecoPip"]) +
            np.array(df[df["class"] == "predicted"]["eRecoPim"])
        ),
        np.array(df[df["class"] == "predicted"]["fd_numu_nu_E"]),
        r'ND $E_{\pi^\pm}^{\mathrm{reco}}$ (GeV)',
        r'FD $E_\nu^{\mathrm{reco}}$ (GeV)',
        "ndfd_pipmE_hist2d_true_pred.pdf",
        logscale=True
    )
    dist2d_plot(
        (40, 70), ((0, 3), (0, 14)),
        np.array(df[df["class"] == "true"]["eRecoPi0"]),
        np.array(df[df["class"] == "true"]["fd_numu_nu_E"]),
        np.array(df[df["class"] == "predicted"]["eRecoPi0"]),
        np.array(df[df["class"] == "predicted"]["fd_numu_nu_E"]),
        r'ND $E_{\pi^0}^{\mathrm{reco}}$ (GeV)',
        r'FD $E_\nu^{\mathrm{reco}}$ (GeV)',
        "ndfd_pi0E_hist2d_true_pred.pdf",
        logscale=True
    )

    get_stats(df[df["class"] == "true"], df[df["class"] == "predicted"])

    nu_true = true_df['Ev']
    nu_nd_reco = true_df['Ev_reco']
    fd_numu_nuE_true = true_df['fd_numu_nu_E']
    pred_numu_nuE = df['fd_numu_nu_E'][df['class'] == 'predicted']

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bins = np.linspace(0, 20, 100)
    common_kwargs = dict(bins=bins, density=True, histtype='step', lw=2.0)
    # ax.hist(nu_true, label='true_Ev', **common_kwargs, color='black')
    ax.hist(nu_nd_reco, label='nd_reco_Ev', **common_kwargs, linestyle='-', color='tab:blue')
    ax.hist(pred_numu_nuE, label='pred_fd_numu_nu_E', **common_kwargs, color='tab:red')
    ax.hist(fd_numu_nuE_true, label='true_fd_numu_nu_E', **common_kwargs, color='gray', alpha=1.0)

    ax.legend()
    ax.set_xlabel('Energy [GeV]')
    # ticks at ever 2 GeV
    ax.set_xticks(np.arange(0, 20, 2))
    plt.grid()
    plt.savefig(os.path.join(args.work_dir, "plot_2.pdf"))
    plt.close()

    bins = np.linspace(0, 25, 25)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    common_kwargs = dict(hue='class', stat='density', element='step', common_norm=True, bins=bins, fill=False)

    sns.histplot(data=df, x='fd_numu_nu_E', ax=ax[0, 0], **common_kwargs, legend=True)
    sns.histplot(data=df, x='fd_numu_had_E', ax=ax[0, 1], **common_kwargs, legend=False)
    sns.histplot(data=df, x='fd_numu_lep_E', ax=ax[1, 0], **common_kwargs, legend=False)

    plt.tight_layout()
    plt.savefig(os.path.join(args.work_dir, "plot_3.pdf"))
    plt.close()

    sns.set_context('talk')
    def by_name(name, x):
        """Select column by name"""
        index = df.columns.get_loc(name)
        return x[:, index]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    y_col = 'Ev_reco' #'eRecoP', 'eRecoN', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther', 'Ev_reco', 'Elep_reco', 'theta_reco'
    x_col = 'fd_numu_nu_E' # 'numu_lep_E', 'numu_had_E', 'numu_nu_E', 'nue_lep_E', 'nue_had_E', 'nue_nu_E'
    x = by_name(x_col, pred)
    y = by_name(y_col, pred)
    bins_y = np.linspace(0, 8.0, 50)
    bins_x = np.linspace(0, 8, 50)
    ax[0].hist2d(x, y, bins=(bins_x, bins_y), cmin=0)
    ax[0].set_xlabel(x_col)
    ax[0].set_ylabel(y_col)
    ax[0].set_title('predicted')

    true = test_dataset.data[:batch_size*num_iter]
    x = by_name(x_col, true)
    y = by_name(y_col, true)
    ax[1].hist2d(x, y, bins=(bins_x, bins_y), cmin=0)
    ax[1].set_xlabel(x_col)
    ax[1].set_title('true')

    plt.savefig(os.path.join(args.work_dir, "plot_4.pdf"))
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", type=str, help="Training data csv.")
    parser.add_argument("work_dir", type=str, help="Experiment directory, somewhere in out/.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

