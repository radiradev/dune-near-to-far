""" Start - training reweight/resampler helpers """

def read_reweight_dir(reweight_dir):
    bins_file = glob.glob(os.path.join(reweight_dir, "*_bins.npy"))
    assert len(bins_file) == 1, "Invalid reweight dir structure."
    weight_bins = np.load(bins_file[0])
    hist_file = glob.glob(os.path.join(reweight_dir, "*_hist.npy"))
    assert len(hist_file) == 1, "Invalid reweight dir structure."
    weight_hist = np.load(hist_file[0])
    var_file = glob.glob(os.path.join(reweight_dir, "*_var.txt"))
    assert len(var_file) == 1, "Invalid reweight dir structure."
    with open(var_file[0], "r") as f:
        var_name = f.read().rstrip("\n")
    return weight_bins, weight_hist, var_name

# Reweights s.t. the most energies are flat and the rest is almost flat
# (very large weights # at the extreme energies can make training unstable)
def get_reweight_uniform(train_sample_weight_var_data):
    bins = np.arange(0.0, 14.25, 0.25)
    train_hist, _ = np.histogram(train_sample_weight_var_data, bins=bins)
    train_hist = train_hist.astype(float)
    train_hist /= np.sum(train_hist)
    target_hist = np.ones_like(train_hist).astype(float)
    target_hist /= np.sum(target_hist)
    ratio_hist = target_hist / train_hist

    bins = np.concatenate([bins, [120.0]])
    ratio_hist = np.concatenate([ratio_hist, [np.max(ratio_hist)]])
    ratio_hist = np.clip(ratio_hist, 0.0, 20.0)

    print("Training sample weights histogram is:")
    print(ratio_hist)
    print(bins)

    return ratio_hist, bins

def get_reweight_scalefactors(train_sample_weight_var_data, target_bins, target_hist):
    train_hist, train_bins = np.histogram(train_sample_weight_var_data, bins=target_bins)
    train_hist = train_hist.astype(float)
    # Fairly sure this is the wrong thing to do... the normalisation of each histogram before
    # taking the ratio should be 1 / sum(counts) not 1 / sum(rates).
    # for i in range(len(train_hist)):
    #     train_hist[i] /= (train_bins[i + 1] - train_bins[i])
    train_hist /= np.sum(train_hist)
    ratio_hist = target_hist / train_hist

    # Dont really care about <0.5GeV and >6GeV
    ratio_hist[-2:] = 1.0
    ratio_hist[0] = 1.0

    print("Training sample weights histogram is:")
    print(ratio_hist)
    print(train_bins)

    return ratio_hist, train_bins

def get_resample_data(args):
    if args.uniform_resampling_Ev:
        resample_data = (np.array([0.5, 6.0]), np.array([1.0]), "Ev", 0.5, 6.0)
    elif args.uniform_resampling_fd_numu_nu_E:
        resample_data = (np.array([0.5, 6.0]), np.array([1.0]), "fd_numu_nu_E", 0.5, 6.0)
    elif args.resampling_ndcaf_Ev:
        bins = np.load("data/ndcafs_all_oa_trueE/allCAF_Ev_oaall_bins.npy")
        hist = np.load("data/ndcafs_all_oa_trueE/allCAF_Ev_oaall_hist.npy") # expect bin counts not rate
        # hist = hist[(bins >= 0.5) & (bins <= 6.0)]
        # bins = bins[(bins >= 0.5) & (bins <= 6.0)]
        hist /= np.sum(hist)
        resample_data = (bins, hist, "Ev", 0.0, 120.0)
    elif args.resampling_osc_Ev:
        bins = np.load("data/prism_nufit_target_fd_flux_norate/FDTargetFlux_bins.npy")
        hist = np.load("data/prism_nufit_target_fd_flux_norate/FDTargetFlux_hist.npy") # expect bin counts not rate
        # hist = hist[(bins >= 0.5) & (bins <= 6.0)]
        # bins = bins[(bins >= 0.5) & (bins <= 6.0)]
        hist /= np.sum(hist)
        resample_data = (bins, hist, "Ev", 0.0, 120.0)
    else:
        resample_data = None

    return resample_data

""" End - training reweight/resampler helpers """
