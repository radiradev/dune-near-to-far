import numpy as np
import h5py
import os 
import pandas as pd
import re
import fire
from tqdm import tqdm


def extract_run_id(string):
    # Extracting the numerical ID using regular expression
    match = re.search(r'FHC\.(\d+)', string)
    
    if match:
        return match.group(1)
    else:
        raise ValueError(f'Could not extract ID from {string}')

def _convert_to_df(h5py_run):
    return pd.DataFrame(np.array(h5py_run))

def convert_to_df(file):
    
    df_list = []
    for key in tqdm(file.keys()):
        data = file[key]
        nd_df = _convert_to_df(data['nd_paramreco'])
        fd_df = _convert_to_df(data['fd_reco'])
        fd_vertices = _convert_to_df(data['fd_vertices'])

        run_id = extract_run_id(key)
        assert len(nd_df) == len(fd_df), f'Length mismatch for {key}'

        # joint the two dataframes
        # add fd prefix to the columns corresponding to FD vars
        fd_df.columns = ['fd_' + col for col in fd_df.columns]
        fd_vertices.columns = ['fd_' + col for col in fd_vertices.columns]

        df = pd.concat([nd_df, fd_df, fd_vertices], axis=1)
        df['run_id'] = run_id

        # create a unique ID for each event
        df['unique_id'] = df['run_id'].astype(str) + '_' + df['eventID'].astype(str)

        df_list.append(df)
    
    return pd.concat(df_list)

def drop_negative_scores(df):
    # drop rows with negative scores
    return df[df['fd_nc_score'] >= 0]


def apply_cuts(df):
    #muon_{contained| tracker | ecal} == 1 & reco_numu == 1 & Ehad_veto > 30
    cuts = ((df['muon_contained'] == 1) | (df['muon_tracker'] == 1)) & (df['reco_numu'] == 1) & (df['Ehad_veto'] < 30)
    return df[cuts]

def clip_leptonic(df):
    # we clip the leptonic energy as we can't map to 0 in the lognormal dist
    leptonic_cols = ['fd_nue_lep_E', 'fd_numu_lep_E']
    for col in leptonic_cols:
        df[col] = df[col].clip(1e-2, None)
    return df

def clip_hadronic(df):
    hadronic_cols = ['fd_nue_had_E', 'fd_numu_had_E']
    for col in hadronic_cols:
        df[col] = df[col].clip(1e-2, None)
    return df

def main(datadir: str):
    """
    Perform data processing and apply cuts to the paired data.

    Parameters:
    - datadir (str): The directory path where the data files are located.

    Usage:
    python3 cut.py <datadir>

    This script reads a HDF5 file named 'FHC.1000000-1005000.noFDhadsel_oldg4params.ndfd_reco_only.h5' located in the specified `datadir` directory.
    It converts the data from the HDF5 file into a pandas DataFrame using the `convert_to_df` function.
    Then, it applies cuts to the DataFrame using the `apply_cuts` function.
    Finally, it saves the resulting DataFrame as a CSV file named 'paired_data_cuts.csv' in the `datadir` directory.
    """

    filename = 'FHC.1000000-1005000.noFDhadsel_oldg4params.ndfd_reco_only.h5'
    file = h5py.File(os.path.join(datadir, filename), 'r')
    df = convert_to_df(file)
    df = apply_cuts(df)
    df = drop_negative_scores(df)
    df = clip_hadronic(df)
    df = clip_leptonic(df)


    df.to_csv(os.path.join(datadir, 'ndfd_reco_only_cuts.noFDhasel_oldg4params.csv'), index=False)
    print(f"Processed data saved to {os.path.join(datadir, 'paired_data_cuts.csv')}")
    print(f"Number of events after applying cuts: {len(df)}")

if __name__ == '__main__':
    fire.Fire(main)