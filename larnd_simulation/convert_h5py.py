import numpy as np
import h5py 
import numpy as np
from dataclasses import dataclass
from glob import glob
import os
import uproot
import numpy as np
from tqdm.contrib.concurrent import process_map
from LarpixParser import event_parser as EvtParser
from LarpixParser import hit_parser as HitParser
import LarpixParser.util as util

def files(dataset_folder):
    dataset_files = glob(dataset_folder + '/*.h5')
    print(f'Found {len(dataset_files)} files in {dataset_folder}')
    return dataset_files


def get_run_number(filepath):
    run_number = filepath.split('/')[-1].split('.')[1]
    # check if the string is a number 
    assert run_number.isdigit() and len(run_number) == 7, "Run number must be 7 digits long"
    return int(run_number)


def get_event_number(filepath):
    event_number = filepath.split('/')[-1].split('.')[2]
    # check if the string is a number 
    assert event_number.isdigit() and len(event_number) == 7, "Event number must be 7 digits long"
    return int(event_number)
    

def paired_info(filepath):
    if not os.path.exists(filepath):
        print("File not found locally, downloading from EOS")

        DEFAULT_FILE = "https://cernbox.cern.ch/s/3FFybAq3wsl3Irb/download"
        os.system(f"wget -O {filepath} {DEFAULT_FILE}")

    flavour = ['numu_score', 'nue_score', 'nutau_score', 'nc_score']
    proton = ['proton0_score', 'proton1_score', 'proton2_score', 'protonN_score']
    pion = ['pion0_score', 'pion1_score', 'pion2_score', 'pionN_score']
    pionzero = ['pionzero0_score', 'pionzero1_score', 'pionzero2_score', 'pionzeroN_score']

    numu_energy_cols = ['numu_nu_E', 'numu_lep_E', 'numu_had_E']
    nue_energy_cols = ['nue_nu_E', 'nue_lep_E', 'nue_had_E']
    nc_energy_cols = ['nc_nu_E', 'nc_lep_E', 'nc_had_E']
    identifiers = ['run', 'eventID']
    vertex = ['vtxX', 'vtxY', 'vtxZ']
    with uproot.open(filepath) as file:
        tree = file['nd_fd_reco']
        array = tree.arrays(flavour + proton + pion + pionzero + numu_energy_cols + nue_energy_cols + nc_energy_cols + identifiers + vertex)
        return array


def test_vertices(paired_ev, vtx, i_ev):
    vtx_paired = np.hstack([paired_ev['vtxX'][0], paired_ev['vtxY'][0], paired_ev['vtxZ'][0]]) #there is only 1 vertex but it's stored as an array
    vtx_larnd = np.hstack([vtx['x_vert'][i_ev] / 10.0, vtx['y_vert'][i_ev] / 10.0, vtx['z_vert'][i_ev] / 10.0])
    assert np.allclose(vtx_paired, vtx_larnd), "Vertices don't match"

def get_filename(run_number):
    data_path = '/global/cfs/cdirs/dune/users/rradev/PRISM/N2FD'
    return f'{data_path}/FHC.{run_number}.larndsim.h5'



def plot_one_file():
    import matplotlib.pyplot as plt


    run_id = unique_run_ids[0]
    filename = get_filename(run_id)
    print(filename)

    run_mask = paired['run'] == run_id
    run_eventIDs = eventIDs[run_mask]
    paired_run = paired[run_mask]

    print(f'Found {len(run_eventIDs)} events in run {run_id}')
    switch_xz = False
    detector = "ndlar"
    with h5py.File(filename, 'r') as f:
        packets = f['packets']
        segs = f['tracks'] # Geant4 truth
        assn = f['mc_packets_assn'] # G4-readout association
        vtx = f['vertices'] # true interaction information

        
        run_config, geom_dict = util.detector_configuration(detector)
        pckt_event_ids = EvtParser.packet_to_eventid(assn, segs)
        t0s = EvtParser.get_t0(packets, run_config)


        # start looping over events
        event_ids = EvtParser.get_eventid(vtx)

        for i_ev in range(len(event_ids)):
    
            t0 = t0s[i_ev]
            # use eventIDs from the paired file
            event_idx = event_ids[i_ev]
            pckt_mask = pckt_event_ids == event_idx
            packets_ev = packets[pckt_mask]

            x,y,z,dQ = HitParser.hit_parser_charge(t0, packets_ev, geom_dict, run_config, switch_xz)

            x = np.array(x) / 10.
            y = np.array(y) / 10.
            z = np.array(z) / 10.
            
            # get more info from the paired file
            paired_ev = paired_run[paired_run['eventID'] == event_idx]

            # check that vertices match
            test_vertices(paired_ev, vtx, i_ev)

            # histogram the dQ
            plt.hist(dQ, bins=100)
            plt.savefig(f'plots/{run_id}_{event_idx}_hist.png')
            plt.close()
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, s=2, marker='s')
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')
            ax.set_zlabel('z [cm]')
            plt.savefig(f'plots/{run_id}_{event_idx}_3d.png')
            plt.close()

def process_run(run_id):
    filename = get_filename(run_id)
    print(filename)

    run_mask = paired['run'] == run_id
    run_eventIDs = eventIDs[run_mask]
    paired_run = paired[run_mask]

    print(f'Found {len(run_eventIDs)} events in run {run_id}')
    switch_xz = False
    detector = "ndlar"
    with h5py.File(filename, 'r') as f:
        packets = f['packets']
        segs = f['tracks'] # Geant4 truth
        assn = f['mc_packets_assn'] # G4-readout association
        vtx = f['vertices'] # true interaction information

        
        run_config, geom_dict = util.detector_configuration(detector)
        pckt_event_ids = EvtParser.packet_to_eventid(assn, segs)
        t0s = EvtParser.get_t0(packets, run_config)


        # start looping over events
        event_ids = EvtParser.get_eventid(vtx)

        for i_ev in range(len(event_ids)):
            
            if i_ev > len(t0s) - 1:
                print(f'More events in edep-sim file than in larndsim file, stopping at event {i_ev}')
                break
            t0 = t0s[i_ev]
            
            # use eventIDs from the paired file
            event_idx = event_ids[i_ev]
            pckt_mask = pckt_event_ids == event_idx
            packets_ev = packets[pckt_mask]

            x,y,z,dQ = HitParser.hit_parser_charge(t0, packets_ev, geom_dict, run_config, switch_xz)

            x = np.array(x) / 10.
            y = np.array(y) / 10.
            z = np.array(z) / 10.
            
            if np.max(z) > 10_000:
                print("Z values are too large, skipping file")
                print(filename)
                continue
            # get more info from the paired file
            paired_ev = paired_run[paired_run['eventID'] == event_idx]

            # check that vertices match
            test_vertices(paired_ev, vtx, i_ev)

            coords = np.vstack([x,y,z]).T

            np.savez(f'/pscratch/sd/r/rradev/near_to_far/larndsim_npz_oct27/{run_id}_{event_idx}.npz', coords=coords, dQ=dQ, paired_data=paired_ev)



# load the paired file
paired = paired_info('/pscratch/sd/r/rradev/near_to_far/paired_ndfd.root')
unique_run_ids = np.unique(paired['run'])
eventIDs = paired['eventID']

print(paired.keys())
# process_map(process_run, unique_run_ids, max_workers=128)

# plot_one_file()


