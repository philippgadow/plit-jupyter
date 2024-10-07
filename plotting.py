import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from hist import Hist

from ftag.hdf5 import H5Reader


# settings in script (hard-coded for the moment)
fname = "/nfs/dust/atlas/user/pgadow/plit/data/ntuples/user.pgadow.LD_2024_04_06.601589.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadron.e8547_s3797_r13144_p6119_TREE/user.pgadow.*.output.h5"

# you can inspect the content of h5 files with
# > h5ls -v /nfs/dust/atlas/user/pgadow/plit/data/ntuples/user.pgadow.LD_2023_11_28.601589.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadron.e8547_s3797_r13144_p5934_TREE/user.pgadow.35693775._000477.output.h5

vars_electron = [
    "qd0", "d0sig", "z0sinTheta",
    # "pt", "eta", "absEta", "phi",
    # "ptfrac_lepton", "ptrel_lepton",
    # "dRtrackjet_lepton", "nTracksTrackjet", 
    # "ptvarcone30Rel", "topoetcone30Rel",
    # "d0sig", "z0sinTheta",
    # "SCTWeightedCharge", "qd0",
    # "caloClusterSumEtRel",
    # "ntracks", "hasTrack"
]

plotconfigs_lepton = {
    "pt": {
        "bins": [60, 0, 200_000],
        "logy": True
    },
    "eta": {
        "bins": [60, -2.5, 2.5],
        "logy": False
    },
    "absEta": {
        "bins": [30, 0., 2.5],
        "logy": False
    },
    "phi": {
        "bins": [60, -3.2, 3.2],
        "logy": False
    },
    "pt_track": {
        "bins": [60, 0, 200_000],
        "logy": True
    },
    "eta_track": {
        "bins": [60, -2.5, 2.5],
        "logy": False
    },
    "absEta_track": {
        "bins": [30, 0., 2.5],
        "logy": False
    },
    "phi_track": {
        "bins": [60, -3.2, 3.2],
        "logy": False
    },
    "ptvarcone30Rel": {
        "bins": [60, 0, 1.5],
        "logy": True
    },
    "ptvarcone30TTVARel": {
        "bins": [60, 0, 1.5],
        "logy": True
    },
    "topoetcone30Rel": {
        "bins": [60, 0, 2.0],
        "logy": True
    },
    "caloClusterERel": {
        "bins": [60, 0, 3.0],
        "logy": True
    },
    "caloClusterSumEtRel": {
        "bins": [60, 0, 3.0],
        "logy": True
    },
    "hasTrack": {
        "bins": [3, -0.5, 2.5],
        "logy": False
    },
    "ptfrac_track": {
        "bins": [60, 0, 3.0],
        "logy": False
    },
    "ptrel_track": {
        "bins": [60, 0, 10_000],
        "logy": True
    },
    "dRtrackjet_track": {
        "bins": [60, 0, 0.4],
        "logy": True
    },
    "ptfrac_lepton": {
        "bins": [60, 0, 3.0],
        "logy": False
    },
    "ptrel_lepton": {
        "bins": [60, 0, 10_000],
        "logy": True
    },
    "dRtrackjet_lepton": {
        "bins": [60, 0, 0.4],
        "logy": True
    },
    "nTracksTrackjet": {
        "bins": [15, -0.5, 15.5],
        "logy": False
    },
    "d0sig": {
        "bins": [20, 0.0, 20.0],
        "logy": False
    },
    "z0sinTheta": {
        "bins": [16, -80., 80.],
        "logy": False
    },
    "SCTWeightedCharge": {
        "bins": [20, 0.0, 5.],
        "logy": False
    },
    "qd0": {
        "bins": [30, -1., 0.5],
        "logy": False
    },
    "ntracks": {
        "bins": [10, -.5, 10.5],
        "logy": False
    },
}

# vars_track = ["pt", "eta", "phi", "ptfrac", "dr_trackjet", "dr_lepton", "btagIp_d0", "btagIp_z0SinTheta", "btagIp_d0_significance"]
# plotconfigs_track = {
#     "pt": {
#         "bins": [60, 0, 10_000],
#         "logy": True
#     },
#     "eta": {
#         "bins": [60, -2.5, 2.5],
#         "logy": True
#     },
#     "phi": {
#         "bins": [60, -3.2, 3.2],
#         "logy": False
#     },
#     "ptfrac": {
#         "bins": [60, 0, 3.0],
#         "logy": True
#     },
#     "dr_trackjet": {
#         "bins": [60, 0, 0.4],
#         "logy": True
#     },
#     "dr_lepton": {
#         "bins": [60, 0, 0.4],
#         "logy": True
#     },
#     "btagIp_d0": {
#         "bins": [60, -4.0, 4.0],
#         "logy": True
#     },
#     "btagIp_z0SinTheta": {
#         "bins": [60, -6.0, 6.0],
#         "logy": True
#     },
#     "btagIp_d0_significance": {
#         "bins": [60, 0, 20.0],
#         "logy": False
#     },
# }

def plot(df, var, plotconfig, obj):
    # style + disable warnings
    hep.style.use(hep.style.ROOT)
    import logging
    logging.getLogger('matplotlib').setLevel(logging.ERROR)


    df_prompt = df[(df["iffClass"] == 2) | (df["iffClass"] == 3)]
    df_photonconv = df[(df["iffClass"] == 5)]
    df_nonprompt = df[(df["iffClass"] != 2) & (df["iffClass"] != 3) & (df["iffClass"] != 5)]
    text_prompt = 'Prompt electrons'
    text_photonconv = 'Photon conversion electrons'
    text_nonprompt = 'Non-prompt electrons'
    
    bin_settings = plotconfig['bins']
    logy = plotconfig['logy']

    f, ax = plt.subplots(figsize=(14, 7))
    hprompt, bins = np.histogram(df_prompt[(df_prompt[var] > -99) & (df_prompt[var] > bin_settings[1]) & (df_prompt[var] < bin_settings[2])][var], bins=bin_settings[0], density=1)
    hnonprompt, bins = np.histogram(df_nonprompt[(df_nonprompt[var] > -99) & (df_nonprompt[var] > bin_settings[1]) & (df_nonprompt[var] < bin_settings[2])][var], bins=bins, density=1)
    hphotonconv, bins = np.histogram(df_photonconv[(df_photonconv[var] > -99) & (df_photonconv[var] > bin_settings[1]) & (df_photonconv[var] < bin_settings[2])][var], bins=bins, density=1)

    hep.histplot((hprompt, bins), label=text_prompt, ax=ax)
    hep.histplot(hnonprompt, bins=bins, label=text_nonprompt, ax=ax)
    hep.histplot(hphotonconv, bins=bins, label=text_photonconv, ax=ax)
    plt.xlabel(var)
    plt.ylabel('Normalised entries')
    if logy: ax.set_yscale("log")
    hep.atlas.label(loc=4, label="Internal", ax=ax)
    plt.legend(loc="best")
    f.savefig(f'plot_{obj}_{var}.png') 
    plt.close(f)

def main():
    
    reader_electrons = H5Reader(fname, batch_size=100_000, jets_name="electrons")
    data_electrons = reader_electrons.load({"electrons": None, "electron_tracks": None}, num_jets=1_000_000)
    print(len(data_electrons["electrons"]))

    electrons = data_electrons["electrons"]
    df_electrons = pd.DataFrame(electrons)

    # tracks_electrons = data_electrons["electron_tracks"]
    # tracks_electrons = tracks_electrons.flatten()
    # tracks_electrons = tracks_electrons[np.where(tracks_electrons["valid"])]
    # df_tracks_electrons = pd.DataFrame(tracks_electrons)

    # plot electron variables
    for var in vars_electron:
        plot(df_electrons, var, plotconfigs_lepton[var], 'electrons')

    # # plot electron track variables
    # for var in vars_track:
    #     plot(df_tracks_electrons, var, plotconfigs_track[var], 'electron_tracks')

if __name__ == "__main__":
    main()
