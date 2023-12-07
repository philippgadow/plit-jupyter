import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from hist import Hist

from ftag.hdf5 import H5Reader


# settings in script (hard-coded for the moment)
fname = "/nfs/dust/atlas/user/pgadow/plit/data/ntuples/user.pgadow.LD_2023_11_28.601589.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadron.e8547_s3797_r13144_p5934_TREE/*.h5"
fname = "/nfs/dust/atlas/user/pgadow/plit/data/ntuples/user.pgadow.LD_2023_11_28.601589.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadron.e8547_s3797_r13144_p5934_TREE/user.pgadow.35693775._000477.output.h5"

# you can inspect the content of h5 files with
# > h5ls -v /nfs/dust/atlas/user/pgadow/plit/data/ntuples/user.pgadow.LD_2023_11_28.601589.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadron.e8547_s3797_r13144_p5934_TREE/user.pgadow.35693775._000477.output.h5

vars_lepton = ["topoetcone30Rel", "nTracksTrackjet"]
vars_muon = vars_lepton + ["pt_track", "eta_track", "phi_track", "ptfrac_track", "ptrel_track", "dRtrackjet_track", "ptvarcone30TTVARel", "caloClusterERel", "muonType"]
vars_electron = vars_lepton + ["pt", "eta", "phi", "ptfrac_lepton", "ptrel_lepton", "dRtrackjet_lepton", "ptvarcone30Rel", "caloClusterSumEtRel", "hasTrack"]

plotconfigs_lepton = {
    "pt": {
        "bins": [60, 0, 200_000],
        "logy": True
    },
    "eta": {
        "bins": [60, -2.5, 2.5],
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
    "muonType": {
        "bins": [6, -0.5, 5.5],
        "logy": False
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
        "bins": [14, -0.5, 14.5],
        "logy": False
    },
}

vars_track = ["pt", "eta", "phi", "ptfrac", "dr_trackjet", "dr_lepton", "btagIp_d0", "btagIp_z0SinTheta", "btagIp_d0_significance"]
plotconfigs_track = {
    "pt": {
        "bins": [60, 0, 10_000],
        "logy": True
    },
    "eta": {
        "bins": [60, -2.5, 2.5],
        "logy": True
    },
    "phi": {
        "bins": [60, -3.2, 3.2],
        "logy": False
    },
    "ptfrac": {
        "bins": [60, 0, 3.0],
        "logy": True
    },
    "dr_trackjet": {
        "bins": [60, 0, 0.4],
        "logy": True
    },
    "dr_lepton": {
        "bins": [60, 0, 0.4],
        "logy": True
    },
    "btagIp_d0": {
        "bins": [60, -4.0, 4.0],
        "logy": True
    },
    "btagIp_z0SinTheta": {
        "bins": [60, -6.0, 6.0],
        "logy": True
    },
    "btagIp_d0_significance": {
        "bins": [60, 0, 20.0],
        "logy": False
    },
}

def plot(df, var, plotconfig, obj):
    # style + disable warnings
    hep.style.use(hep.style.ROOT)
    import logging
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    
    if obj == 'muons':
        df_prompt = df[(df["iffClass"] == 4)]
        df_nonprompt = df[(df["iffClass"] != 4)]
        text_prompt = 'Prompt muons'
        text_nonprompt = 'Non-prompt muons'
    # there was the need to update labels due to moving to a newer analysis release due to
    # https://gitlab.cern.ch/atlas/athena/-/commit/f2833421bece919d44cbe6527e278fdf6ab6a21f
    elif obj == 'muon_tracks':
        df_prompt = df[(df["ftagTruthTypeLabel"] == 6) | (df["ftagTruthTypeLabel"] == -6)]
        df_nonprompt = df[~(df["ftagTruthTypeLabel"] == 6) | (df["ftagTruthTypeLabel"] == -6)]
        text_prompt = 'Muon tracks'
        text_nonprompt = 'Other tracks'
    elif obj == 'electrons':
        df_prompt = df[(df["iffClass"] == 2)]
        df_nonprompt = df[(df["iffClass"] != 2)]
        text_prompt = 'Prompt electrons'
        text_nonprompt = 'Non-prompt electrons'
    # there was the need to update labels due to moving to a newer analysis release due to
    # https://gitlab.cern.ch/atlas/athena/-/commit/f2833421bece919d44cbe6527e278fdf6ab6a21f
    elif obj == 'electron_tracks':
        df_prompt = df[(df["ftagTruthTypeLabel"] == 5) | (df["ftagTruthTypeLabel"] == -5)]
        df_nonprompt = df[~(df["ftagTruthTypeLabel"] == 5) | (df["ftagTruthTypeLabel"] == -5)]
        text_prompt = 'Electron tracks'
        text_nonprompt = 'Other tracks'

    bin_settings = plotconfig['bins']
    logy = plotconfig['logy']

    f, ax = plt.subplots(figsize=(14, 7))
    hsig, bins = np.histogram(df_prompt[(df_prompt[var] > -99) & (df_prompt[var] > bin_settings[1]) & (df_prompt[var] < bin_settings[2])][var], bins=bin_settings[0], density=1)
    hbkg, bins = np.histogram(df_nonprompt[(df_nonprompt[var] > -99) & (df_nonprompt[var] > bin_settings[1]) & (df_nonprompt[var] < bin_settings[2])][var], bins=bins, density=1)
    hep.histplot((hsig, bins), label=text_prompt, ax=ax)
    hep.histplot(hbkg, bins=bins, label=text_nonprompt, ax=ax)
    plt.xlabel(var)
    plt.ylabel('Normalised entries')
    if logy: ax.set_yscale("log")
    hep.atlas.label(loc=4, label="Internal", ax=ax)
    plt.legend(loc="best")
    f.savefig(f'plot_{obj}_{var}.png') 
    plt.close(f)

def main():
    do_muons = True
    do_electrons = False

    if do_muons:
        reader_muons = H5Reader(fname, batch_size=100_000, jets_name="muons")
        data_muons = reader_muons.load({"muons": None, "muon_tracks": None}, num_jets=1_000_000)
        print(len(data_muons["muons"]))

        muons = data_muons["muons"]
        df_muons = pd.DataFrame(muons)

        tracks_muons = data_muons["muon_tracks"]
        tracks_muons = tracks_muons.flatten()
        import numpy.lib.recfunctions as rfn
        iffClass_repeated = df_muons["iffClass"].apply(lambda x: np.repeat(x, 40)).to_numpy().flatten()
        iffClass_repeated = iffClass_repeated.astype(tracks_muons.dtype)
        tracks_muons = rfn.append_fields(tracks_muons, 'iffClass', iffClass_repeated)
        tracks_muons = tracks_muons[np.where(tracks_muons["valid"])]
        df_tracks_muons = pd.DataFrame(tracks_muons)

        print(df_tracks_muons)
        return
        # plot muon variables
        for var in vars_muon:
            plot(df_muons, var, plotconfigs_lepton[var], 'muons')

        # plot muon track variables
        for var in vars_track:
            plot(df_tracks_muons, var, plotconfigs_track[var], 'muon_tracks')

    if do_electrons:
        reader_electrons = H5Reader(fname, batch_size=100_000, jets_name="electrons")
        data_electrons = reader_electrons.load({"electrons": None, "electron_tracks": None}, num_jets=1_000_000)
        print(len(data_electrons["electrons"]))

        electrons = data_electrons["electrons"]
        df_electrons = pd.DataFrame(electrons)

        tracks_electrons = data_electrons["electron_tracks"]
        tracks_electrons = tracks_electrons.flatten()
        tracks_electrons = tracks_electrons[np.where(tracks_electrons["valid"])]
        df_tracks_electrons = pd.DataFrame(tracks_electrons)

        # plot electron variables
        for var in vars_electron:
            plot(df_electrons, var, plotconfigs_lepton[var], 'electrons')

        # plot electron track variables
        for var in vars_track:
            plot(df_tracks_electrons, var, plotconfigs_track[var], 'electron_tracks')

if __name__ == "__main__":
    main()
