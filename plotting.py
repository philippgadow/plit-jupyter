import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns
import hist
from hist import Hist

from ftag.hdf5 import H5Reader


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
    elif obj == 'muon_tracks':
        df_prompt = df[(df["ftagTruthTypeLabel"] == 5) | (df["ftagTruthTypeLabel"] == -5)]
        df_nonprompt = df[~(df["ftagTruthTypeLabel"] == 5) | (df["ftagTruthTypeLabel"] == -5)]
        text_prompt = 'Muon tracks'
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



fname = "/nfs/dust/atlas/user/pgadow/plit/data/ntuples/user.pgadow.LD_2023_11_28.601589.PhPy8EG_A14_ttbar_hdamp258p75_nonallhadron.e8547_s3797_r13144_p5934_TREE//*.h5"

vars_muon = ["pt_track", "eta_track", "phi_track", "ptvarcone30TTVARel", "topoetcone30Rel", "ptfrac_track", "ptrel_track", "dRtrackjet_track", "nTracksTrackjet"]
plotconfigs_muon = {
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
    "ptvarcone30TTVARel": {
        "bins": [60, 0, 1.5],
        "logy": True
    },
    "topoetcone30Rel": {
        "bins": [60, 0, 2.0],
        "logy": True
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
    "nTracksTrackjet": {
        "bins": [15, 0, 15],
        "logy": False
    },
}

vars_track = ["pt", "eta", "phi", "ptfrac", "dr_trackjet", "dr_muon", "btagIp_d0", "btagIp_z0SinTheta", "btagIp_d0_significance"]
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
    "dr_muon": {
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

reader = H5Reader(fname, batch_size=100_000)
data = reader.load({"muons": None, "muon_tracks": None}, num_jets=1_000_000)
print(len(data["muons"]))

leptons = data["muons"]
df_leptons = pd.DataFrame(leptons)

tracks = data["muon_tracks"]
tracks = tracks.flatten()
tracks = tracks[np.where(tracks["valid"])]
df_tracks = pd.DataFrame(tracks)

# plot muon variables
for var in vars_muon:
    plot(df_leptons, var, plotconfigs_muon[var], 'muons')

# plot track variables
for var in vars_track:
    plot(df_tracks, var, plotconfigs_track[var], 'muon_tracks')
