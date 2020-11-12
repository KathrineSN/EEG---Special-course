import os
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#path="C:\\Users\\kathr\\OneDrive\\Documents\\EEG Specialkursus"
path="C:\\Users\\kathr\\OneDrive\\Documents\\EEG Specialkursus"
os.chdir(path)

raw1 = mne.io.read_raw_bdf("sj0016a_unshared.bdf", preload=True)

raw1.plot(n_channels=6, scalings={"eeg": 800e-7}, start=100, block = True)


