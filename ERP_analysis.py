# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:19:17 2020

@author: kathr
"""

import os
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
path="C:\\Users\\kathr\\OneDrive\\Documents\\EEG Specialkursus"
os.chdir(path)
from preprocessing.py import *

#################################
#%% Analysis of N170 component ##
#################################
print(new_ch_names)

# Condition 231
N170_picks = ['PO7','PO8','P7','P8']

evoked_231_N170 = epochs_231_resampled.average(picks = N170_picks)
evoked_231_N170.plot()

# Condition 224a
evoked_224a_N170 = epochs_224a_resampled.average(picks = N170_picks)
evoked_224a_N170.plot()

# Condition 225a
evoked_225a_N170 = epochs_225a_resampled.average(picks = N170_picks)
evoked_225a_N170.plot()

# Angry faces
evoked_Angry_N170 = epochs_a_resampled['Angry1','Angry2'].average(picks = N170_picks)
evoked_Angry_N170.plot()

# Happy faces
evoked_Happy_N170 = epochs_a_resampled['Happy1','Happy2'].average(picks = N170_picks)
evoked_Happy_N170.plot()

# Neutral faces
evoked_Neutral_N170 = epochs_a_resampled['Neutral1','Neutral2'].average(picks = N170_picks)
evoked_Neutral_N170.plot()

#################################
#%% Analysis of EPN component ##
#################################

# Condition 231
EPN_picks = ['PO7','PO8','P7','P8','P9','P10']

evoked_231_EPN = epochs_231_resampled.average(picks = EPN_picks)
evoked_231_EPN.plot()

# Condition 224a
evoked_224a_EPN = epochs_224a_resampled.average(picks = EPN_picks)
evoked_224a_EPN.plot()

# Condition 225a
evoked_225a_EPN = epochs_225a_resampled.average(picks = EPN_picks)
evoked_225a_EPN.plot()

# Angry faces
evoked_Angry_EPN = epochs_a_resampled['Angry1','Angry2'].average(picks = N170_picks)
evoked_Angry_EPN.plot()

# Happy faces
evoked_Happy_EPN = epochs_a_resampled['Happy1','Happy2'].average(picks = N170_picks)
evoked_Happy_EPN.plot()

# Neutral faces
evoked_Neutral_EPN = epochs_a_resampled['Neutral1','Neutral2'].average(picks = N170_picks)
evoked_Neutral_EPN.plot()

#################################
#%% Analysis of LPP component ##
#################################

# Condition 231
LPP_picks = ['CP1', 'CP3', 'P3', 'CP2', 'CP4', 'P4']

evoked_231_LPP = epochs_231_resampled.average(picks = LPP_picks)
evoked_231_LPP.plot()

# Condition 224a
evoked_224a_LPP = epochs_224a_resampled.average(picks = LPP_picks)
evoked_224a_LPP.plot()

# Condition 225a
evoked_225a_LPP = epochs_225a_resampled.average(picks = LPP_picks)
evoked_225a_LPP.plot()

# Angry faces
evoked_Angry_LPP = epochs_a_resampled['Angry1','Angry2'].average(picks = LPP_picks)
evoked_Angry_LPP.plot()

# Happy faces
evoked_Happy_LPP = epochs_a_resampled['Happy1','Happy2'].average(picks = LPP_picks)
evoked_Happy_LPP.plot()

# Neutral faces
evoked_Neutral_LPP = epochs_a_resampled['Neutral1','Neutral2'].average(picks = LPP_picks)
evoked_Neutral_LPP.plot()


