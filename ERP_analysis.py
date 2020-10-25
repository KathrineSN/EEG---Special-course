# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:19:17 2020

@author: kathr
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#path="C:\\Users\\kathr\\OneDrive\\Documents\\EEG Specialkursus"
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\EEG---Special-course"
os.chdir(path)
#from preprocessing.py import *
from re_load import *


#########################################
#%% Analysis of N170 and EPN component ##
#########################################
print(new_ch_names)

# Condition 231
N170_picks = ['PO7','PO8','P7','P8'] 

evoked_231_N170 = epochs_231_resampled.average()
N170_idx1 = mne.pick_channels(evoked_231_N170.info['ch_names'],['PO7','PO8','P7','P8','P9','P10'] )
N170_dict = dict(N170 = N170_idx1)
evoked_231_N170_combined = mne.channels.combine_channels(evoked_231_N170, N170_dict, method = 'mean')
#evoked_231_N170.plot()
title = 'Average of PO7, PO8, P7, P8, P9, P10 (231)'
evoked_231_N170_combined.plot(titles = dict(eeg = title))

# Condition 224a
evoked_224a_N170 = epochs_224a_resampled.average()
N170_idx2 = mne.pick_channels(evoked_224a_N170.info['ch_names'],['PO7','PO8','P7','P8','P9','P10'] )
N170_dict2 = dict(N170 = N170_idx2)
evoked_224a_N170_combined = mne.channels.combine_channels(evoked_224a_N170, N170_dict2, method = 'mean')
#evoked_224a_N170.plot()
title2 = 'Average of PO7, PO8, P7, P8, P9, P10 (224)'
evoked_231_N170_combined.plot(titles = dict(eeg = title2))

# Condition 225a
evoked_225a_N170 = epochs_225a_resampled.average()
N170_idx3 = mne.pick_channels(evoked_225a_N170.info['ch_names'],['PO7','PO8','P7','P8','P9','P10'] )
N170_dict3 = dict(N170 = N170_idx3)
evoked_225a_N170_combined = mne.channels.combine_channels(evoked_225a_N170, N170_dict3, method = 'mean')
#evoked_225a_N170.plot()
title3 = 'Average of PO7, PO8, P7, P8, P9, P10 (225)'
evoked_225a_N170_combined.plot(titles = dict(eeg = title3))

# Angry faces
evoked_Angry_N170 = epochs_a_resampled['Angry1','Angry2'].average()
N170_idx4 = mne.pick_channels(evoked_Angry_N170.info['ch_names'],['PO7','PO8','P7','P8','P9','P10'] )
N170_dict4 = dict(N170 = N170_idx4)
evoked_Angry_N170_combined = mne.channels.combine_channels(evoked_Angry_N170, N170_dict4, method = 'mean')
#evoked_Angry_N170.plot()
title4 = 'Average of PO7, PO8, P7, P8, P9, P10 (40/41)'
evoked_Angry_N170_combined.plot(titles = dict(eeg = title4))


# Happy faces
evoked_Happy_N170 = epochs_a_resampled['Happy1','Happy2'].average()
N170_idx5 = mne.pick_channels(evoked_Happy_N170.info['ch_names'],['PO7','PO8','P7','P8','P9','P10'] )
N170_dict5 = dict(N170 = N170_idx5)
evoked_Happy_N170_combined = mne.channels.combine_channels(evoked_Happy_N170, N170_dict5, method = 'mean')
#evoked_Happy_N170.plot()
title5 = 'Average of PO7, PO8, P7, P8, P9, P10 (50/51)'
evoked_Happy_N170_combined.plot(titles = dict(eeg = title5))

# Neutral faces
evoked_Neutral_N170 = epochs_a_resampled['Neutral1','Neutral2'].average()
N170_idx6 = mne.pick_channels(evoked_Neutral_N170.info['ch_names'],['PO7','PO8','P7','P8','P9','P10'] )
N170_dict6 = dict(N170 = N170_idx6)
evoked_Neutral_N170_combined = mne.channels.combine_channels(evoked_Neutral_N170, N170_dict6, method = 'mean')
#evoked_Neutral_N170.plot()
title6 = 'Average of PO7, PO8, P7, P8, P9, P10 (60/61)'
evoked_Neutral_N170_combined.plot(titles = dict(eeg = title6))

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
evoked_Angry_EPN = epochs_a_resampled['Angry1','Angry2'].average(picks = EPN_picks)
evoked_Angry_EPN.plot()

# Happy faces
evoked_Happy_EPN = epochs_a_resampled['Happy1','Happy2'].average(picks = EPN_picks)
evoked_Happy_EPN.plot()

# Neutral faces
evoked_Neutral_EPN = epochs_a_resampled['Neutral1','Neutral2'].average(picks = EPN_picks)
evoked_Neutral_EPN.plot()

#################################
#%% Analysis of LPP component ##
#################################

LPP_picks = ['CP1', 'CP3', 'P3', 'CP2', 'CP4', 'P4']


evoked_231_LPP = epochs_231_resampled.average()
LPP_idx1 = mne.pick_channels(evoked_231_LPP.info['ch_names'],LPP_picks)
LPP_dict = dict(LPP = LPP_idx1)
evoked_231_LPP_combined = mne.channels.combine_channels(evoked_231_LPP, LPP_dict, method = 'mean')
#evoked_231_N170.plot()
title = 'Avg. of CP1, CP3, P3, CP2, CP4, P4 (231)'
evoked_231_LPP_combined.plot(titles = dict(eeg = title))

# Condition 224a
evoked_224a_LPP = epochs_224a_resampled.average()
LPP_idx2 = mne.pick_channels(evoked_224a_LPP.info['ch_names'],LPP_picks)
LPP_dict2 = dict(LPP = LPP_idx2)
evoked_224a_LPP_combined = mne.channels.combine_channels(evoked_224a_LPP, LPP_dict2, method = 'mean')
#evoked_231_N170.plot()
title = 'Avg. of CP1, CP3, P3, CP2, CP4, P4 (224)'
evoked_224a_LPP_combined.plot(titles = dict(eeg = title))

# Condition 225a
evoked_225a_LPP = epochs_225a_resampled.average()
LPP_idx3 = mne.pick_channels(evoked_224a_LPP.info['ch_names'],LPP_picks)
LPP_dict3 = dict(LPP = LPP_idx3)
evoked_225a_LPP_combined = mne.channels.combine_channels(evoked_225a_LPP, LPP_dict3, method = 'mean')
#evoked_231_N170.plot()
title = 'Avg. of CP1, CP3, P3, CP2, CP4, P4 (225)'
evoked_225a_LPP_combined.plot(titles = dict(eeg = title))

# Angry faces
evoked_Angry_LPP = epochs_a_resampled['Angry1','Angry2'].average()
LPP_idx4 = mne.pick_channels(evoked_Angry_LPP.info['ch_names'],LPP_picks)
LPP_dict4 = dict(LPP = LPP_idx4)
evoked_Angry_LPP_combined = mne.channels.combine_channels(evoked_Angry_LPP, LPP_dict4, method = 'mean')
#evoked_Angry_N170.plot()
title4 = 'Avg. of CP1, CP3, P3, CP2, CP4, P4 (40/41)'
evoked_Angry_LPP_combined.plot(titles = dict(eeg = title4))


# Happy faces
evoked_Happy_LPP = epochs_a_resampled['Happy1','Happy2'].average()
LPP_idx5 = mne.pick_channels(evoked_Happy_LPP.info['ch_names'], LPP_picks)
LPP_dict5 = dict(LPP = LPP_idx5)
evoked_Happy_LPP_combined = mne.channels.combine_channels(evoked_Happy_LPP, LPP_dict5, method = 'mean')
#evoked_Happy_N170.plot()
title5 = 'Average of CP1, CP3, P3, CP2, CP4, P4 (50/51)'
evoked_Happy_LPP_combined.plot(titles = dict(eeg = title5))

# Neutral faces
evoked_Neutral_LPP = epochs_a_resampled['Neutral1','Neutral2'].average()
LPP_idx6 = mne.pick_channels(evoked_Neutral_LPP.info['ch_names'],LPP_picks)
LPP_dict6 = dict(LPP = LPP_idx6)
evoked_Neutral_LPP_combined = mne.channels.combine_channels(evoked_Neutral_LPP, LPP_dict6, method = 'mean')
#evoked_Neutral_N170.plot()
title6 = 'Average of CP1, CP3, P3, CP2, CP4, P4 (60/61)'
evoked_Neutral_LPP_combined.plot(titles = dict(eeg = title6))

#################
# Grand Average #
#################

evoked_231_N170_combined





