# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:19:41 2020

@author: kathr
"""

import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\EEG---Special-course"
os.chdir(path)
from EEG_functions import *

#Saving the evoked object
save_evoked([13,14,15,16,17,18])

#Grand average plots
ga_224 = grand_average([13,14,15,16,17,18], '224', 'N170', plot = 1)
ga_225 = grand_average([13,14,15,16,17,18], '225', 'N170', plot = 1)
ga_231 = grand_average([13,14,15,16,17,18], '231', 'N170', plot = 1)

ga_Angry = grand_average([13,14,15,16,17,18], 'Angry', 'N170', plot = 1)
ga_Happy = grand_average([13,14,15,16,17,18], 'Happy', 'N170', plot = 1)
ga_Neutral = grand_average([13,14,15,16,17,18], 'Neutral', 'N170', plot = 1)


ga_224_lpp = grand_average([13,14,15,16,17,18], '224', 'LPP', plot = 1)
ga_225_lpp = grand_average([13,14,15,16,17,18], '225', 'LPP', plot = 1)
ga_231_lpp = grand_average([13,14,15,16,17,18], '231', 'LPP', plot = 1)

ga_Angry_lpp = grand_average([13,14,15,16,17,18], 'Angry', 'LPP', plot = 1)
ga_Happy_lpp = grand_average([13,14,15,16,17,18], 'Happy', 'LPP', plot = 1)
ga_Neutral_lpp = grand_average([13,14,15,16,17,18], 'Neutral', 'LPP', plot = 1)


mne.viz.plot_compare_evokeds(dict(shared_with_feedback = ga_224, shared_without_feedback = ga_225, unshared = ga_231), ci = True)
mne.viz.plot_compare_evokeds(dict(Angry = ga_Angry, Happy = ga_Happy, Neutral = ga_Neutral))
mne.viz.plot_compare_evokeds(dict(shared_with_feedback = ga_224_lpp, shared_without_feedback = ga_225_lpp, unshared = ga_231_lpp))
mne.viz.plot_compare_evokeds(dict(Angry = ga_Angry_lpp, Happy = ga_Happy_lpp, Neutral = ga_Neutral_lpp))

#Grand average with CI
grand_average_with_CI('Social', 'N170')
grand_average_with_CI('Emotional', 'N170')
grand_average_with_CI('Social', 'LPP')
grand_average_with_CI('Emotional', 'LPP')

#%% Create evoked for statistics    
df_N170 = create_dataframe([13,14,15,16,17,18], 'N170')
df_N170 = df_N170.replace({'Social condition': '233'}, '231')
df_N170.to_csv(r'C:\Users\kathr\OneDrive\Documents\GitHub\EEG---Special-course\N170data.csv', index = False)

df_EPN = create_dataframe([13,14,15,16,17,18], 'EPN')
df_EPN = df_EPN.replace({'Social condition': '233'}, '231')
df_EPN.to_csv(r'C:\Users\kathr\OneDrive\Documents\GitHub\EEG---Special-course\EPNdata.csv', index = False)

df_EarlyLPP = create_dataframe([13,14,15,16,17,18], 'Early LPP')
df_EarlyLPP = df_EarlyLPP.replace({'Social condition': '233'}, '231')
df_EarlyLPP.to_csv(r'C:\Users\kathr\OneDrive\Documents\GitHub\EEG---Special-course\EarlyLPPdata.csv', index = False)

df_LateLPP = create_dataframe([13,14,15,16,17,18], 'Late LPP')
df_LateLPP = df_LateLPP.replace({'Social condition': '233'}, '231')
df_LateLPP.to_csv(r'C:\Users\kathr\OneDrive\Documents\GitHub\EEG---Special-course\LateLPPdata.csv', index = False)



# pingouin as pg

#df = df.dropna()

#stat = df.rm_anova(dv='Avg. amplitude', within = ['Social condition','Emotional condition'],

#            subject='Subject',  detailed=False)
    




