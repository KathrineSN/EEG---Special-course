# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:26:15 2020

@author: kathr
"""

import os
import mne
import numpy as np
from matplotlib import pyplot as plt
path="C:\\Users\\kathr\\OneDrive\\Documents\\EEG Specialkursus"
os.chdir(path)

####################
# Loading the data #
####################

mne.set_log_level("WARNING")
raw1 = mne.io.read_raw_bdf("sj0013a_unshared.bdf", preload=True)
raw2 = mne.io.read_raw_bdf("sj0013ab_shared.bdf", preload=True)
raw3 = mne.io.read_raw_bdf("sj0013b_unshared.bdf", preload=True)

raw1.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)

rawa = mne.concatenate_raws([raw1,raw2])
rawb = mne.concatenate_raws([raw2,raw3])
print(rawa.info)

######################
# Filtering the data #
######################

#Applying high pass filter at 0.1 Hz and low pass filter at 40 Hz 

f_rawa = rawa.filter(l_freq=0.1, h_freq=40, picks="eeg") 
f_rawb = rawb.filter(l_freq=0.1, h_freq=40, picks="eeg")

f_rawa.plot(n_channels=32, scalings={"eeg": 600e-8}, start=100, block = True)
f_rawa.plot(n_channels=32, scalings={"eeg": 600e-8}, start=100, block = True)

mne.viz.plot_raw(f_rawa, n_channels = 100)


print(rawa.info.ch_names)

#####################
# Epoching the data #
#####################

eventsa = mne.find_events(f_rawa, stim_channel = 'Status')
eventsb = mne.find_events(f_rawb, stim_channel = 'Status')
print(mne.find_events(f_rawa))
print('Number of events:', len(eventsa))
print('Unique event codes:', np.unique(eventsa[:, 2]))

events_231 = eventsa[0:600,:]
events_224_a = eventsa[601:1257,:]
events_225_a = eventsa[1257:1901,:]

events_233 = eventsb[0:656,:]
events_224_b = eventsb[657:1301,:]
events_225_b = eventsb[1302:1901,:]

## Epochs for person A
# Epochs for A unshared condition 

event_dict = {'Angry1': 40, 'Angry2': 41, 'Happy1': 50,'Happy2': 51, 'Neutral1': 60, 'Neutral2':61 }

epochs_231 = mne.Epochs(f_rawa, events_231, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_231.plot(n_epochs=10)

#Epochs for AB shared with feedback

epochs_224_a = mne.Epochs(f_rawa, events_224_a, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_224_a.plot(n_epochs=10)

#Epochs for AB shared without feedback

epochs_225_a = mne.Epochs(f_rawa, events_225_a, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_225_a.plot(n_epochs=10)

## Epochs for person B

epochs_233 = mne.Epochs(f_rawa, events_233, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_233.plot(n_epochs=10)

#Epochs for AB shared with feedback

epochs_224_b = mne.Epochs(f_rawa, events_224_b, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_224_b.plot(n_epochs=10)

#Epochs for AB shared without feedback

epochs_225_b = mne.Epochs(f_rawa, events_225_b, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_225_b.plot(n_epochs=10)

#########################
# Downsampling the data #
#########################

#Downsampling person A
epochs_231_resampled = epochs_231.copy().resample(256, npad = 'auto')
epochs_224a_resampled = epochs_224_a.copy().resample(256, npad = 'auto')
epochs_225a_resampled = epochs_225_a.copy().resample(256, npad = 'auto')

#Downsampling person B
epochs_233_resampled = epochs_233.copy().resample(256, npad = 'auto')
epochs_224b_resampled = epochs_224_b.copy().resample(256, npad = 'auto')
epochs_225b_resampled = epochs_225_b.copy().resample(256, npad = 'auto')

#Plotting example of downsampling
plt.figure(figsize=(7, 3))
n_samples_to_plot = int(0.5 * epochs_231.info['sfreq'])  # plot 0.5 seconds of data
plt.plot(epochs_231.times[:n_samples_to_plot],
         epochs_231.get_data()[0, 0, :n_samples_to_plot], color='black')

n_samples_to_plot = int(0.5 * epochs_231_resampled.info['sfreq'])
plt.plot(epochs_231_resampled.times[:n_samples_to_plot],
         epochs_231_resampled.get_data()[0, 0, :n_samples_to_plot],
         '-o', color='red')

plt.xlabel('time (s)')
plt.legend(['original', 'downsampled'], loc='best')
plt.title('Effect of downsampling')
mne.viz.tight_layout()

############################
# Identifying bad channels #
############################

#Power spectral density
picks_a = []
picks_b = []
channels = epochs_231_resampled.info.ch_names

for i in range(len(channels)):
    if channels[i].startswith('1-A') or channels[i].startswith('1-B'):
        picks_a.append(channels[i])

print(picks_a)

for i in range(len(channels)):
    if channels[i].startswith('2-A') or channels[i].startswith('2-B'):
        picks_b.append(channels[i])

print(picks_b)

## Person A ##
# Three conditions for angry faces 
epochs_231_resampled['Angry1','Angry2'].plot_psd(picks = picks_a, color = 'red')
plt.legend()

epochs_224a_resampled['Angry1','Angry2'].plot_psd(picks = picks_a)
epochs_225a_resampled['Angry1','Angry2'].plot_psd(picks = picks_a)

# Three conditions for happy faces 
epochs_231_resampled['Happy1','Happy2'].plot_psd(picks = picks_a)
epochs_224a_resampled['Happy1','Happy2'].plot_psd(picks = picks_a)
epochs_225a_resampled['Happy1','Happy2'].plot_psd(picks = picks_a)

# Three conditions for neutral faces
epochs_231_resampled['Neutral1','Neutral2'].plot_psd(picks = picks_a)
epochs_224a_resampled['Neutral1','Neutral2'].plot_psd(picks = picks_a)
epochs_225a_resampled['Neutral1','Neutral2'].plot_psd(picks = picks_a)


print(epochs_231_resampled.info.ch_names)

d_231 = epochs_231_resampled['Angry1','Angry2'].getdata()



#Plotting one condition
epochs_231_resampled['Angry1','Angry2'].plot(n_epochs = 10)

#Brug mne .getdata til at finde channel statistics

from autoreject import Ransac
from autoreject.utils import interpolate_bads

print(epochs_231_resampled.info['chs'])

print(epochs_231_resampled.info)

f_rawa.info['bads'] = []
f_rawa.info['projs'] = list()
picks = mne.pick_types(epochs_231_resampled.info, eeg=True,
                       stim=False, eog=False,
                       include=[], exclude=[])
ransac = Ransac(verbose='progressbar', picks=picks_a, n_jobs=1)
epochs_clean = ransac.fit_transform(epochs_231_resampled)






# Testing things





