# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:05:56 2023

@author: meryl_malezieux
"""

# load hr and detect heartbeats 


import os
import numpy as np
import pandas as pd 
from neo import io
from scipy import signal 
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter
# from hr_cleaning_GUI import EventEditor 
from scipy.interpolate import interp1d 
#%%
# load Data_to_Analyze.xlsx to to be able to load data
sheets = ['hr_only'] 
mouse_ids = np.empty(0) 
# cells = np.empty(0) 
data_folder = np.empty(0) 
ephy_file = np.empty(0) 
running= np.empty(0)
good_seconds_start = np.empty(0)
good_seconds_stop = np.empty(0) 
pupil_file = np.empty(0)
hr_file = np.empty(0) 
hr_thresh = np.empty(0) 
hr_refrac = np.empty(0)
hr_min_dur = np.empty(0)
highpass = np.empty(0) 
sex = np.empty(0) 
lick_thresh = np.empty(0) 
events_file = np.empty(0) 
hr_abs = np.empty(0) 
for i in np.arange(len(sheets)):
    data_list = pd.read_excel(r"\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Data_to_analyze\Data_to_analyze_siliconprobe.xlsx", 
                              sheet_name=sheets[i])
    mouse_ids = np.append(mouse_ids, data_list.loc[:, 'mouse ID'].values)
    # cells = np.append(cells, data_list.loc[:, 'cell #'].values) 
    data_folder = np.append(data_folder, data_list.loc[:, 'Raw Data Folder'].values)
    ephy_file = np.append(ephy_file, data_list.loc[:, 'kilosort path'].values)
    running = np.append(running, data_list.loc[:, 'running file'].values)
    good_seconds_start = np.append(good_seconds_start, data_list.loc[:, 'good data start (seconds)'].values)
    good_seconds_stop = np.append(good_seconds_stop, data_list.loc[:, 'good data end (seconds)'].values)
    pupil_file = np.append(pupil_file, data_list.loc[:, 'pupil file'].values)
    hr_file = np.append(hr_file, data_list.loc[:, 'hr file'].values)
    hr_thresh = np.append(hr_thresh, data_list.loc[:, 'hr_thresh'].values)
    hr_refrac = np.append(hr_refrac, data_list.loc[:, 'hr_refrac'].values)
    hr_min_dur = np.append(hr_min_dur, data_list.loc[:, 'hr_min_duration'].values)
    highpass = np.append(highpass, data_list.loc[:, 'highpass'].values) 
    sex = np.append(sex, data_list.loc[:, 'sex'].values)
    lick_thresh = np.append(lick_thresh, data_list.loc[:, 'lick_thresh'].values)
    events_file = np.append(events_file, data_list.loc[:, 'events file'].values)
    hr_abs = np.append(hr_abs, data_list.loc[:, 'hr_abs'].values)

#%%
## bunch of definitions

# definition for downsampling
def ds(ts, signal, ds_factor):
    signal_ds = np.mean(np.resize(signal,
                        (int(np.floor(signal.size/ds_factor)), ds_factor)), 1)
    ds_ts = ts[np.arange(int(np.round(ds_factor/2)), ts.size, ds_factor)]
    # trim off last time stamp if necessary
    ds_ts = ds_ts[0:signal_ds.size]
    return ds_ts, signal_ds

# load data 
def load_data_MIND(mouse_ids):

    i = mouse_ids

    # load some Axon data from ABF files
    file_name = os.path.join(data_folder[i], hr_file[i])
    # r is the name bound to the object created by io.AxonIO
    r = io.AxonIO(filename=file_name)
    # bl is the object that actually has the data, created by read_block
    bl = r.read_block(signal_group_mode='split-all')

    # get list of channel names
    channel_list = []
    for asig in bl.segments[0].analogsignals:
        channel_list.append(asig.name)
    
    full_ts = np.copy(bl.segments[0].analogsignals[0].times)  
        
    if isinstance(hr_file[i], str):      
        if 'IN2' in channel_list:
            ind = channel_list.index('IN2')
            full_ts = np.copy(bl.segments[0].analogsignals[0].times)
            hr = np.copy(bl.segments[0].analogsignals[ind].data)
            hr = hr[(full_ts >= good_seconds_start[i]) &
                    (full_ts < good_seconds_stop[i])]
            hr_ts = full_ts[(full_ts >= good_seconds_start[i]) &
                        (full_ts < good_seconds_stop[i])]
            hr = np.squeeze(hr)
        else:
            full_ts = np.copy(bl.segments[0].analogsignals[0].times)
            hr = np.copy(bl.segments[0].analogsignals[2].data)
            hr = hr[(full_ts >= good_seconds_start[i]) &
                              (full_ts < good_seconds_stop[i])]
            hr = np.squeeze(hr)
            hr_ts = full_ts[(full_ts >= good_seconds_start[i]) &
                            (full_ts < good_seconds_stop[i])]
        # downsample hr to 2000Hz 
        ds_factor = 10
        hr_ts, hr = ds(hr_ts, hr, ds_factor)
        
        #smoothing the data by applying filter
        hr = savgol_filter(hr, 21, 2)
    else:
        hr = np.empty(0)
        hr_ts = np.empty(0)

    return hr, hr_ts 


def find_inst_bpm(hr, sp_times, ts):
    inst_bpm = np.zeros(hr.size)
    for j in np.arange(inst_bpm.size):
        if j+int(ts.size/ts[-1]) < inst_bpm.size:
            #take how many spikes (beats) are in 200 timepoints
            inst_bpm[j] = sp_times[(sp_times > ts[j]) & (sp_times < ts[j+ 200])].size
            #multiply be 60 to have bpm
            inst_bpm[j] = inst_bpm[j] * 600 
        else: 
            #replace the last second where measurement is not possible with last value
            inst_bpm[j] = inst_bpm[j-1] 
    #filter the data further to have something smoothed. I have checked and it doesnt distort the signal
    inst_bpm = savgol_filter(inst_bpm, 1001, 2)
    #smooth the data with a rolling window of 1s because the data is super chunky        
    inst_bpm = pd.Series(inst_bpm).rolling(window=1000, center=True).mean()
    return inst_bpm

# definition for finding HR spikes indices
def find_hr_sp_ind(ts, hr, thresh, refrac, min_dur):
    #Returns the indices of spikes.  Refrac is in ms
    # #highpass filter hr data above 50Hz to remove mouvement artifacts
    # samp_freq = 1/(ts[1] - ts[0])
    # nyq = samp_freq/2
    # b, a = signal.butter(4, 50/nyq, "high", analog=False)
    # hr = signal.filtfilt(b, a, hr)
    hr_diff = np.reshape(hr, (1, hr.size))
    dVdt = np.diff(hr_diff)
    dVdt = np.reshape(dVdt, (dVdt.size))
    # show the threshold and where the spike would be detected
    dVdt_thresh = np.array(dVdt > thresh, float)
    if sum(dVdt_thresh) == 0:
        # there are no spikes
        hr_sp_ind = np.empty(shape=0)
    elif sum(dVdt_thresh) == 1:
        hr_sp_ind = np.where(np.diff(dVdt_thresh) == 1)[0]
        #remove spikes that are too short (artefacts)
        stop = np.where(np.diff(dVdt_thresh) == -1)[0]
        difference = stop - hr_sp_ind 
        hr_sp_ind = hr_sp_ind[np.where(difference > min_dur)]
    else:
        # keep just the first index per spike
        #sp_ind = np.squeeze(np.where(np.diff(dVdt_thresh) == 1))
        hr_sp_ind = np.where(np.diff(dVdt_thresh) == 1)[0]
        #remove spikes that are too short (artefacts)
        stop = np.where(np.diff(dVdt_thresh) == -1)[0]
        if stop[0] < hr_sp_ind[0]:
            stop = stop[1:]
        if stop[-1] < hr_sp_ind[-1]:
            hr_sp_ind = hr_sp_ind[:-1]
        difference = stop - hr_sp_ind 
        hr_sp_ind = hr_sp_ind[np.where(difference > min_dur)]
        # remove any duplicates of spikes that occur within refractory period
        samp_rate = np.round(1/(ts[1]-ts[0]))
        hr_sp_ind = hr_sp_ind[np.ediff1d(hr_sp_ind,
                        to_begin=int(samp_rate*refrac/1000+1)) > samp_rate*refrac/1000]      
    return dVdt_thresh, hr_sp_ind

#%%   
## do something with the data   


# set up list with an empty dictionary for each cell
data = [{'mouse_id': 0, 'synch': 0}
        for k in np.arange(mouse_ids.size)]


for i in np.arange(mouse_ids.size, dtype=int):
    # load the raw data from 1 cell at a time
    hr, hr_ts = load_data_MIND(i)
    
    #find peaks of hr data and calculate average and instant bpm
    dVdt_thresh = np.empty(0)
    hr_sp_times = np.empty(0)
    inst_bpm = np.empty(0)
    inst_bpm_ts = np.empty(0)
    hr_sp_ind = np.empty(0)
    RR = np.empty(0)
    rmssd = np.empty(0)     
    sd = np.empty(0)
    ssd = np.empty(0)
    mssd = np.empty(0)
    rmssd = np.empty(0) 
    rmssd_to_max = np.empty(0)
    bpm_to_max = np.empty(0)
    if hr.size > 0: 
        #sometimes, hr can be too noisy or contaminated by some oscillation - it can be useful to highpass filter the hr signal
        #in that case, add the highpass value to the data_to_analyze file
        if highpass[i] > 0:
            samp_freq = 1/(hr_ts[1] - hr_ts[0])
            nyq = samp_freq/2
            b, a = signal.butter(4, highpass[i]/nyq, "high", analog=False)
            hr_highpass = signal.filtfilt(b, a, hr)
            dVdt_thresh, hr_sp_ind = find_hr_sp_ind(hr_ts, hr_highpass, hr_thresh[i], hr_refrac[i], hr_min_dur[i])        
            # make the list of spike times if there are any
            hr_sp_times = np.empty(0)
            if hr_sp_ind.size > 0:
                hr_sp_times = hr_ts[hr_sp_ind]
        #sometimes, hr peak is mostly negative - so it can be usefull to take the absolute value of hr and then look for the peak times
        elif hr_abs[i] == 'yes': 
            hr = np.abs(hr)
            # find spike indices
            dVdt_thresh, hr_sp_ind = find_hr_sp_ind(hr_ts, hr, hr_thresh[i], hr_refrac[i], hr_min_dur[i])        
            # make the list of spike times if there are any
            hr_sp_times = np.empty(0)
            if hr_sp_ind.size > 0:
                hr_sp_times = hr_ts[hr_sp_ind]
            hr_highpass = np.empty(0)

        else:
            # find spike indices
            dVdt_thresh, hr_sp_ind = find_hr_sp_ind(hr_ts, hr, hr_thresh[i], hr_refrac[i], hr_min_dur[i])        
            # make the list of spike times if there are any
            hr_sp_times = np.empty(0)
            if hr_sp_ind.size > 0:
                hr_sp_times = hr_ts[hr_sp_ind]
            hr_highpass = np.empty(0)
        #calculate instantaneous bpm (per seconds)
        inst_bpm = find_inst_bpm(hr, hr_sp_times, hr_ts)
        # calculate RMSSD (Studer 2020 NatComm paper)
        RR = np.diff(hr_sp_times)
        #put in ms
        RR = RR*1000
        #successive differences in RR intervals 
        sd = np.diff(RR)
        #replace outliers (mean+-2sd) with median value
        outlier_max = np.nanmean(sd)+np.nanstd(sd)*2
        outlier_min = np.nanmean(sd)-np.nanstd(sd)*2
        sd = np.where((sd > outlier_max), np.median(sd, axis=0), sd)
        sd = np.where((sd < outlier_min), np.median(sd, axis=0), sd)
        ssd = np.square(sd) #square of successive differences (put everything positive)
        mssd = pd.Series(ssd).rolling(window=30, center=True).mean() #mean square of successive differences
        rmssd = np.sqrt(mssd)
        #smoothing of rmssd (same as inst_bpm)
        #rmssd = savgol_filter(rmssd, 21, 2)
        #compute rmssd_to_max
        rmssd_to_max = (rmssd*100)/np.max(rmssd[~np.isnan(rmssd)])
        #compute inst_bpm_to_max
        bpm_to_max = (inst_bpm*100)/np.max(inst_bpm[~np.isnan(inst_bpm)])
        #replace too low values (where bpm_to_max drops below median-4sd with median value
        bpm_to_max = np.where((bpm_to_max < (np.nanmedian(bpm_to_max, axis=0) - 4*np.nanstd(bpm_to_max))), 
                          np.nanmedian(bpm_to_max, axis=0), bpm_to_max)
        #replace too low values (where bpm_to_max drops below 30 with median value
        bpm_to_max = np.where((bpm_to_max < 30), 
                          np.nanmedian(bpm_to_max, axis=0), bpm_to_max)
        #replace too high rmssd values (where rmssdgoes above median+4sd with median value
        rmssd = np.where((rmssd > (np.nanmedian(rmssd, axis=0) + 4*np.nanstd(rmssd))), 
                          np.nanmedian(rmssd, axis=0), rmssd)

                
    # #downsample and filter inst_bpm for HR increase and decrease detection
    # ds_factor = 10
    # hr_ts_ds = np.empty(0)
    # hr_ds = np.empty(0)
    # if inst_bpm.size > 0:
    #     hr_ts_ds, hr_ds = ds(hr_ts, inst_bpm, ds_factor)
    #     hr_ds = savgol_filter(hr_ds, 21, 2)   
            
        
    # save relevant numbers in data list
    data[i]['mouse_id'] = mouse_ids[i]
    # data[i]['cell_id'] = cells[i]
    data[i]['hr'] = hr
    data[i]['hr_ts'] = hr_ts 
    # data[i]['dVdt_thresh'] = dVdt_thresh
    data[i]['R_start'] = hr_sp_times
    data[i]['inst_bpm'] = inst_bpm
    data[i]['inst_bpm_ts'] = hr_ts
    data[i]['bpm_to_max'] = bpm_to_max
    data[i]['hr_sp_ind'] = hr_sp_ind
    # data[i]['hrv'] = RR
    # data[i]['sd'] = sd 
    # data[i]['ssd'] = ssd
    # data[i]['mssd'] = mssd
    # data[i]['rmssd'] = rmssd 
    # data[i]['rmssd_to_max'] = rmssd_to_max    
    # data[i]['inst_bpm_ds_ts'] = hr_ts_ds
    # data[i]['inst_bpm_ds'] = hr_ds
    data[i]['hr_highpass'] = hr_highpass                

#%%
#check hr detection    
for i in np.arange(len(data)): 
    plt.figure() 
    plt.plot(data[i]['hr_ts'], data[i]['hr'], zorder=0) 
#    plt.plot(data[i]['hr_ts'][:-1], data[i]['dVdt_thresh']) 
    plt.scatter(data[i]['R_start'], 0*np.ones(data[i]['R_start'].size), c='r', zorder=1)

#%% clean hr detection with GUI - right click delete event, left click add event 
time_series = hr.astype(np.float16) 
events = hr_sp_times.tolist() 
# Example usage 
editor = EventEditor(hr_ts, time_series, events) 

#%%
np.array(events).size 
hr_sp_times.size 
hr_sp_times = np.array(np.sort(events)) #NOTE:Jan 2024 added np.sort because the added spikes are appended at the end of the hr_sp_times and not at their corresponding position
plt.figure() 
plt.plot(bpm_to_max) 

#%%
inst_bpm = find_inst_bpm(hr, hr_sp_times, hr_ts) 
# calculate RMSSD (Studer 2020 NatComm paper) 
RR = np.diff(hr_sp_times) 
#put in ms
RR = RR*1000
#successive differences in RR intervals 
sd = np.diff(RR) 
#replace outliers (mean+-2sd) with median value 
outlier_max = np.nanmean(sd)+np.nanstd(sd)*2
outlier_min = np.nanmean(sd)-np.nanstd(sd)*2 
sd = np.where((sd > outlier_max), np.median(sd, axis=0), sd) 
sd = np.where((sd < outlier_min), np.median(sd, axis=0), sd)
ssd = np.square(sd) #square of successive differences (put everything positive) 
mssd = pd.Series(ssd).rolling(window=30, center=True).mean() #mean square of successive differences
rmssd = np.sqrt(mssd)
#smoothing of rmssd (same as inst_bpm)
#rmssd = savgol_filter(rmssd, 21, 2)
#compute rmssd_to_max
rmssd_to_max = (rmssd*100)/np.max(rmssd[~np.isnan(rmssd)])
#replace too low values (where bpm_to_max drops below median-4sd with median value
inst_bpm = np.where((inst_bpm < (np.nanmedian(inst_bpm, axis=0) - 4*np.nanstd(inst_bpm))), 
                  np.nanmedian(inst_bpm, axis=0), inst_bpm)
inst_bpm = np.where((inst_bpm > (np.nanmedian(inst_bpm, axis=0) + 4*np.nanstd(inst_bpm))), 
                  np.nanmedian(inst_bpm, axis=0), inst_bpm)
#compute inst_bpm_to_max
bpm_to_max = (inst_bpm*100)/np.max(inst_bpm[~np.isnan(inst_bpm)]) 
#interpolate between missing data so that bpm does not drop down too low 
bpm_to_max = np.where((bpm_to_max < 10), 
                  np.nanmedian(bpm_to_max, axis=0), bpm_to_max) 
drop_start = np.where(np.ediff1d(bpm_to_max) > 5)
drop_stop = np.where(np.ediff1d(bpm_to_max) < -5)
if np.array(drop_start).size == np.array(drop_stop).size:
    for k in np.arange(drop_start[0].size):
        bpm_to_max[(drop_start[0][k]-1000):(drop_stop[0][k]+1000)] = 0
elif (np.array(drop_start).size + 1) == np.array(drop_stop).size: #if 1 more drop stop than drop start
    for k in np.arange(drop_start[0].size - 1):
        bpm_to_max[(drop_start[0][k]-1000):(drop_stop[0][k]+1000)] = 0
elif np.array(drop_start).size == 1 and np.array(drop_stop).size == 0: #if only one drop start at the end of rec
    for k in np.arange(drop_start[0].size):
        bpm_to_max[(drop_start[0][k]-1000):-1] = 0    
y = bpm_to_max 
x = np.arange(len(y))
idx = np.where(y!=0)        #or np.nonzero(y) 
f = interp1d(x[idx],y[idx])
bpm_to_max = f(x)
#replace too high rmssd values (where rmssd goes above median+4sd with median value
rmssd = np.where((rmssd > (np.nanmedian(rmssd, axis=0) + 4*np.nanstd(rmssd))), 
                  np.nanmedian(rmssd, axis=0), rmssd)

plt.plot(bpm_to_max)

data[i]['mouse_id'] = mouse_ids[i]
data[i]['hr'] = hr 
data[i]['hr_ts'] = hr_ts 
# data[i]['dVdt_thresh'] = dVdt_thresh
data[i]['R_start'] = hr_sp_times
data[i]['inst_bpm'] = inst_bpm
data[i]['inst_bpm_ts'] = hr_ts
data[i]['bpm_to_max'] = bpm_to_max
data[i]['hr_sp_ind'] = hr_sp_ind
# data[i]['hrv'] = RR
# data[i]['sd'] = sd
# data[i]['ssd'] = ssd 
# data[i]['mssd'] = mssd 
# data[i]['rmssd'] = rmssd 
# data[i]['rmssd_to_max'] = rmssd_to_max    
# data[i]['inst_bpm_ds_ts'] = hr_ts_ds
# data[i]['inst_bpm_ds'] = hr_ds
data[i]['hr_highpass'] = hr_highpass
#%%another version Jan 2024 for cleaner inst_bpm and bpm_to_max
inst_bpm = find_inst_bpm(data[i]['hr'], hr_sp_times, data[i]['hr_ts'])
# calculate RMSSD (Studer 2020 NatComm paper) 
RR = np.diff(hr_sp_times)
#put in ms
RR = RR*1000 
#successive differences in RR intervals 
sd = np.diff(RR)
#replace outliers (mean+-2sd) with median value 
outlier_max = np.nanmean(sd)+np.nanstd(sd)*2 
outlier_min = np.nanmean(sd)-np.nanstd(sd)*2 
sd = np.where((sd > outlier_max), np.median(sd, axis=0), sd)
sd = np.where((sd < outlier_min), np.median(sd, axis=0), sd) 
ssd = np.square(sd) #square of successive differences (put everything positive) 
mssd = pd.Series(ssd).rolling(window=30, center=True).mean() #mean square of successive differences
rmssd = np.sqrt(mssd)
#smoothing of rmssd (same as inst_bpm) 
#rmssd = savgol_filter(rmssd, 21, 2)
#compute rmssd_to_max
rmssd_to_max = (rmssd*100)/np.max(rmssd[~np.isnan(rmssd)])
#replace too low values (where bpm_to_max drops below median-4sd with median value
inst_bpm = np.where((inst_bpm < (np.nanmedian(inst_bpm, axis=0) - 4*np.nanstd(inst_bpm))), 
                  np.nanmedian(inst_bpm, axis=0), inst_bpm)
inst_bpm = np.where((inst_bpm > (np.nanmedian(inst_bpm, axis=0) + 4*np.nanstd(inst_bpm))), 
                  np.nanmedian(inst_bpm, axis=0), inst_bpm)
#compute inst_bpm_to_max
bpm_to_max = (inst_bpm*100)/np.max(inst_bpm[~np.isnan(inst_bpm)]) 
#interpolate between missing data so that bpm does not drop down too low 
bpm_to_max = np.where((bpm_to_max < 10), 
                  np.nanmedian(bpm_to_max, axis=0), bpm_to_max) 
drop_start = np.where(np.ediff1d(bpm_to_max) > 5)
drop_stop = np.where(np.ediff1d(bpm_to_max) < -5) 
if np.array(drop_start).size == np.array(drop_stop).size:
    for k in np.arange(drop_start[0].size):
        bpm_to_max[(drop_start[0][k]-1000):(drop_stop[0][k]+1000)] = 0
elif (np.array(drop_start).size + 1) == np.array(drop_stop).size: #if 1 more drop stop than drop start
    for k in np.arange(drop_start[0].size - 1):
        bpm_to_max[(drop_start[0][k]-1000):(drop_stop[0][k]+1000)] = 0
elif np.array(drop_start).size == 1 and np.array(drop_stop).size == 0: #if only one drop start at the end of rec
    for k in np.arange(drop_start[0].size):
        bpm_to_max[(drop_start[0][k]-1000):-1] = 0    
y = bpm_to_max 
x = np.arange(len(y))
idx = np.where(y!=0)        #or np.nonzero(y) 
f = interp1d(x[idx],y[idx]) 
bpm_to_max = f(x)
#replace too high rmssd values (where rmssdgoes above median+4sd with median value
rmssd = np.where((rmssd > (np.nanmedian(rmssd, axis=0) + 4*np.nanstd(rmssd))), 
                  np.nanmedian(rmssd, axis=0), rmssd)

plt.plot(bpm_to_max)

# data[i]['dVdt_thresh'] = dVdt_thresh
data[i]['R_start'] = hr_sp_times
data[i]['inst_bpm'] = inst_bpm
data[i]['bpm_to_max'] = bpm_to_max
 #%% 
# Save clean hr dictionaries using numpy 
for i in np.arange(len(data)): 
    np.save((r"\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Datasets\silicon_probes\Atenolol\hr_cleaned\mouse_" + 
             str((data[i]['mouse_id'])) + '.npy'), data[i]) 
             
#%%load clean hr
dataset_folder = (r"\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Datasets\silicon_probes\Atenolol\hr_cleaned")

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i]) 
    data[i] = np.load(full_file, allow_pickle=True).item()
    
 #%%example of combining datasets
    
dataset_folder = (r"D:\Postdoc\Analysis\Datasets\silicon_probes\ephys")

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()



CS_folder = (r"D:\Postdoc\Analysis\Datasets\silicon_probes\ephys")
CS_files = os.listdir(CS_folder)
for i in np.arange(len(CS_files)):
    full_file = os.path.join(CS_folder, CS_files[i])
    CS_data = np.load(full_file, allow_pickle=True).item()
    data[i].update(CS_data)    

##IN CASE THE TWO DATASETS DONT MATCH LENGTH AND TO ALIGN WITH THE SAME CELL OR MOUSE_IDS 
CS_folder = (r"D:\Postdoc\Analysis\Datasets\silicon_probes\hr_cleaned")
CS_files = os.listdir(CS_folder)
for g in np.arange(len(CS_files)):
    full_file = os.path.join(CS_folder, CS_files[g])
    CS_data = np.load(full_file, allow_pickle=True).item()
for i in np.arange(len(data)):
    ind = (data[i]['cell_id'] == CS_data['cell_id'])
    if ind == True:
        for g in np.arange(len(CS_files)):
            full_file = os.path.join(CS_folder, CS_files[g])
            CS_data = np.load(full_file, allow_pickle=True).item()
            data[i].update(CS_data)
#%%ecg waveform extraction 

t_win = [-0.05, 0.05]
ecg_waveforms, ecg_waveforms_ts = prepare_eta(data[i]['hr'], data[i]['hr_ts'], data[i]['R_start'], t_win)
ecg_wf_mean = np.nanmean(ecg_waveforms, axis=1)
fig, ax = plt.subplots(1, 1)
plt.figure()
plt.plot(ecg_waveforms_ts, ecg_waveforms[:, 50:100], c='lightgray', alpha=0.3)
plt.plot(ecg_waveforms_ts, ecg_wf_mean, c='black', linewidth=3)
plt.axis('off')

#%%if needed to reprocess the inst_bpm from the already processed datasets after loading them 

for i in np.arange(len(data)):
    plt.figure()
    inst_bpm = find_inst_bpm(data[i]['hr'], data[i]['R_start'], data[i]['hr_ts'])
    # calculate RMSSD (Studer 2020 NatComm paper)
    RR = np.diff(data[i]['R_start'])
    #put in ms
    RR = RR*1000
    #successive differences in RR intervals 
    sd = np.diff(RR)
    #replace outliers (mean+-2sd) with median value
    outlier_max = np.nanmean(sd)+np.nanstd(sd)*2
    outlier_min = np.nanmean(sd)-np.nanstd(sd)*2
    sd = np.where((sd > outlier_max), np.median(sd, axis=0), sd)
    sd = np.where((sd < outlier_min), np.median(sd, axis=0), sd)
    ssd = np.square(sd) #square of successive differences (put everything positive)
    mssd = pd.Series(ssd).rolling(window=30, center=True).mean() #mean square of successive differences
    rmssd = np.sqrt(mssd)
    #smoothing of rmssd (same as inst_bpm)
    #rmssd = savgol_filter(rmssd, 21, 2)
    #compute rmssd_to_max
    rmssd_to_max = (rmssd*100)/np.max(rmssd[~np.isnan(rmssd)])
    #compute inst_bpm_to_max
    bpm_to_max = (inst_bpm*100)/np.max(inst_bpm[~np.isnan(inst_bpm)])
    #replace too low values (where bpm_to_max drops below median-4sd with median value
    bpm_to_max = np.where((bpm_to_max < (np.nanmedian(bpm_to_max, axis=0) - 5*np.nanstd(bpm_to_max))), 
                      np.nanmedian(bpm_to_max, axis=0), bpm_to_max)
    bpm_to_max = np.where((bpm_to_max > (np.nanmedian(bpm_to_max, axis=0) + 4*np.nanstd(bpm_to_max))), 
                      np.nanmedian(bpm_to_max, axis=0), bpm_to_max)
    #interpolate between missing data so that bpm does not drop down too low 
    bpm_to_max = np.where((bpm_to_max < 10), 
                      np.nanmedian(bpm_to_max, axis=0), bpm_to_max) 
    drop_start = np.where(np.ediff1d(bpm_to_max) > 5)
    drop_stop = np.where(np.ediff1d(bpm_to_max) < -5)
    if drop_start[0].size == drop_stop[0].size:
        for k in np.arange(drop_start[0].size): 
            bpm_to_max[(drop_start[0][k]-1000):(drop_stop[0][k]+1000)] = 0
        y = bpm_to_max 
        x = np.arange(len(y)) 
        idx = np.where(y!=0)        #or np.nonzero(y) 
        f = interp1d(x[idx],y[idx])
        bpm_to_max = f(x)
    elif drop_start[0].size > drop_stop[0].size: #if the missing data is at the end of recording
        for k in np.arange(drop_start[0].size): 
            bpm_to_max[(drop_start[0][k]-1000):(drop_start[0][k]+1000)] = 0
        y = bpm_to_max  
        x = np.arange(len(y)) 
        idx = np.where(y!=0)        #or np.nonzero(y) 
        f = interp1d(x[idx],y[idx])
        bpm_to_max = f(x)
    #replace too high rmssd values (where rmssdgoes above median+4sd with median value 
    rmssd = np.where((rmssd > (np.nanmedian(rmssd, axis=0) + 4*np.nanstd(rmssd))), 
                      np.nanmedian(rmssd, axis=0), rmssd)

    plt.plot(bpm_to_max)
    data[i]['inst_bpm'] = inst_bpm
    data[i]['bpm_to_max'] = bpm_to_max 
    