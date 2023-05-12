"""
Author: boren li <borenli@cea-igp.ac.cn>
Created: 2019-6-14
"""
import os
import shutil
import fnmatch
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import re

import numpy as np
import pandas as pd
import h5py

from obspy import read, read_events, UTCDateTime
from obspy.core import Stream, Trace
from obspy.core.event import Catalog, ResourceIdentifier, Pick, WaveformStreamID
from obspy.core.inventory import Inventory, Network, Station, Channel
from obspy.core.inventory.util import Latitude, Longitude, Distance
from obspy.clients.filesystem.sds import Client

from pick import get_pdf, get_exist_picks, get_pick_list
from signal import signal_preprocessing, trim_stream, get_snr

from utils.paths import list_files_path

import ipdb as pdb
import pretty_errors
from tqdm import tqdm


# Global variables
FREQ_MIN=2
FREQ_MAX=20

#-----------------------------------------
# Globe variable
CUT_WIN_POS = -10   # The position of the cut windown referenced the phase time. Default: -10.
# CUT_WIN_POS = -25   # The position of the cut windown referenced the phase time
# CUT_WIN_POS = -30   # For p-phase random in the 30s window
#-----------------------------------------


def get_dir_list(file_dir, limit=None):
    """
    """
    file_list = []
    for file in _list_generator(file_dir):
        file_list.append(os.path.join(file_dir, file))

        if limit and len(file_list) >= limit:
            break

    return file_list


def _list_generator(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def read_event_list(file_list):
    """
    """
    catalog = Catalog()
    for file in file_list:
        catalog += _read_event(file)

    catalog.events.sort(key=lambda event: event.origins[0].time)

    return catalog


def _read_event(event_file):
    catalog = Catalog()
    try:
        cat = read_events(event_file)
        for event in cat.events:
            event.file_name = event_file

        catalog += cat

    except Exception as err:
        print(err)

    return catalog



def read_waveform3(wfFile, pick_time, phase="P", wf_name_by_picktime=True, wf_name_offset=60, cut_win_pos=CUT_WIN_POS, random=True, trace_length=30, sample_rate=100, norm_mode='std'):    
    """
    Parse the name of Z-channel waveform from the specified file and Generate the stream (3 channels) corresponding to the station.
        
    :param wfFile: The name of the waveform file of the Z-channel, including full path.  
    :param pick_time: The time of phase arrival.      
    :param wf_name_offset: The value (s) used by naming the original waveform. .
    :param cut_win_pos: The position of the windwo to be cut for training, i.e. -6 - before P pick time.
    :param random: The flag for random start time depending the P-pick time. Default to True (using random).
    :param trace_length: The length of cut window (seconds).
    :param sample_rate: 
    :param norm_mode: The normalization mode: 'std' or 'max'    
    :returns: Stream object for the event (including all triggered trace of event), The meata inforatmon of the generated waveform.    
    """
    components_list = ['E', 'N', 'Z']
    
    path, fileZChann = os.path.split(wfFile)
    if not fileZChann.endswith('Z'):
        return
        
    count = 0
    traces = []
    for chSubfix in components_list:
        tmp = list(wfFile)
        tmp[-1] = chSubfix           # replace z-channel name with E/N/Z-channel
        wfFilename = ''.join(tmp)  
#         wfFile = '%s.%s.%d%03d%02d%02d%02d.00.%s' % (net, sta, y2, j2, h2, M2, s2, chFilter)            
        wfFilename = os.path.join(path, wfFilename)
        if not os.path.isfile(wfFilename):
            continue
        
        count += 1
        try:
            trace = read(wfFilename)[0]
            traces.append(trace)
        except:
            pass

#     pdb.set_trace()
    stOrg = Stream(traces=traces)
    if random:
        tStart = pick_time + cut_win_pos + np.random.randint(abs(cut_win_pos))  # For P/S-phase waveforms cut
    else:
        tStart = pick_time - trace_length - 5  # Mainly for noise waveforms 
        
    stream = stOrg.slice(starttime=tStart, endtime=tStart+trace_length)

    return stream
    
    
        
def write_training_pkl3(event_list, sds_root, pkl_dir,  wf_name_by_picktime=True, remove_dir=False, norm_mode='std', phase="PS", sigma=0.1, filter=False, sigmaEx=None):
    """
    Write the train dataset (3 channels) for P-phase and S-phase.
    
    :param event_list: 
    :param sds_root: The path of the original waveforms, i.e, sds_root/waveform_folder/waveforms_file.
    :param pkl_dir: The output path of the cut window waveforms.
    :param remove_dir: Delete the dir whether not when the folder exists.
    :param norm_mode: The normalization mode: 'std' or 'max'. 
    :param phase: The phase information, i.e. 'PS' - P-phase and S-phase;'P'-P-phase(use for testing single phase).     
    :param components_list: The channel list of the station.  
    :param sigma: The sigma param of Gausian norm.    
    :param sigmaEx: The extra sigma param of Gausian norm is used to the S-phase when using the different value to P-phase and S-phase. Default to None.
    """
    wf_name_offset = 60    # used for retrieve the name of download waveforms
    
        # The meta information of the generatee waveform 
    metaWaveform = {
            'stream_name': [],      #  The name of the pick file
            'wavefrom_id': [],     # Store waveform id (the pick.waveform_id.get_SEED_string())
            'waveform_fname': [],  # the name of the original waveform file
            'id_event': [], 'origtime' :[], 'event_lat': [], 'event_lon': [], 'event_depth': [], 'mag': [],        # the information of the evenet 
            'network': [], 'station': [], 'channel': [],          # the information of the station 
            'phase': [], 'arrival_time': [], 'epi_distance_km': [], 'snr_db': []
           }  # event meta keys
       
    if remove_dir:
        shutil.rmtree(pkl_dir, ignore_errors=True)
    os.makedirs(pkl_dir, exist_ok=True)
    
#     pdb.set_trace()
    catalog = read_event_list(event_list)
    pick_list = get_pick_list(catalog)
    count = 0
    with tqdm(total=len(catalog)) as bar:      
        for event in catalog:
            evid = str(-999)
            strEvid = event.resource_id.id
            if strEvid.startswith('quakeml:'):
                res = re.findall('eventid=\d+', strEvid)
                evid = res[0]   
                
            origin = event.preferred_origin() or event.origins[0]
            arrs = origin.arrivals   
            
            # event information
            mag = event.magnitudes[0].mag
            event_lat = origin.latitude
            event_lon = origin.longitude
            event_depth = origin.depth            

            # Retrive the director of the waveforms
            t = origin.time
            y = t.year
            m = t.month
            d = t.day
            j = t.julday
            h = t.hour
            M = t.minute
            s = t.second
            ms = int(t.microsecond/1e3)
            # 事件波形文件夹：发震日期.juliday.发震时刻（GMT时间）
            pathEvt = '%d%02d%02d.%03d.%02d%02d%02d.%03d' % (y, m, d, j, h, M, s, ms)    
        
            # Iterate the Z-channel
            for arr in arrs:
                if not arr.phase == 'P':
                    continue
                    
                pickId = arr.pick_id
                ref_pick = ResourceIdentifier(pickId)
                pick = ref_pick.get_referred_object()

                waveform_id = pick.waveform_id
                net = waveform_id.network_code
                sta = waveform_id.station_code                
                ch = waveform_id.channel_code   # ch is **Z   
                if not ch[-1] == 'Z':            
                #             print('Skip: ' + waveform_id.get_seed_string())
                    continue 
                    
                if wf_name_by_picktime:  # 原scedc下载波形有一部分已发震时刻（-60s）命名（M>=3.0）;还有一部分是P震相时刻（-60s）命名
                    t = pick.time
     
                t2 = t - wf_name_offset  # The 60s values must match the download program, build the file name of the waveform                    
                y2 = t2.year
                j2 = t2.julday
                h2 = t2.hour
                M2 = t2.minute
                s2 = t2.second
        
                loc = '*'
#                 count = 0

                # 波形文件名network.station.波形起始时间(日期时刻)（发震/P震相前60s时间，GMT时间）.00.通道
                wfFile = '%s.%s.%d%03d%02d%02d%02d.00.%s' % (net, sta, y2, j2, h2, M2, s2, ch)  # ch=*Z,  Z-channel       
#                 pdb.set_trace()    
                wfFile = os.path.join(sds_root, pathEvt, wfFile)
                if not os.path.isfile(wfFile):
                    continue
                
                # Write a stream (3 channels) corresponding to one triggered station
#                 nsc  = (net, sta, ch)   # the tuple id of the station: (net, station, channel)
#                 _write_picked_trace3(event, pick_list=pick_list, sds_root=sds_root, pathEvt, nsc, pkl_dir=pkl_dir, norm_mode=norm_mode, phase=phase, sigma=sigma, filter=filter)
                stream_name = 'None'   #  The name of the pick file
                snr_db = -999          # Z-channel SNR at P-phase before and after
                result = _write_picked_trace3(wfFile, pick.time, pick_list=pick_list, pkl_dir=pkl_dir, norm_mode=norm_mode, phase=phase, sigma=sigma, filter=filter, sigmaEx=sigmaEx)
                if result is not None:
                    stream_name, snr_db = result
    
#                 epi_distance_km = arr.distance
                    epi_distance_km = arr.distance*111   # arr.distance unit is deg for scedc, need Unit:deg ==> km

                    # Store the meta inforamtion of the waveform 
                    metaWaveform['stream_name'].append(stream_name)            
                    metaWaveform['wavefrom_id'].append(waveform_id.get_seed_string())
                    metaWaveform['waveform_fname'].append(wfFile)
                    metaWaveform['id_event'].append(evid)
                    metaWaveform['origtime'].append(origin.time)
                    metaWaveform['event_lat'].append(event_lat)
                    metaWaveform['event_lon'].append(event_lon)
                    metaWaveform['event_depth'].append(event_depth/1e3)  # Unit:m ==> km
                    metaWaveform['mag'].append(mag)
                    metaWaveform['network'].append(net)
                    metaWaveform['station'].append(sta)
                    metaWaveform['channel'].append(ch[:-1])
                    metaWaveform['phase'].append(pick.phase_hint)
                    metaWaveform['arrival_time'].append(pick.time)
                    metaWaveform['epi_distance_km'].append(epi_distance_km)      
                    metaWaveform['snr_db'].append(snr_db) 

                    count += 1
                    bar.update(1)
                    if count%1000 == 0:
                        print('.', end='')
    
    # The log dataframe for recording the generated waveform   
    print('event coun %d' % count)
    dfMeta = pd.DataFrame(data=metaWaveform)    
#     pdb.set_trace()
    fileMeta = "genenrated_log.csv"
    dfMeta.to_csv(os.path.join(pkl_dir, fileMeta), index=False)
    print('The log of generating data: %s' % os.path.join(pkl_dir, fileMeta))    
    

        
def _write_picked_trace3(wfFile, pick_time, pick_list, pkl_dir, norm_mode, phase='PS', sigma=0.1, filter=False, sigmaEx=None):  
    """
    Write the traces (3 channels) into the pickle files, including P and S phase picks, and the value of their pdf.
    
    :param wfFile: The name of the waveform file of the Z-channel.
    :param pick_time: The P-phse pick time, used to find the waveform file. 
    :param pkl_dir: The output path of the cut window waveforms.
    :param norm_mode: The normalization mode: 'std' or 'max'.    
    :param phase: The phase information, i.e. 'PS' - P-phase and S-phase;'P'-P-phase(use for testing single phase). 
    :param sigma: The sigma param of Gausian norm.       
    :param filter: Whether using bandpass filter on the traces.
    :param sigmaEx: The extra sigma param of Gausian norm is used to the S-phase when using the different value to P-phase and S-phase. Default to None.  
    """
    trace_length=30 
    points = 3001
    components_list=['E', 'N', 'Z']
    
#     pdb.set_trace()
    stream = read_waveform3(wfFile, pick_time)    # default
#     stream = read_waveform3(wfFile, pick_time, random=False)   # For testing, fix picktime when cut window       
    if len(stream) != 3:
        return
    try:   
        tStart = stream[0].stats.starttime
        stream.trim(starttime=tStart, endtime=tStart+trace_length)
        for i, tr in enumerate(stream):
            stream.traces[i] = signal_preprocessing(tr, mode=norm_mode, filter=filter)
        trim_stream(stream, points)     

        pickPoint = (pick_time - stream[2].stats.starttime) * stream[2].stats.sampling_rate + 1
        snr_db = get_snr(stream[2].data, pickPoint)  # Z-chennel SNR at P-phase before and after

#         pdb.set_trace()
        # picks list for P-pahse and S-phase
        picks_p = get_exist_picks(stream[0], pick_list, phase='P')   # Z-channel
        picks_s = get_exist_picks(stream[0], pick_list, phase='S')     

        # Process P-phase/S-pahse - used for testing
        if phase.upper() == 'P' or phase.upper() == 'S':        
            # The picks attributes of the stream object
            if phase.upper() == 'P':
                stream.picks = {'P': picks_p, 'S': [] }
            else:
                stream.picks = {'P': [], 'S': picks_s }
        # Process P-phase and S-phase                
        else:  
            # The picks attributes of the stream object
            stream.picks = {'P': picks_p, 'S': picks_s }

        # Pdf for P-pahse and S-phase
        pdf_p = get_pdf(stream[0], stream.picks['P'], sigma=sigma)
        if sigmaEx is None:
            pdf_s = get_pdf(stream[0], stream.picks['S'], sigma=sigma)
        else:
            pdf_s = get_pdf(stream[0], stream.picks['S'], sigma=sigmaEx)            
        stream.pdf = [pdf_p, pdf_s] 

        time_stamp = stream[0].stats.starttime.isoformat() 
        ids = stream[0].get_id().split('.')
        stream_name = '%s.%s' % (ids[0], ids[1])
        stream_name = time_stamp + stream_name + '.pkl'
        stream.write(pkl_dir + '/' + stream_name, format='PICKLE')
    
        return (stream_name, snr_db)
    
    except:
        return None
    

def _gen_pick(dt_attrs, phase='P'):
    """
    Generate a Pick object from the dataset meta, i.e STEAD.
    
    :param dt_attrs: The dataset meta.
    :param phase: The phase name, 'P' or 'S'.
    :returns: The pick list.
    """
    dt = 0.01  # Unit: s(second)
    net = dt_attrs['network_code']
    station = dt_attrs['receiver_code']
    channel_code = '%sZ' % dt_attrs['receiver_type']
    narrvial_sample = '%s_arrival_sample' % phase.lower()
    if dt_attrs[narrvial_sample]*dt > 30.0:
        return []
    else:
        t = UTCDateTime(dt_attrs['trace_start_time']) + dt_attrs[narrvial_sample]*dt
    
    pick = Pick()
    wid = WaveformStreamID(network_code=net, station_code=station, channel_code=channel_code)
    pick.waveform_id = wid
    pick.time = t  
    pick.phase_hint = phase
    pick.evaluation_mode = 'manual'
    listPick = []   # picks list 
    listPick.append(pick)
 
    return listPick
  
    

        
   
    

  
    