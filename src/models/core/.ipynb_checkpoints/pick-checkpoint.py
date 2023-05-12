import fnmatch
import os
import shutil
import pickle
from bisect import bisect_left, bisect_right
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as ss
from obspy import read
from obspy.core.event.base import QuantityError, WaveformStreamID, ResourceIdentifier
from obspy.core.event.origin import Pick
from scipy.signal import find_peaks
from tqdm import tqdm

from signal import get_snr

import ipdb as pdb


def get_pick_list(catalog):
    pick_list = []
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        arrs = origin.arrivals
#         for p in event.picks:
        for arr in arrs:
            pickId = arr.pick_id
            ref_pick = ResourceIdentifier(pickId)
            p = ref_pick.get_referred_object()
            
            # Only receive the manual pick <-------------!!!!< Prefer filtered by catalog, not Here!!!
            if p.evaluation_mode != 'manual':
                continue
            
            if event.origins:
                p.origin = origin
            pick_list.append(p)
            
    pick_list.sort(key=lambda pick: pick.time)
    return pick_list


def get_pdf(trace, picks, sigma=0.1):
    """
    Get the pdf value of the pick's time of the trace. 
    
    :param trace: The object of the trace in obspy.
    :param picks: The array of obspy's Pick object.
    """
    start_time = trace.stats.starttime
#     x_time = trace.times(reftime=start_time)
#     pdf = np.zeros((len(x_time),))
    dataLength = len(trace)
    maskWindow = 2 * int(sigma * trace.stats.sampling_rate) + 1 
    pdf = np.zeros((dataLength,))
    
    for pick in picks:
#         pick_time = pick.time - start_time
#         pick_pdf = ss.norm.pdf(x_time, pick_time, sigma)
        pickPoint = int((pick.time - start_time) * trace.stats.sampling_rate)
        pick_pdf = _gen_pdf_func(dataLength, pickPoint, maskWindow)

        if pick_pdf.max():
            pdf += pick_pdf / pick_pdf.max()

    if pdf.max():
        pdf = pdf / pdf.max()

    return pdf
    

def _gen_pdf_func(data_length, point, mask_window):
    """
    Generating label data (target).
    
    :param data_length: target function length
    :param point: point of phase arrival
    :param mask_window: length of mask, must be odd number 
                 (mask_window//2+1+mask_window//2)
    """
    target = np.zeros(data_length)
    if point < 0:
        return target
    
    half_win = (mask_window-1)//2
    gaus = np.exp(-(
        np.arange(-half_win, half_win+1))**2 / (2*(half_win//2)**2))
    # print(gaus.std())
    gaus_first_half = gaus[:mask_window//2]
    gaus_second_half = gaus[mask_window//2+1:]
    target[point] = gaus.max()
    # #print(gaus.max())
    if point < half_win:
        reduce_pts = half_win-point
        start_pt = 0
        gaus_first_half = gaus_first_half[reduce_pts:]
    else:
        start_pt = point-half_win

    target[start_pt:point] = gaus_first_half
    target[point+1:point+half_win+1] = \
        gaus_second_half[:len(target[point+1:point+half_win+1])]
    
    return target


def get_picks_from_pdf(trace, pdf, phase='P',  height=0.5, distance=100, width=10):
    """
    Convert the pdf valut to the  pick list.
     
    :param trace:    
    :param pdf: The pdf distribution of the trace, which have the same dimension.
    :param phase: The type of the pick object. i.e. 'P' or 'S'.
    :param height:
    :param distance:    
    """
    start_time = trace.stats.starttime
    peaks, properties = find_peaks(pdf, height=height, distance=distance, width=width)

    picks = []
    for i, p in enumerate(peaks):
        if p:
            time = start_time + p / trace.stats.sampling_rate
            phase_hint = "P"
            pick = Pick(time=time, phase_hint=phase_hint)
            pick.phase_hint = phase
            pick.waveform_id = WaveformStreamID(network_code=trace.stats.network, station_code=trace.stats.station,
                                                location_code=trace.stats.location, channel_code=trace.stats.channel)
            pick.prob = properties['peak_heights'][i] 
            picks.append(pick)

    return picks


def get_picks_from_dataset(dataset):
    """
    Retrieve the pick list from the dataset (trace).
    
    :param dataset: A trace.
    :param phase: The phase name which want to be retrieve from the trace.   
    """
    pick_list = []
    trace = read(dataset, headonly=True).traces[0]
    picks = trace.picks
    pick_list.extend(picks)
    return pick_list



def get_picks_from_dataset3(dataset, phase='P'):
    """
    Retrieve the pick list from the dataset (trace), which element includes 'P' and 'S' keys. 
    
    :param dataset: A stream, i.e. a pickle file.
    :param phase: The phase name which want to be retrieve from the stream.   
    """
    pick_list = []
    with open(dataset, 'rb') as fp:
        st = pickle.load(fp)

    picks = st.picks[phase]
    pick_list.extend(picks)
    return pick_list



def _search_pick(pick_list, start_time, end_time):
    # binary search, pick_list must be sorted by time
    pick_time_key = []
    for pick in pick_list:
        pick_time_key.append(pick.time)

    left = bisect_left(pick_time_key, start_time)
    right = bisect_right(pick_time_key, end_time)
    pick_list = pick_list[left:right]

    return pick_list


def get_exist_picks(trace, pick_list, phase="P"):
    start_time = trace.stats.starttime
    end_time = trace.stats.endtime
    network = trace.stats.network
    station = trace.stats.station
    location = trace.stats.location
    channel = "*" + trace.stats.channel[-1]

    pick_list = _search_pick(pick_list, start_time, end_time)

    tmp_pick = []
    for pick in pick_list:
        network_code = pick.waveform_id.network_code
        station_code = pick.waveform_id.station_code
        location_code = pick.waveform_id.location_code
        channel_code = pick.waveform_id.channel_code

        if not start_time < pick.time < end_time:
            continue

        if not pick.phase_hint == phase:
            continue

        if network and network_code and not network_code == 'NA':
            if not fnmatch.fnmatch(network_code, network):
                continue

        if station:
            if not fnmatch.fnmatch(station_code, station):
                continue

#         if location and location_code:
#             if not fnmatch.fnmatch(location_code, location):
#                 continue

        # Modify the following code for gaining two phase (P/S) of a trace
        if channel:
            if not fnmatch.fnmatch(channel_code[:-1], channel[:-1]):
                continue
                
        #[Note] some auto-picks maybe set into 'manual'
        pick.evaluation_mode = "manual"
        tmp_pick.append(pick)

    return tmp_pick


def write_pdf_to_dataset(predict, dataset_list, dataset_output_dir, pick_phase='P', remove_dir=False, height=0.5, distance=100, normalize_pdf=False):
    """
    Write the pdf value into the trace, only for a type phase.
    
    :param pick_phase: The type of the pick, i.e. 'P' or 'S'
    :param normalize_pdf: The flag of pdf value whether normalization. Default to False.
    """      
    if remove_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # The information of the prediction result     
    predResult = {'trace_name': [],   # The name of the pick file
                  'trace_name_pred': [],   # The name of the pick file after prediction
                  'trace_id': [],   # trace.get_id()
                  'network': [], 'station': [], 'channel': [],          # the information of the station 
                  'trace_start_time': [], 'trace_end_time': [],
                  'phase': [], 
                  'p_arrival_time': [], 'p_probability': [], 'p_time_errors': [], 
                  'p_pdf_max': [], 'p_snr_db': [],
                  'evaluation_mode': [], 'evaluation_status': []
                  }   
    
    print("Output file:")
    with tqdm(total=len(dataset_list)) as pbar:
        for i, prob in enumerate(predict):
            try:
                trace = read(dataset_list[i]).traces[0]

            except IndexError:
                print('indexError = %d' % i)
                break

            trace_length = trace.data.size
            pdf = prob.reshape(trace_length, )

            if normalize_pdf and pdf.max():      # normalize the prediction pdf
                trace.pdf = pdf / pdf.max()
            else:                                # Non-normalize the prediction pdf (recommendation)
                trace.pdf = pdf     
            pdf_picks = get_picks_from_pdf(trace, height=height, distance=distance)
#             pdb.set_trace()

            if trace.picks:
                for val_pick in trace.picks:
                    for pre_pick in pdf_picks:
                        pre_pick.phase_hint = pick_phase                        
                        pre_pick.evaluation_mode = "automatic"

                        residual = get_time_residual(val_pick, pre_pick, delta=5.0)
                        pre_pick.time_errors = QuantityError(residual)

                        if is_close_pick(val_pick, pre_pick, delta=0.1):
#                         if is_close_pick(val_pick, pre_pick, delta=0.2):
#                         if is_close_pick(val_pick, pre_pick, delta=0.3):
#                         if is_close_pick(val_pick, pre_pick, delta=0.4):
#                         if is_close_pick(val_pick, pre_pick, delta=0.5):
                            pre_pick.evaluation_status = "confirmed"
                        elif is_close_pick(val_pick, pre_pick, delta=1):
                            pre_pick.evaluation_status = "rejected"

            else:
                trace.picks = []
                for pre_pick in pdf_picks:
                    pre_pick.evaluation_mode = "automatic"
                    pre_pick.evaluation_status = "preliminary"

            # Store the result of the prediction for a trace
            time_stamp = trace.stats.starttime.isoformat()
            filePred = dataset_output_dir + '/' + time_stamp + trace.get_id() + ".pkl"     # The name of the prediction file
            
            traceId = trace.get_id()
            net = trace.stats.network
            sta = trace.stats.station
            ch = trace.stats.channel
            sTime = trace.stats.starttime.isoformat()
            eTime = trace.stats.endtime.isoformat()
            for pick in pdf_picks: 
                predResult['trace_name'].append(dataset_list[i])
                predResult['trace_name_pred'].append(filePred)                
                predResult['trace_id'].append(traceId)
                predResult['network'].append(net)
                predResult['station'].append(sta)
                predResult['channel'].append(ch)                
                predResult['trace_start_time'].append(sTime)            
                predResult['trace_end_time'].append(eTime)
                predResult['phase'].append(phase)
                predResult['p_arrival_time'].append(pick.time)
                predResult['p_pdf_max'].append(pdf.max())  
                predResult['evaluation_mode'].append(pick.evaluation_mode)
                predResult['evaluation_status'].append(pick.evaluation_status)  
                if pick.time_errors.uncertainty is not None:
                    predResult['p_time_errors'].append(pick.time_errors.uncertainty)                    
                else:
                    predResult['p_time_errors'].append(-999)
                if pick.prob: 
                    predResult['p_probability'].append(pick.prob)
                else:
                    predResult['p_probability'].append(-999)
                             
                pickPoint = (pick.time - trace.stats.starttime) * trace.stats.sampling_rate + 1
                snr_db = get_snr(trace.data, pickPoint)
                predResult['p_snr_db'].append(snr_db)
                             
            trace.picks.extend(pdf_picks)
            trace.write(filePred, format="PICKLE")
            pbar.update()
    
    # Store the summary of prediction 
    dfPred = pd.DataFrame(data=predResult)
    fileOut = "prediction_results.csv"
    dfPred.to_csv(os.path.join(dataset_output_dir, fileOut), index=False)            
            
               
def write_pdf_to_dataset3(predict, dataset_list, dataset_output_dir, remove_dir=False, height=0.5, distance=100, normalize_pdf=False):
    """
    Write the pdf value into the stream (Z-channel), incuding P/S-phave two length list.
    """
    if remove_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # The information of the prediction result     
    predResult = {'trace_name': [],   # The name of the pick file
                  'trace_name_pred': [],   # The name of the pick file after prediction
                  'trace_id': [],   # trace.get_id()
                  'network': [], 'station': [], 'channel': [],          # the information of the station 
                  'trace_start_time': [], 'trace_end_time': [],
                  'phase': [], 
                  'arrival_time': [], 'probability': [], 'time_errors': [], 
                  'pdf_max': [], 'snr_db': [],
                  'evaluation_mode': [], 'evaluation_status': []
                  }  
    
    print("Output file:")
    with tqdm(total=len(dataset_list)) as pbar:
#         for i, prob in enumerate(zip(predict[0], predict[1]):   # P-phase, S-phase probability
        for i in range(len(predict[0])):
            prob = (predict[0][i], predict[1][i])  # P-phase, S-phase probability
            try:
                trace = read(dataset_list[i]).traces[0]

            except IndexError:
                break

            trace_length = trace.data.size
            pdf = []        # length=2
            pdf_picks = []  # length=2
            if len(prob)==2:
                pdf.append(prob[0].reshape(trace_length, ))  # P-phase probability
                pdf.append(prob[1].reshape(trace_length, ))  # S-phase probability
            
            for j in range(len(pdf)):
                if normalize_pdf and pdf[j].max():      # normalize the prediction pdf
                    pdf[j] = pdf[j] / pdf[j].max()
                else:                                   # Non-normalize the prediction pdf ---recommendation
                    pdf[j] = pdf[j]    
                
                #TODO: 需验证2021.6.28
                pdf_pick = get_picks_from_pdf(trace, pdf[j], height=height, distance=distance)
                pdf_picks.append(pdf_pick)
        
            trace.pdf = pdf          

            # Upate the P-phase pick information
            _upate_trace_picks(trace, pdf_picks[0], pick_phase='P')
            # Upate the S-phase pick information
            _upate_trace_picks(trace, pdf_picks[1], pick_phase='S')
#             pdb.set_trace() 

            traceId = trace.get_id()
            net = trace.stats.network
            sta = trace.stats.station
            ch = trace.stats.channel
            sTime = trace.stats.starttime.isoformat()
            eTime = trace.stats.endtime.isoformat()
            for pick in pdf_picks: 
                predResult['trace_name'].append(dataset_list[i])
                predResult['trace_name_pred'].append(filePred)                
                predResult['trace_id'].append(traceId)
                predResult['network'].append(net)
                predResult['station'].append(sta)
                predResult['channel'].append(ch)                
                predResult['trace_start_time'].append(sTime)            
                predResult['trace_end_time'].append(eTime)
                predResult['phase'].append(phase)
                predResult['arrival_time'].append(pick.time)
                predResult['pdf_max'].append(pdf.max())  
                predResult['evaluation_mode'].append(pick.evaluation_mode)
                predResult['evaluation_status'].append(pick.evaluation_status)  
                if pick.time_errors.uncertainty is not None:
                    predResult['time_errors'].append(pick.time_errors.uncertainty)                    
                else:
                    predResult['time_errors'].append(-999)
                if pick.prob: 
                    predResult['probability'].append(pick.prob)
                else:
                    predResult['probability'].append(-999)
                             
                pickPoint = (pick.time - trace.stats.starttime) * trace.stats.sampling_rate + 1
                snr_db = get_snr(trace.data, pickPoint)
                predResult['snr_db'].append(snr_db)
                             
            trace.picks['P'].extend(pdf_picks[0])
            trace.picks['S'].extend(pdf_picks[1])
            time_stamp = trace.stats.starttime.isoformat()
            trace.write(dataset_output_dir + '/' + time_stamp + trace.get_id() + ".pkl", format="PICKLE")
            pbar.update()

            

def write_pdf_to_dataset3_ex(predict, dataset_list, dataset_output_dir, remove_dir=False, height=0.5, distance=100, normalize_pdf=False):
    """
    Write the pdf value into the stream (3-channels trace), incuding P/S-phave two length list.
    
    :param predict:
    :param dataset_list:
    :param dataset_output_dir:
    :param remove_dir:
    :param height:
    :param distance:
    :param normalize_pdf: 
    """
    if remove_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # The information of the prediction result     
    predResult = {'stream_name': [],   # The name of the pick file
                  'stream_name_pred': [],   # The name of the pick file after prediction
                  'stream_id': [],   # trace.get_id() exclude the channel information
                  'network': [], 'station': [],          # the information of the station 
                  'stream_start_time': [], 'stream_end_time': [],
                  'phase': [], 
                  'arrival_time': [], 'probability': [], 'time_errors': [], 
                  'pdf_max': [], 'snr_db': [],
                  'evaluation_mode': [], 'evaluation_status': []
                  }  
    
    print("Output file:")
    with tqdm(total=len(dataset_list)) as pbar:
#         for i, prob in enumerate(zip(predict[0], predict[1]):
        for i in range(len(predict[0])):
            prob = (predict[0][i], predict[1][i])  # P-phase, S-phase probability
            try:
                with open(dataset_list[i], 'rb') as fp:
                    stream = pickle.load(fp) 

            except IndexError:
                break
                
            if len(stream) != 3:
                continue

#             pdb.set_trace()
            filePathStream = dataset_list[i]
            
            # Retrievt the pdf and picks information
            trace = stream[2]               
            trace_length = trace.data.size
            pdf = []        # length=2
            pdf_picks = []  # length=2, including two pick list.
            phases = ['P', 'S']
            if len(prob)==2:
                pdf.append(prob[0].reshape(trace_length, ))  # P-phase probability
                pdf.append(prob[1].reshape(trace_length, ))  # S-phase probability

            for j in range(len(pdf)):                
                if normalize_pdf and pdf[j].max():      # normalize the prediction pdf
                    pdf[j] = pdf[j] / pdf[j].max()
                else:                                   # Non-normalize the prediction pdf ---recommendation
                    pdf[j] = pdf[j] 
                
                pdf_pick = get_picks_from_pdf(trace, pdf[j], phase=phases[j], height=height, distance=distance)   # picks list 
                pdf_picks.append(pdf_pick)
            
            stream.pdf = pdf 

            # Upate the pick information in pdf_picks for P-phase: pdf_picks[0]
            _upate_stream_picks(stream, pdf_picks[0], pick_phase='P')
            # PUpate the pick information in pdf_picks for P-phase for S-phase: pdf_picks[1]
            _upate_stream_picks(stream, pdf_picks[1], pick_phase='S')  
            
            # Store the result of the prediction for a stream
#             streamId = trace.get_id()[:trace.get_id().rfind('.')-1]
            ids = trace.get_id().split('.')
            streamId = '%s.%s' % (ids[0], ids[1])            
            time_stamp = trace.stats.starttime.isoformat()
            filePred = dataset_output_dir + '/' + time_stamp + streamId + ".pkl"     # The name of the prediction 
            
            net = trace.stats.network
            sta = trace.stats.station
#             ch = trace.stats.channel
            sTime = trace.stats.starttime.isoformat()
            eTime = trace.stats.endtime.isoformat()
            for j, picks in enumerate(pdf_picks):     # pdf_picks[0] - P-phase picks; pdf_picks[1] - S-phase picks
                for pick in picks: 
                    predResult['stream_name'].append(filePathStream)
                    predResult['stream_name_pred'].append(filePred)                
                    predResult['stream_id'].append(streamId)
                    predResult['network'].append(net)
                    predResult['station'].append(sta)
    #                 predResult['channel'].append(ch)
                    predResult['stream_start_time'].append(sTime)            
                    predResult['stream_end_time'].append(eTime)
                    predResult['phase'].append(phases[j])         
                    predResult['arrival_time'].append(pick.time)
                    predResult['pdf_max'].append(pdf[j].max())  
                    predResult['evaluation_mode'].append(pick.evaluation_mode)
                    predResult['evaluation_status'].append(pick.evaluation_status)  
                    if pick.time_errors.uncertainty is not None:
                        predResult['time_errors'].append(pick.time_errors.uncertainty)                    
                    else:
                        predResult['time_errors'].append(-999)
                    if pick.prob: 
                        predResult['probability'].append(pick.prob)
                    else:
                        predResult['probability'].append(-999)

                    pickPoint = int((pick.time - trace.stats.starttime) * trace.stats.sampling_rate + 1)
                    snr_db = get_snr(trace.data, pickPoint)
                    predResult['snr_db'].append(snr_db)   

            stream.picks['P'].extend(pdf_picks[0])
            stream.picks['S'].extend(pdf_picks[1])
            time_stamp = trace.stats.starttime.isoformat()
            stream.write(filePred, format="PICKLE")
            pbar.update()
                 
    # Store the summary of prediction 
    dfPred = pd.DataFrame(data=predResult)
    fileOut = "prediction_results.csv"
    dfPred.to_csv(os.path.join(dataset_output_dir, fileOut), index=False)       

    
def write_pdf_to_dataset3_ex2(predict, dataset_list, dataset_output_dir, remove_dir=False, height=0.5, distance=100, normalize_pdf=False):
    """
    Write the pdf value into the stream (3-channels trace), incuding P/S-phave two length list for softmax (P-phase and Noise(S-phase)).
    
    :param predict:
    :param dataset_list:
    :param dataset_output_dir:
    :param remove_dir:
    :param height:
    :param distance:
    :param normalize_pdf: 
    """
    if remove_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # The information of the prediction result     
    predResult = {'stream_name': [],   # The name of the pick file
                  'stream_name_pred': [],   # The name of the pick file after prediction
                  'stream_id': [],   # trace.get_id() exclude the channel information
                  'network': [], 'station': [],          # the information of the station 
                  'stream_start_time': [], 'stream_end_time': [],
                  'phase': [], 
                  'arrival_time': [], 'probability': [], 'time_errors': [], 
                  'pdf_max': [], 'snr_db': [],
                  'evaluation_mode': [], 'evaluation_status': []
                  }  
    
    print("Output file:")
    with tqdm(total=len(dataset_list)) as pbar:
        for i in range(len(predict)):
            prob = predict[i]  # the pdf of P-phase and noise predition 
            try:
                with open(dataset_list[i], 'rb') as fp:
                    stream = pickle.load(fp) 

            except IndexError:
                break
                
            if len(stream) != 3:
                continue

#             pdb.set_trace()
            filePathStream = dataset_list[i]
            
            # Retrievt the pdf and picks information
            trace = stream[2]               
            trace_length = trace.data.size
            pdf = []        # length=2
            pdf_picks = []  # length=2, including two pick list.
            phases = ['P', 'S']
            tmpSphasePdf = np.zeros(trace_length)    # temporary varibale to be deleted util finishing S-pdf
            pdf.append(prob[:, :, 0].reshape(trace_length, ))  # P-phase probability
            pdf.append(tmpSphasePdf)  # noise probability

            for j in range(len(pdf)):                
                if normalize_pdf and pdf[j].max():      # normalize the prediction pdf
                    pdf[j] = pdf[j] / pdf[j].max()
                else:                                   # Non-normalize the prediction pdf ---recommendation
                    pdf[j] = pdf[j] 
                
                pdf_pick = get_picks_from_pdf(trace, pdf[j], phase=phases[j], height=height, distance=distance)   # picks list 
                pdf_picks.append(pdf_pick)
            
            stream.pdf = pdf 

            # Upate the pick information in pdf_picks for P-phase: pdf_picks[0]
            _upate_stream_picks(stream, pdf_picks[0], pick_phase='P')
            # PUpate the pick information in pdf_picks for P-phase for S-phase: pdf_picks[1]
            _upate_stream_picks(stream, pdf_picks[1], pick_phase='S')  
            
            # Store the result of the prediction for a stream
#             streamId = trace.get_id()[:trace.get_id().rfind('.')-1]
            ids = trace.get_id().split('.')
            streamId = '%s.%s' % (ids[0], ids[1])  
            time_stamp = trace.stats.starttime.isoformat()
            filePred = dataset_output_dir + '/' + time_stamp + streamId + ".pkl"     # The name of the prediction 
            
            net = trace.stats.network
            sta = trace.stats.station
#             ch = trace.stats.channel
            sTime = trace.stats.starttime.isoformat()
            eTime = trace.stats.endtime.isoformat()
            for j, picks in enumerate(pdf_picks):     # pdf_picks[0] - P-phase picks; pdf_picks[1] - S-phase picks
                for pick in picks: 
                    predResult['stream_name'].append(filePathStream)
                    predResult['stream_name_pred'].append(filePred)                
                    predResult['stream_id'].append(streamId)
                    predResult['network'].append(net)
                    predResult['station'].append(sta)
    #                 predResult['channel'].append(ch)
                    predResult['stream_start_time'].append(sTime)            
                    predResult['stream_end_time'].append(eTime)
                    predResult['phase'].append(phases[j])         
                    predResult['arrival_time'].append(pick.time)
                    predResult['pdf_max'].append(pdf[j].max())  
                    predResult['evaluation_mode'].append(pick.evaluation_mode)
                    predResult['evaluation_status'].append(pick.evaluation_status)  
                    if pick.time_errors.uncertainty is not None:
                        predResult['time_errors'].append(pick.time_errors.uncertainty)                    
                    else:
                        predResult['time_errors'].append(-999)
                    if pick.prob: 
                        predResult['probability'].append(pick.prob)
                    else:
                        predResult['probability'].append(-999)

                    pickPoint = int((pick.time - trace.stats.starttime) * trace.stats.sampling_rate + 1)
                    snr_db = get_snr(trace.data, pickPoint)
                    predResult['snr_db'].append(snr_db)   

            stream.picks['P'].extend(pdf_picks[0])
            stream.picks['S'].extend(pdf_picks[1])
            time_stamp = trace.stats.starttime.isoformat()
            stream.write(filePred, format="PICKLE")
            pbar.update()
                 
    # Store the summary of prediction 
    dfPred = pd.DataFrame(data=predResult)
    fileOut = "prediction_results.csv"
    dfPred.to_csv(os.path.join(dataset_output_dir, fileOut), index=False)       


def write_pdf_to_dataset3_ex3(predict, dataset_list, dataset_output_dir, remove_dir=False, height=(0.5,0.5), distance=100, normalize_pdf=False):
    """
    Write the pdf value into the stream (3-channels trace), incuding P and S-phave two length list for softmax  (P-phase ，S-phase and Noise).
    
    :param predict:
    :param dataset_list:
    :param dataset_output_dir:
    :param remove_dir:
    :param height: The threshold of the pdf for P-phase and S-phase, i.e. tuple(p-thr, s-thr). 
    :param distance:
    :param normalize_pdf: 
    """
    if remove_dir:
        shutil.rmtree(dataset_output_dir, ignore_errors=True)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # The information of the prediction result     
    predResult = {'stream_name': [],   # The name of the pick file
                  'stream_name_pred': [],   # The name of the pick file after prediction
                  'stream_id': [],   # trace.get_id() exclude the channel information
                  'network': [], 'station': [],          # the information of the station 
                  'stream_start_time': [], 'stream_end_time': [],
                  'phase': [], 
                  'arrival_time': [], 'probability': [], 'time_errors': [], 
                  'pdf_max': [], 'snr_db': [],
                  'evaluation_mode': [], 'evaluation_status': []
                  }  
    
    print("Output file:")
    with tqdm(total=len(dataset_list)) as pbar:
        for i in range(len(predict)):
            prob = predict[i]  # the pdf of P-phase and noise predition 
            try:
                with open(dataset_list[i], 'rb') as fp:
                    stream = pickle.load(fp) 

            except IndexError:
                break
                
            if len(stream) != 3:
                continue

#             pdb.set_trace()
            filePathStream = dataset_list[i]
            
            # Retrievt the pdf and picks information
            trace = stream[2]               
            trace_length = trace.data.size
            pdf = []        # length=2
            pdf_picks = []  # length=2, including two pick list.
            phases = ['P', 'S']
            pdf.append(prob[:, :, 0].reshape(trace_length, ))  # P-phase probability
            pdf.append(prob[:, :, 1].reshape(trace_length, ))  # S-phase probability

            for j in range(len(pdf)):                
                if normalize_pdf and pdf[j].max():      # normalize the prediction pdf
                    pdf[j] = pdf[j] / pdf[j].max()
                else:                                   # Non-normalize the prediction pdf ---recommendation
                    pdf[j] = pdf[j] 
                
                pdf_pick = get_picks_from_pdf(trace, pdf[j], phase=phases[j], height=height[j], distance=distance)   # picks list 
                pdf_picks.append(pdf_pick)
            stream.pdf = pdf 

            # Upate the pick information in pdf_picks for P-phase: pdf_picks[0]
            _upate_stream_picks(stream, pdf_picks[0], pick_phase='P')
            # PUpate the pick information in pdf_picks for P-phase for S-phase: pdf_picks[1]
            _upate_stream_picks(stream, pdf_picks[1], pick_phase='S')  
            
            # Store the result of the prediction for a stream
#             streamId = trace.get_id()[:trace.get_id().rfind('.')-1]
            ids = trace.get_id().split('.')
            streamId = '%s.%s' % (ids[0], ids[1])  
            time_stamp = trace.stats.starttime.isoformat()
            filePred = dataset_output_dir + '/' + time_stamp + streamId + ".pkl"     # The name of the prediction 
            
            net = trace.stats.network
            sta = trace.stats.station
#             ch = trace.stats.channel
            sTime = trace.stats.starttime.isoformat()
            eTime = trace.stats.endtime.isoformat()
            for j, picks in enumerate(pdf_picks):     # pdf_picks[0] - P-phase picks; pdf_picks[1] - S-phase picks
                for pick in picks: 
                    predResult['stream_name'].append(filePathStream)
                    predResult['stream_name_pred'].append(filePred)                
                    predResult['stream_id'].append(streamId)
                    predResult['network'].append(net)
                    predResult['station'].append(sta)
    #                 predResult['channel'].append(ch)
                    predResult['stream_start_time'].append(sTime)            
                    predResult['stream_end_time'].append(eTime)
                    predResult['phase'].append(phases[j])         
                    predResult['arrival_time'].append(pick.time)
                    predResult['pdf_max'].append(pdf[j].max())  
                    predResult['evaluation_mode'].append(pick.evaluation_mode)
                    predResult['evaluation_status'].append(pick.evaluation_status)  
                    if pick.time_errors.uncertainty is not None:
                        predResult['time_errors'].append(pick.time_errors.uncertainty)                    
                    else:
                        predResult['time_errors'].append(-999)
                    if pick.prob: 
                        predResult['probability'].append(pick.prob)
                    else:
                        predResult['probability'].append(-999)

                    pickPoint = int((pick.time - trace.stats.starttime) * trace.stats.sampling_rate + 1)
                    snr_db = get_snr(trace.data, pickPoint)
                    predResult['snr_db'].append(snr_db)   

            stream.picks['P'].extend(pdf_picks[0])
            stream.picks['S'].extend(pdf_picks[1])
            time_stamp = trace.stats.starttime.isoformat()
            stream.write(filePred, format="PICKLE")
            pbar.update()
                 
    # Store the summary of prediction 
    dfPred = pd.DataFrame(data=predResult)
    fileOut = "prediction_results.csv"
    dfPred.to_csv(os.path.join(dataset_output_dir, fileOut), index=False)       
    
        
def _upate_trace_picks(trace, phase_pdf_picks, pick_phase='P'):
    """
    Upate the pick information in phase_pdf_picks, depending on the trace and the phase name.
    
    :param trace:
    :param phase_pdf_picks: The picks from the pdf value to be update information for a phase.
    :param pick_phase: The specified phase name, i.e. 'P' or 'S'.
    """
     # Upate the P-phase pick information of the trace 
#     _upate_trace_picks(trace, pick_phase='P', pdf_picks[0])
    
    if trace.picks and trace.picks[pick_phase]:
        for val_pick in  trace.picks[pick_phase]:
            for pre_pick in phase_pdf_picks:
                pre_pick.phase_hint = pick_phase
                pre_pick.evaluation_mode = "automatic"

                residual = get_time_residual(val_pick, pre_pick, delta=5.0)
                pre_pick.time_errors = QuantityError(residual)

                if is_close_pick(val_pick, pre_pick, delta=0.1):
#                 if is_close_pick(val_pick, pre_pick, delta=0.2):
#                 if is_close_pick(val_pick, pre_pick, delta=0.3):
#                 if is_close_pick(val_pick, pre_pick, delta=0.4):
#                 if is_close_pick(val_pick, pre_pick, delta=0.5):                    
                    pre_pick.evaluation_status = "confirmed"                    
                elif is_close_pick(val_pick, pre_pick, delta=1):
                    pre_pick.evaluation_status = "rejected"

    else:
        trace.picks[pick_phase] = []
        for pre_pick in phase_pdf_picks:
            pre_pick.phase_hint = pick_phase
            pre_pick.evaluation_mode = "automatic"

            

def _upate_stream_picks(stream, phase_pdf_picks, pick_phase='P'):
    """
    Upate the pick information in phase_pdf_picks, depending on the trace and the phase name.
    
    :param stream: The stream including the 3 channels.
    :param phase_pdf_picks: The picks from the pdf value to be update information for a phase.
    :param pick_phase: The specified phase name, i.e. 'P' or 'S'.
    """
    if stream.picks and stream.picks[pick_phase]:
        for val_pick in  stream.picks[pick_phase]:
            for pre_pick in phase_pdf_picks:
                pre_pick.phase_hint = pick_phase
                pre_pick.evaluation_mode = "automatic"

                residual = get_time_residual(val_pick, pre_pick, delta=5.0)
                pre_pick.time_errors = QuantityError(residual)
                
                if is_close_pick(val_pick, pre_pick, delta=0.1):
#                 if is_close_pick(val_pick, pre_pick, delta=0.2):
#                 if is_close_pick(val_pick, pre_pick, delta=0.3):
#                if is_close_pick(val_pick, pre_pick, delta=0.4):
#                 if is_close_pick(val_pick, pre_pick, delta=0.5):                       
                    pre_pick.evaluation_status = "confirmed"
                elif is_close_pick(val_pick, pre_pick, delta=1):
                    pre_pick.evaluation_status = "rejected"

    else:
        stream.picks[pick_phase] = []
        for pre_pick in phase_pdf_picks:
            pre_pick.phase_hint = pick_phase
            pre_pick.evaluation_mode = "automatic"
            
    
    
def is_close_pick(validate_pick, predict_pick, delta=0.1):
    pick_upper_bound = predict_pick.time + delta
    pick_lower_bound = predict_pick.time - delta
    if pick_lower_bound < validate_pick.time < pick_upper_bound:
        return True
    else:
        return False

    

def get_time_residual(val_pick, pre_pick, delta=0.5):
    if is_close_pick(val_pick, pre_pick, delta=delta):
        residual = val_pick.time - pre_pick.time
        return residual
