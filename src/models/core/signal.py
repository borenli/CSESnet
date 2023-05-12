"""
"""
import math

import numpy as np

import ipdb as pdb



def signal_preprocessing(data, mode='std', filter=False):
    """
    Preprocess the trace.
    
    :param data: The trace object.
    :pram mode: The normalization mode.
    """
#     pdb.set_trace()    
    data.detrend('demean')
#     data.data -= np.mean(data.data)
    data.detrend('linear')
    
    bandpass = [1, 45]
    if filter:
        data.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])    
    if mode == 'std':
        stdData = np.std(data.data)
        if stdData == 0:  stdData= 1.0
        data.data /= stdData   
    else:
        data.normalize()
#         maxData = np.max(abs(data.data))
#         if maxData == 0:  maxData= 1.0
#         data.data /= maxData
    data.resample(100)
    return data



def trim_stream(stream, points=3001):
    """
    Cut all traces of this Stream object to points from  its starttime.
    
    :param stream: Stream object.
    :param points: 
    """ 
    trace = stream[0]
    start_time =trace.stats.starttime
    dt = (trace.stats.endtime - trace.stats.starttime) / (trace.data.size - 1)
    end_time = start_time + dt * (points - 1)
    stream.trim(start_time, end_time, nearest_sample=False, pad=True, fill_value=0)
    for trace in stream:
        if not trace.data.size == points:
            time_stamp = trace.stats.starttime.isoformat()
            trName = time_stamp + trace.get_id()
            raise LengthError("Trace length is not correct: %s" % trName)
    return stream



def get_snr(data, pat, window=200):
    """ 
    Estimates SNR.
    
    Ref: EQTransformer
    :param data : numpy array,3 component data.    
    :param pat: Sample point where a specific phase arrives. 
    :param window: The length of the window for calculating the SNR (in the samples). default=200        
    :returns: snr, Estimated SNR in db.    
    """
#     pdb.set_trace()
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)         
        except Exception:
            pass
        
    return snr 


def get_snr_abs(data, pat, window=200):
    """ 
    Estimates SNRm using 95% percentile of data absolute values.
    
    Ref: EQTransformer
    :param data : numpy array,3 component data.    
    :param pat: Sample point where a specific phase arrives. 
    :param window: The length of the window for calculating the SNR (in the samples). default=200        
    :returns: snr, Estimated SNR in db.    
    """
#     pdb.set_trace()
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(abs(sw1),95)/np.percentile(abs(nw1),95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(abs(sw1),95)/np.percentile(abs(nw1),95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(abs(sw1),95)/np.percentile(abs(nw1),95))**2), 1)         
        except Exception:
            pass
        
    return snr 