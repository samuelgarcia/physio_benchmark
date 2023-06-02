import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pprint import pprint

import physio




base_folder = Path('/home/samuel/Documents/physio_benchmark/dataset_716/experiment_data/')

cases = ['hand_bike', 'jogging', 'maths', 'sitting', 'walking']


def plot_one_subject(subject_id, case):
    folder = base_folder / subject_id / case
    data = np.loadtxt(folder / 'ECG.tsv')
    
    r_peak0 = np.loadtxt(folder / 'annotation_cs.tsv', dtype='int64')
    r_peak1 = np.loadtxt(folder / 'annotation_cables.tsv', dtype='int64')
    print(r_peak0)
    print(r_peak1)
    
    fig, axs = plt.subplots(nrows=data.shape[1], sharex=True)
    for i in range(data.shape[1]):
        ax = axs[i]
        ax.plot(data[:, i])
    axs[0].set_title(f'{subject_id} {case}')
    
    axs[0].scatter(r_peak0, data[r_peak0, 0], marker='o', color='m')
    axs[1].scatter(r_peak1, data[r_peak1, 1], marker='o', color='m')
    
    plt.show()


def plot_many_subject():
    for i in range(25):
        #~ for case in cases:
        for case in [ 'jogging']:
            subject_id = f'subject_{i:02}'
            plot_one_subject(subject_id, case)


def count_matching_peak(times1, times2, delta=10):
    times_concat = np.concatenate((times1, times2))
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where((diffs <= delta) & (membership_sorted[:-1] != membership_sorted[1:]))[0]
    if len(inds) == 0:
        return 0
    inds2 = np.where(inds[:-1] + 1 != inds[1:])[0]
    return len(inds2) + 1

def compute_perf(times1, times2, delta=10):
    if times1.size == 0:
        return
    if times2.size == 0:
        return
    
    tp = count_matching_peak(times1, times2, delta=delta)
    fn = times1.size - tp
    fp = times2.size - tp
    
    perfs = pd.Series(dtype=float)
    perfs['tp'] = tp
    perfs['fn'] = fn
    perfs['fp'] = fp
    perfs['accuracy'] = tp / (tp + fp + fn)
    perfs['precision'] = tp / (tp + fp)
    perfs['recall'] = tp / (tp + fn)
    
    
    return perfs
    

def benchmark_one_subject(subject_id, case):
    folder = base_folder / subject_id / case
    data = np.loadtxt(folder / 'ECG.tsv')
    if (folder / 'annotation_cs.tsv').exists():
        true_r_peak = np.loadtxt(folder / 'annotation_cs.tsv', dtype='int64')
    else:
        true_r_peak = None
    
    # this is chest_strap_V2_V1
    raw_ecg = data[:, 0]
    srate = 250.
    
    delta = int(srate*0.1)
    #~ print('delat', delta)

    from ecgdetectors import Detectors
    detectors = Detectors(srate)

    
    r_peaks = detectors.hamilton_detector(raw_ecg)
    #~ r_peaks = detectors.christov_detector(raw_ecg)
    #~ r_peaks = detectors.engzee_detector(raw_ecg)
    #~ r_peaks = detectors.pan_tompkins_detector(raw_ecg)
    #~ r_peaks = detectors.swt_detector(raw_ecg)
    #~ r_peaks = detectors.two_average_detector(raw_ecg)
    #~ r_peaks = detectors.matched_filter_detector(raw_ecg,template_file)
    #~ r_peaks = detectors.wqrs_detector(raw_ecg)
    r_peaks = np.asarray(r_peaks)
    
    parameters = physio.get_ecg_parameters('shuman_ecg')
    #~ pprint(parameters)
    ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameters=parameters)

    if true_r_peak is not None:
        #~ n = count_matching_peak(true_r_peak, r_peaks, delta=delta)
        # perfs = compute_perf(true_r_peak, r_peaks, delta=delta)

        #~ n = count_matching_peak(true_r_peak, ecg_peaks, delta=delta)
        perfs = compute_perf(true_r_peak, ecg_peaks, delta=delta)
        return perfs
    
    #~ fig, axs = plt.subplots(nrows=2, sharex=True)
    #~ ax = axs[0]
    #~ ax.plot(raw_ecg)
    #~ if true_r_peak is not None:
        #~ ax.scatter(true_r_peak, raw_ecg[true_r_peak], marker='o', color='g', s=50)
    #~ ax.scatter(r_peaks, raw_ecg[r_peaks], marker='o', color='r', s=30)
    
    #~ ax = axs[1]
    #~ ax.plot(ecg)
    #~ ax.scatter(ecg_peaks, ecg[ecg_peaks], marker='o', color='m')
    
    
    #~ axs[0].set_title(f'{subject_id} {case}')
    #~ plt.show()


def benchmark_many_subject():
    #~ all_perfs = pd.DataFrame(dtype='float')
    all_perfs = []
    index = []
    for i in range(25):
        for case in cases:
        #~ for case in [ 'jogging']:
            subject_id = f'subject_{i:02}'
            #~ print(subject_id)
            perfs = benchmark_one_subject(subject_id, case)
            if perfs is None:
                continue
            all_perfs.append(pd.DataFrame(perfs).T)
            index.append((subject_id, case))
    all_perfs = pd.concat(all_perfs, axis=0)
    all_perfs.index = pd.MultiIndex.from_tuples(index, names=['subject_id', 'case'])
    return all_perfs
    

if __name__ == '__main__':
    #~ subject_id = 'subject_20'
    #~ plot_one_subject('subject_20', 'maths')
    
    #~ plot_many_subject()
    
    #~ benchmark_one_subject('subject_01', 'jogging')
    
    benchmark_many_subject()

