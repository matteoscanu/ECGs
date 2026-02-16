import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import wfdb

from scipy.io import savemat
from sktime.forecasting.ets import AutoETS
from sktime.split import temporal_train_test_split
from sktime.utils.plotting import plot_series

from frechetdist import frdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from tslearn.metrics import dtw

lista = {"100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
         "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
         "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
         "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
         "222", "223", "228", "230", "231", "232", "233", "234"}

paced = {"102", "104", "107", "217"}

lista = list(lista - paced)
print(f'File considerati (senza pacemaker): {len(lista)}.\n')

# save everything into a dataframe

symbol_mapping = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    'Q': 'Q'
}

data = []
pre = 100
post = 100
signal_length = pre + post
lead = 0  # we'll save only lead I

for number in lista:
    record = wfdb.rdsamp(f'/home/mscanu/mit-bih-arrhythmia-database-1.0.0/{number}')
    annotation = wfdb.rdann(f'/home/mscanu/mit-bih-arrhythmia-database-1.0.0/{number}',
                            'atr')
    for center, symbol in zip(annotation.sample, annotation.symbol):
        if center > pre:
            tmp = record[0][center - pre : center + post, lead]
            if len(tmp) == signal_length and symbol in symbol_mapping:
                data.append([tmp, symbol_mapping[symbol], '0'])

# I want to save it into a pandas dataframe

columns = ["Lead_I", "Class", "Augmented"]
df = pd.DataFrame(data, columns=columns)
df.index = np.arange(len(df))

print('Original class split: \n', df.value_counts("Class"))


def creation(data_, class_, forecaster_, pre_peak, post_peak, pre_window, post_window):

    # sepate peaks from the rest of the beat

    signal_length_ = pre_peak + post_peak
    window_ = pre_window + post_window

    chosen = data_["Lead_I"].explode(ignore_index=True)
    mask_rest = np.ones_like(chosen, dtype=bool)
    mask_peaks = np.zeros_like(chosen, dtype=bool)
    for ii in range(n):
        idx = signal_length * ii + pre
        start = int(idx - pre_window)
        end = int(idx + post_window)
        mask_rest[start:end] = False
        mask_peaks[start:end] = True

    y_rest = pd.Series(chosen[mask_rest].values, dtype='float64')
    y_peaks = pd.Series(chosen[mask_peaks].values, dtype='float64')

    train_peaks, test_peaks = temporal_train_test_split(y_peaks, test_size=window_)
    train_rest, test_rest = temporal_train_test_split(y_rest, test_size=(signal_length_ - window_))

    fh_peaks = np.arange(1, window_ + 1)

    # forecaster

    forecaster_.fit(train_peaks)
    y_pred = forecaster_.predict(fh=fh_peaks)
    
    # it is needed to recompose the beat

    noise_1 = pd.Series(np.random.normal(0, 0.005,
                        size=(signal_length_ - window_)), index=test_rest.index)
    rest_noised = pd.Series(test_rest + noise_1).reset_index(drop=True)

    # what if there is a discontinuity?

    jump_1 = rest_noised[pre_peak - pre_window - 1] - y_pred.reset_index(drop=True)[0]

    # noised artificial beats

    new_signal = np.concatenate((rest_noised[: pre_peak - pre_window].values,
                                    y_pred.values + jump_1))
    
    jump_2 = rest_noised[pre_peak - pre_window] - new_signal[- 1]

    new_signal = np.concatenate((new_signal,
                                    rest_noised[pre_peak - pre_window :].values - jump_2))         
                                    

    # recompose the beat

    real = pd.concat([test_rest[: pre_peak - pre_window],
                      test_peaks,
                      test_rest[pre_peak - pre_window:]],
                      ignore_index=True)

    # create an augmented beat like in the paper in order to measure them and 
    # the ones augmented

    noise_2 = pd.Series(np.random.normal(0, np.sqrt(0.05), size=signal_length))
    fake = real.values + noise_2

    return [new_signal, fake.values, real.values]


# creation

sampling_ets = 5
pre_window = 25
post_window = 35
classi = ['V', 'F', 'S']
ets_dtw = 0
fake_dtw = 0
ets_was = 0
fake_was = 0
ets_cos = 0
fake_cos = 0
ets_fre = 0
fake_fre = 0
time_passed_ets = 0
ets = AutoETS(auto=False, error="add", trend=None, seasonal="add",
              sp=(pre_window + post_window), information_criterion="aic", n_jobs=-1)
aug_ets = []

for classe in classi:
    df_class = df[df["Class"] == classe]

    if classe == 'V':
        n = 5
    else:
        n = 10

    for number in range(sampling_ets):
        np.random.seed(None)
        extracted = df_class.sample(n=n)
        start_time = time.perf_counter()
        [new_signal, fake, real] = creation(data_=extracted, class_=classe, 
                                               forecaster_=ets, 
                                               pre_peak=pre, post_peak=post, 
                                               pre_window=pre_window, 
                                               post_window=post_window)
        end_time = time.perf_counter()

        if number <= 2:
        
            plt.figure()
            plot_series(pd.Series(new_signal), pd.Series(real))
            plt.legend(["Generated", "Real"])
            title = str(number) + " ets generated vs real beat of class " + classe + ".jpg"
            plt.savefig(title, format="jpg")
            
        if number == 0:
        
            plt.figure()
            plot_series(pd.Series(fake), pd.Series(real))
            plt.legend(["Augmented", "Real"])
            title = "augmented vs real beat of class " + classe + ".jpg"
            plt.savefig(title, format="jpg")
        
        elapsed_time = end_time - start_time
        time_passed_ets = time_passed_ets + elapsed_time
        aug_ets.append([new_signal, classe, '1'])
        ets_dtw = ets_dtw + dtw(new_signal, real, be="numpy")
        fake_dtw = fake_dtw + dtw(fake, real, be="numpy")
        ets_was = ets_was + wasserstein_distance(new_signal, real)
        fake_was = fake_was + wasserstein_distance(fake, real)
        ets_cos = ets_cos + cosine_similarity([new_signal], [real])[0][0]
        fake_cos = fake_cos + cosine_similarity([fake], [real])[0][0]
        ets_fre = ets_fre + frdist([new_signal], [real])
        fake_fre = fake_fre + frdist([fake], [real])

den = len(classi) * sampling_ets

output = "output_ets.txt"

with open(output, "w") as f:

    print(f"AVERAGE TIME FOR CREATING A BEAT WITH ETS: {round(time_passed_ets / den, 3)} seconds.", file=f)
    print('', file=f)
    print('AVERAGE METRICS SUMMARY FOR ETS', file=f)
    print('', file=f)
    print('DTW:\t\t ', round(ets_dtw / den, 5), file=f)
    print('', file=f)
    print('WASSERSTEIN DISTANCE:\t\t ', round(ets_was / den, 5), file=f)
    print('', file=f)
    print('COSINE SIMILARITY:\t\t ', round(ets_cos / den, 5), file=f)
    print('', file=f)
    print('FRECHET DISTANCE:\t\t ', round(ets_fre / den, 5), file=f)
    print('', file=f)

df_aug_ets = pd.DataFrame(aug_ets, columns=columns)
dict_aug_ets = {"dataframe": df_aug_ets.to_dict("list")}
savemat("beats_ets.mat", dict_aug_ets)
