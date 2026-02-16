import matplotlib.pyplot as plt
import numpy as np
import wfdb
import pandas as pd
import time

from scipy.io import savemat
from sktime.forecasting.arima import AutoARIMA
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
print(f'File considered (without pacemaker): {len(lista)}.\n')

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
window = 30
signal_length = pre + post
lead = 0  # we'll save only lead I

for number in lista:
    record = wfdb.rdsamp(f'/home/mscanu/mit-bih-arrhythmia-database-1.0.0/{number}')
    annotation = wfdb.rdann(f'/home/mscanu/mit-bih-arrhythmia-database-1.0.0/{number}', 'atr')
    for center, symbol in zip(annotation.sample, annotation.symbol):
        if center > pre:
            tmp = record[0][center - pre: center + post, lead]
            if len(tmp) == signal_length and symbol in symbol_mapping:
                data.append([tmp, symbol_mapping[symbol], '0'])

# I want to save it into a pandas dataframe

columns = ["Lead_I", "Class", "Augmented"]
df = pd.DataFrame(data, columns=columns)
df.index = np.arange(len(df))

print('Original class split: \n', df.value_counts("Class"))

def creation(data_, classe_, forecaster):

    # separo i picchi dal resto

    pre_ = 100
    post_ = 100
    signal_length_ = pre_ + post_
    window_ = 30

    scelti = data_["Lead_I"].explode(ignore_index=True)
    mask_rest = np.ones_like(scelti, dtype=bool)
    mask_peaks = np.zeros_like(scelti, dtype=bool)
    for ii in range(n):
        idx = signal_length_ * ii + pre_
        start = int(idx - window_ / 2)
        end = int(idx + window_ / 2)
        mask_rest[start:end] = False
        mask_peaks[start:end] = True

    y_rest = pd.Series(scelti[mask_rest].values, dtype='float64')
    y_peaks = pd.Series(scelti[mask_peaks].values, dtype='float64')

    train_peaks, test_peaks = temporal_train_test_split(y_peaks, test_size=window_)
    train_rest, test_rest = temporal_train_test_split(y_rest, test_size=(signal_length_ - window_))

    fh_peaks = np.arange(1, window_ + 1)

    # forecaster

    forecaster.fit(train_peaks)
    y_pred = forecaster.predict(fh=fh_peaks)

    # it is needed to recompose the beat

    noise_1 = pd.Series(np.random.normal(0, 0.005,
                        size=(signal_length_ - window_)), index=test_rest.index)
    rest_noised = pd.Series(test_rest + noise_1).reset_index(drop=True)

    # what if there is a discontinuity?

    pre_window = int(window / 2)

    salto_1 = rest_noised[pre_ - pre_window - 1] - y_pred.reset_index(drop=True)[0]

    # noised artificial beats

    segnale_nuovo = np.concatenate((rest_noised[: pre_ - pre_window].values,
                                    y_pred.values + salto_1))
    
    salto_2 = rest_noised[pre_ - pre_window] - segnale_nuovo[- 1]

    segnale_nuovo = np.concatenate((segnale_nuovo,
                                    rest_noised[pre_ - pre_window :].values - salto_2))

    # recompose one beat

    real = pd.concat([test_rest[: pre_ - int(window / 2)],
                      test_peaks,
                      test_rest[pre_ - int(window / 2):]],
                      ignore_index=True)

    return [segnale_nuovo, real.values]


# creation

n = 8
sampling_arima = 600
arima_dtw = 0
arima_was = 0
arima_cos = 0
arima_fre = 0
time_passed_arima = 0
arima = AutoARIMA(sp=window, suppress_warnings=True, maxiter=25, n_jobs=-1)
classi = ['V', 'F', 'S']

aug_arima = []

for i, classe in enumerate(classi):
	df_class = df[df["Class"] == classe]
	for numero in range(sampling_arima):
		np.random.seed(None)
		estrazione = df_class.sample(n=n)
		start_time = time.perf_counter()
		[segnale_nuovo, real] = creation(data_=estrazione, classe_=classe, forecaster=arima)
		end_time = time.perf_counter()
		
		elapsed_time = end_time - start_time
		time_passed_arima = time_passed_arima + elapsed_time
		aug_arima.append([segnale_nuovo, classe, '1'])
		
		# metrics for the peak only

		arima_dtw = arima_dtw + dtw(segnale_nuovo, real, be="numpy")
		arima_was = arima_was + wasserstein_distance(segnale_nuovo, real)
		arima_cos = arima_cos + cosine_similarity([segnale_nuovo], [real])[0][0]
		arima_fre = arima_fre + frdist([segnale_nuovo], [real])

		if numero % 100 == 0:
			
			den = (i + 1) * (numero + 1)
			output = "output_arima.txt"
			with open(output, "w") as f:

				print(f"AVERAGE TIME FOR CREATING A BEAT WITH ARIMA: {round(time_passed_arima / den, 3)} seconds.", file=f)
				print('', file=f)
				print('AVERAGE METRICS SUMMARY FOR ARIMA', file=f)
				print('', file=f)
				print('DTW:\t\t ', round(arima_dtw / den, 5), file=f)
				print('', file=f)
				print('WASSERSTEIN DISTANCE:\t\t ', round(arima_was / den, 5), file=f)
				print('', file=f)
				print('COSINE SIMILARITY:\t\t ', round(arima_cos / den, 5), file=f)
				print('', file=f)
				print('FRECHET DISTANCE:\t\t ', round(arima_fre / den, 5), file=f)
				print('', file=f)

				df_aug_arima = pd.DataFrame(aug_arima, columns=columns)
				dict_aug_arima = {"dataframe": df_aug_arima.to_dict("list")}
				savemat("beats_arima.mat", dict_aug_arima)



den = len(classi) * sampling_arima

with open(output, "w") as f:

    print(f"AVERAGE TIME FOR CREATING A BEAT WITH ARIMA: {round(time_passed_arima / den, 3)} seconds.", file=f)
    print('', file=f)
    print('AVERAGE METRICS SUMMARY FOR ARIMA', file=f)
    print('', file=f)
    print('DTW:\t\t ', round(arima_dtw / den, 5), file=f)
    print('', file=f)
    print('WASSERSTEIN DISTANCE:\t\t ', round(arima_was / den, 5), file=f)
    print('', file=f)
    print('COSINE SIMILARITY:\t\t ', round(arima_cos / den, 5), file=f)
    print('', file=f)
    print('FRECHET DISTANCE:\t\t ', round(arima_fre / den, 5), file=f)
    print('', file=f)

df_aug_arima = pd.DataFrame(aug_arima, columns=columns)
dict_aug_arima = {"dataframe": df_aug_arima.to_dict("list")}
savemat("beats_arima.mat", dict_aug_arima)
