import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import wfdb
 
from scipy.io import savemat
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.split import temporal_train_test_split
from sktime.utils import plot_series
 
from frechetdist import frdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from tslearn.metrics import dtw

# DATASET PREPROCESSING

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
    record = wfdb.rdsamp(f'./mit-bih-arrhythmia-database-1.0.0/{number}')
    annotation = wfdb.rdann(f'./mit-bih-arrhythmia-database-1.0.0/{number}',
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

    # separo i picchi dal resto

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

    forecaster_.fit(train_peaks, fh=fh_peaks)
    y_pred = forecaster_.predict()
    
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
                                    

    # recompose one beat

    real = pd.concat([test_rest[: pre_peak - pre_window],
                      test_peaks,
                      test_rest[pre_peak - pre_window:]],
                      ignore_index=True)

    return [new_signal, real.values]


# CREATION

classi = ['V', 'F', 'S']
sampling_model = 5000
model_dtw = 0
model_was = 0
model_cos = 0
model_fre = 0
time_passed_model = 0

aug_model = []

for i, classe in enumerate(classi):
		df_class = df[df["Class"] == classe]  
		  
		if classe == 'V':
			n = 6
			pre_window = 25
			post_window = 25
			model = NeuralForecastLSTM(learning_rate=0.0001, max_steps=500, batch_size=128)
		elif classe == 'F':
			n = 10
			pre_window = 25
			post_window = 25
			model = NeuralForecastLSTM(learning_rate=0.0005, max_steps=500, batch_size=128)
		elif classe =='S':
			n = 16
			pre_window = 15
			post_window = 15
			model = NeuralForecastLSTM(learning_rate=0.001, max_steps=500, batch_size=128)

		for numero in range(sampling_model):
				np.random.seed(None)
				extracted = df_class.sample(n=n)
				start_time = time.perf_counter()
				[new_signal, real] = creation(data_=extracted, class_=classe, 
								 forecaster_=model, 
								 pre_peak=pre, post_peak=post, 
								 pre_window=pre_window, 
								 post_window=post_window)
				end_time = time.perf_counter()

				elapsed_time = end_time - start_time
				time_passed_model = time_passed_model + elapsed_time
				aug_model.append([new_signal, classe, '1'])

				# metrics

				model_dtw = model_dtw + dtw(new_signal, real, be="numpy")
				model_was = model_was + wasserstein_distance(new_signal, real)
				model_cos = model_cos + cosine_similarity([new_signal], [real])[0][0]
				model_fre = model_fre + frdist([new_signal], [real])

				if numero % 1000 == 0:

					den = (i + 1) * (numero + 1)

					output = "output_lstm.txt"

					with open(output, "w") as f:

						print(f"AVERAGE TIME FOR CREATING A BEAT WITH LSTM: {round(time_passed_model / den, 3)} seconds.", file=f)
						print('', file=f)
						print('AVERAGE METRICS SUMMARY FOR LSTM', file=f)
						print('', file=f)
						print('DTW:\t\t ', round(model_dtw / den, 5), file=f)
						print('', file=f)
						print('WASSERSTEIN DISTANCE:\t\t ', round(model_was / den, 5), file=f)
						print('', file=f)
						print('COSINE SIMILARITY:\t\t ', round(model_cos / den, 5), file=f)
						print('', file=f)
						print('FRECHET DISTANCE:\t\t ', round(model_fre / den, 5), file=f)
						print('', file=f)

					df_aug_model = pd.DataFrame(aug_model, columns=columns)
					dict_aug_model = {"dataframe": df_aug_model.to_dict("list")}
					savemat("beats_lstm_1.mat", dict_aug_model)


# METRICS

den = len(classi) * sampling_model
output = "output_lstm.txt"

with open(output, "w") as f:

    print(f"AVERAGE TIME FOR CREATING A BEAT WITH LSTM: {round(time_passed_model / den, 3)} seconds.", file=f)
    print('', file=f)
    print('AVERAGE METRICS SUMMARY FOR ETS', file=f)
    print('', file=f)
    print('DTW:\t\t ', round(model_dtw / den, 5), file=f)
    print('', file=f)
    print('WASSERSTEIN DISTANCE:\t\t ', round(model_was / den, 5), file=f)
    print('', file=f)
    print('COSINE SIMILARITY:\t\t ', round(model_cos / den, 5), file=f)
    print('', file=f)
    print('FRECHET DISTANCE:\t\t ', round(model_fre / den, 5), file=f)
    print('', file=f)

df_aug_model = pd.DataFrame(aug_model, columns=columns)
dict_aug_model = {"dataframe": df_aug_model.to_dict("list")}
savemat("beats_lstm.mat", dict_aug_model)
