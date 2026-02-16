% to download the whole dataset uncomment the following block.
% you can also download it directly in:
% aggiungi link per scaricare mit-bih dataset
% remember to change the directory!
%{
% uncomment the following line remembering to add your project directory:
% cd %ADD YOUR DIRECTORY HERE!
[old_path] = which('rdsamp'); 
if(~isempty(old_path))
    rmpath(old_path(1 : end - 8)); 
end
wfdb_url = 'https://physionet.org/physiotools/matlab/wfdb-app-matlab/wfdb-app-toolbox-0-10-0.zip';
filestr = websave('wfdb-app-toolbox-0-10-0.zip', wfdb_url);
unzip('wfdb-app-toolbox-0-10-0.zip');
cd mcode
addpath(pwd)
savepath
wfdbdemo
%}


% analyze one single file:

lista = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112", ...
         "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200", ...
         "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215", ...
         "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"];
% if you need all the beats, included the ones with a pacemaker, just comment the following two lines.
paced = ["102", "104", "107", "217"];
lista = setdiff(lista, paced);
seed = rng;
pre = 99;
post = 100;
sample_points = pre + post + 1;
fprintf("File considerati (quelli senza pacemaker): %d\n\n", length(lista))


% let's analyze one file and see what's inside it.

fff = "100";
path = "database\mitdb\" + fff;
[record, Fs] = rdsamp(char(path));
[ann, anntype] = rdann(char(path), 'atr');
index = 3;
k = ann(index);
tipo = anntype(index);
prova_1 = record(k - pre : k + post, 1);
prova_2 = record(k - pre : k + post, 2);
fprintf("Signal length: %d datapoints, or %.2f seconds.\n\n", ...
        sample_points, sample_points / Fs)
f1 = figure('Visible', 'off');
p1 = plot(prova_1);
hold on
grid on
xlabel("Sample points")
ylabel("Leads (mV)")
p2 = plot(prova_2);
legend([p1 p2], ["Lead I", "Lead II"])
sss = "wst paper - file " + fff + " beat number " + string(index);
hold off
saveas(f1, sss, 'jpg')
fprintf('Plot in figure 1: beat number %d of file %s, type %c.\n\n', ...
        index, fff, tipo)


% now let's save everything we need
        
% i suggest to uncomment this portion of the code the first time you run it,
% then comment it again, since it is much faster to load the saved 
% dataset: those beats do not change and it takes more than five minutes to 
% repeat the operation each time.
%{
% if for some reasons you need the original label of each beat and not their 
% corresponding ANSI:AMII label, comment these lines:
symbol_mapping = containers.Map({'N', 'L', 'R', 'e', 'j', ...
                                 'A', 'a', 'J', 'S', ...
                                 'V', 'E', ...
                                 'F', ...
                                 'Q', 'f', '/'}, ...
                                {'N', 'N', 'N', 'N', 'N', ...
                                 'S', 'S', 'S', 'S', ...
                                 'V', 'V', ...
                                 'F', ...
                                 'Q', 'Q', 'Q'});

total = 360000;
I = zeros(total, sample_points);
annotations = blanks(total)';
k = 1;
ttime = tic;
for i = 1 : length(lista)
    number = lista(i);
    [record, Fs] = rdsamp(char('database\mitdb\' + number));
    [ann, anntype] = rdann(char('database\mitdb\' + number), 'atr');
    for j = 1:length(ann)
        center = ann(j);
        symbol = anntype(j);
        if center > pre && center < length(record(:, 1)) - post && isKey(symbol_mapping, symbol)
            tmp1 = record(center - pre : center + post, 1);
            I(k, :) = tmp1;
            annotations(k) = symbol_mapping(symbol);
            k = k + 1;
        end
    end
end
I = I(1 : k - 1, :);
annotations = annotations(1 : k - 1);
num = length(annotations);
tend = toc(ttime);
fprintf('Number of beats for each class (pre-augmentation):\n')
classCounter(annotations)
fprintf('Total number of beats: %d.\n\n', num)
fprintf('Time spent to import the dataset: %.3f seconds.\n\n', tend)
%}


% drop all q-type beats

num = length(I);
q_index = find(annotations == 'Q');
x_q = setdiff((1 : num), q_index);
I = I(x_q, :);
num = length(I);
annotations = annotations(x_q);
augmented = zeros(num, 1);
fprintf('Number of beats for each class (after dropping q-labelled beats):\n')
classCounter(annotations)
fprintf('Total number of beats: %d.\n\n', num)
classes = unique(annotations);


% execute the following block of lines only if you want to use the naive augmentation process,
% otherwise use whatever type of data augmentation you need to and then go ahead.
%{
num_per_class = int(total / length(classes))
[I_V, annotations_V, augmented_V] = augmentation(I, annotations, 'V', ...
                                                 num_per_class, sample_points)
[I_S, annotations_S, augmented_S] = augmentation(I, annotations, 'S', ...
                                                 num_per_class, sample_points)
[I_F, annotations_F, augmented_F] = augmentation(I, annotations, 'F', ...
                                                 num_per_class, sample_points)
%}


% if you used any time-series forecasting method please load here what you obtained and then move
% onwards. comment the following block of lines only if the naive forecasting procedure is being
% used.

load_augmented = load('beats_rnn.mat');
df_augmented = load_augmented.dataframe;
[classi, ~, idx] = unique(df_augmented.Class);
                       
conteggio = accumarray(idx, 1);
num_per_class = conteggio(1);
I = [I; df_augmented.Lead_I];
annotations = [annotations; df_augmented.Class];
augmented = [augmented; df_augmented.Augmented];


% now it is important to balance each class, and take the same amount of beats for each
% class. to do that, use function 'balance'. at the end function 'organize' casually
% order the beats so that manually creating a k-fold results very easy.
% please don't comment the following block of lines.

[I_N, ann_N, aug_N] = balance(I, annotations, augmented, 'N', num_per_class);
[I_V, ann_V, aug_V] = balance(I, annotations, augmented, 'V', num_per_class);
[I_S, ann_S, aug_S] = balance(I, annotations, augmented, 'S', num_per_class);
[I_F, ann_F, aug_F] = balance(I, annotations, augmented, 'F', num_per_class);
[I_new, annotations_new, augmented_new] = organize([I_N; I_V; I_S; I_F], ...
                                                   [ann_N; ann_V; ann_S; ann_F], ...
                                                   [aug_N; aug_V; aug_S; aug_F], seed);


% to save some time, the following lines allow to save and load the dataset
% and the label of each beat: they are already centered in its r-peak.
% at this moment i put it here so that you save the dataset already augmented, but if for some
% reasons you need just the original one you can put this block before the 'remove q-index block'.
save('dataset.mat', 'I')
save('classes.mat', 'annotations')
%{
loading_dataset = load('dataset.mat');
I = loading_dataset.I;
loading_classi = load('classes.mat');
annotations = loading_classes.annotations;
%}

fprintf('Number of beats for each class (post-augmentation):\n')
classCounter(annotations_new)
fprintf('Total number of beats: %d.\n\n', num)



% let the classification process start. the following three lines are needed for
% knn and k-fold: you can modify them, but please don't comment them!

k_fold = 10; % number of folds for cross-validation
n = 4; % this is for knn
ts = totale / k_fold;


% a first k-fold classification process with no wst is executed here. it is very slow
% and it was only used to note the massive performance gain when using transformed data.
% uncomment only if you need it.
%{
ttt = datestr(clock, 'YYYY/mm/dd HH:MM:SS');
y_pred = char(zeros(ts, k_fold));
y_test = char(zeros(ts, k_fold));
parpool('local');
start_time = tic;
fprintf('\nWe now begin classification without WST. Date and time: %s.\n\n', ttt)
for i = 1 : k_fold
    idx_test = (i - 1) * ts + 1 : i * ts;
    X_test = I(idx_test, :);
    y_test(:, i) = annotations_new(idx_test);
    idx_train = setdiff((1 : totale), idx_test);
    X_train = I(idx_train, :);
    y_train = annotations_new(idx_train);
    neigh = fitcknn(X_train, y_train, ...
                    'NumNeighbors', n, ...
                    'NSMethod', 'kdtree', ...
                    'Distance', 'euclidean', ...
                    'DistanceWeight', 'inverse');
    y_pred(:, i) = predict(neigh, X_test);
end
elapsed_time = toc(start_time);
delete(cgp('nocreate'));
fprintf('Execution time: %d minutes and %d seconds.\n', ...
        floor(elapsed_time / 60), ...
        floor(60 * (elapsed_time - floor(elapsed_time))))
fprintf('Average execution time per each fold: %.3f seconds.\n\n', ...
        elapsed_time / k_fold)
fprintf('Results for %d-fold cross validation KNN mean accuracy without WST: \n', k_fold)
classificationReport(y_test, y_pred, classes);
%}


% let's transform data using wst

invariance = 0.5;
Q = [8 1];
sf = waveletScattering('SignalLength', sample_points, ...
                       'SamplingFrequency', Fs,  ...
                       'InvarianceScale', invariance, ...
                       'QualityFactors', Q);
display(sf)

% first we try wst on one signal: remember that the first beat in this 
% dataset is of type N.

x_wst = featureMatrix(sf, prova_1');
paths = size(x_wst, 1);
time_windows = size(x_wst, 2);

fprintf('Dimensions of a single beat after using WST:\n')
fprintf(' - number of paths -> %d;\n', paths)
fprintf(' - number of time windows -> %d;\n', time_windows)


% to reduce dimensionality, only one time-window is taken: the one with more
% information, which correspond to be the one with the highest peak.

massimi = max(x_wst, [], 1);
[peak, idx] = max(massimi);
fprintf('Average maximum values for each time window: %.4f\n', ...
        mean(massimi))
fprintf('Max values on time windows: %.4f\n', peak)
fprintf('Time window which yielded maximum value: %d.\n\n', ...
        idx);


% plot of each time window for one beat

f4 = figure('Visible', 'off');
for i = 1 : time_windows
    plot(x_wst(:, i))
    hold on
end
ylim([-0.3 0.25])
grid on
hold off
saveas(f4, 'wst paper - time_windows - normal beat', 'jpg')


% now let's transform the entire dataset

parpool('local');
sx = zeros(totale, paths, time_windows);
ttt = datestr(clock, 'YYYY/mm/dd HH:MM:SS');
fprintf('We now begin the transformation of the dataset. ')
fprintf('Date and time: %s.\n', ttt)
start_time_t = tic;
for i = 1 : totale
    sx(i, :, :) = featureMatrix(sf, I_new(i, :)');
end
elapsed_time_t = toc(start_time_t);
fprintf('Execution time: %d minutes and %d seconds.\n\n', ...
        floor(elapsed_time_t / 60), ...
        floor(60 * (elapsed_time_t - floor(elapsed_time_t))))


% if needed, you can save/load any transformed dataset here. 
% in this way, if already transformed everything, you can 
% basically start here with the classification process using 
% transformed data.

% load('transformed.mat')
% load('annotations.mat')
% load('aug.mat')
% save('transformed.mat', 'sx')
% save('annotations.mat', 'annotations')
% save('aug.mat', 'augmented')


% now look for the time window which yielded the highest peak.

fprintf('Dataset dimensions after using WST:\n')
fprintf(' - number of paths -> %d;\n', paths)
fprintf(' - number of time windows -> %d;\n', time_windows)
massimi = max(mean(sx(:, 2:end, :)));
[peak, idx] = max(massimi);
fprintf('Average maximum values for each time window: %.4f\n', ...
        mean(massimi))
fprintf('Max values on time windows: %.4f\n', peak)
fprintf('Time window which yielded maximum value: %d.\n\n', ...
        idx);
window = idx;


% transformed data classification (with KNN)

ttt = datestr(clock, 'YYYY/mm/dd HH:MM:SS');
fprintf('We now begin classification with WST. Date and time: %s.\n\n', ttt)
X_WST = sx(:, :, window);
y_pred = char(zeros(ts, k_fold));
y_test = char(zeros(ts, k_fold));
k = 4;
start_time = tic;
for i = 1 : k_fold
    idx_test = (i - 1) * ts + 1 : i * ts;
    X_test = X_WST(idx_test, :);
    y_test(:, i) = annotations_new(idx_test);
    idx_train = setdiff((1 : totale), idx_test);
    X_train = X_WST(idx_train, :);
    y_train = annotations_new(idx_train);
    neigh = fitcknn(X_train, y_train, ...
                    'NumNeighbors', k, ...
                    'NSMethod', 'kdtree', ...
                    'Distance', 'euclidean', ...
                    'DistanceWeight', 'inverse');
    y_pred(:, i) = predict(neigh, X_test);
end
elapsed_time = toc(start_time);
delete(gcp('nocreate'));
fprintf('Execution time: %d minutes and %d seconds.\n', ...
        elapsed_time)
fprintf('Average execution time per each fold: %.3f seconds.\n\n', ...
        elapsed_time / k_fold)
fprintf('Results for %d-fold cross validation using %d-Nearest Neighbours algorithm: \n\n', k_fold, k)
classificationReport(y_test, y_pred, classes);


% plots
% plot the filter banks

[fb, f, filterparams] = filterbank(sf);
f5 = figure('Visible', 'off');
subplot(2, 1, 1)
plot(f, fb{2}.psift)
xlim([0 Fs])
ylim([0 1.4])
grid on

subplot(2, 1, 2)
plot(f, fb{3}.psift)
xlim([0 Fs])
ylim([0 1.4])
grid on
xlabel('Frequency (Hz)')
saveas(f5, 'wst paper - filterbanks', 'jpg')

% scaling function vs first order filter

f6 = figure('Visible', 'off');
phi = ifftshift(ifft(fb{1}.phift));
psiL1 = ifftshift(ifft(fb{2}.psift(:, end)));
t = (- length(phi) / 2 : length(phi) / 2 - 1) .* 1 / Fs;
scalplt = plot(t, phi);
xline(- invariance / 2, 'k--');
xline(invariance / 2, 'k--');
xlim([-0.4 0.4])
ylim([-0.04 0.04])
hold on
grid on
xlabel("Seconds")
ylabel("Amplitude")
wavplt = plot(t,[real(psiL1) imag(psiL1)]);
legend([scalplt wavplt(1) wavplt(2)], ...
       ["Scaling Function","Wavelet-Real Part","Wavelet-Imaginary Part"])
hold off
saveas(f6, 'wst paper - scaling_function', 'jpg')

f7 = figure('Visible', 'off');
xline(- invariance / 2, 'k--');
xline(invariance / 2, 'k--');
psiL2 = ifftshift(ifft(fb{3}.psift(:, end)));
hold on
grid on
xlabel("Seconds")
ylabel("Amplitude")
wavplt = plot(t, [real(psiL2) imag(psiL2)]);
legend([wavplt(1) wavplt(2)], ...
       ["Wavelet-Real Part","Wavelet-Imaginary Part"])
hold off
saveas(f7, 'wst paper - filter_plot_second_order', 'jpg')

% we can even plot the scalograms for zeroth, first and second-order 
% coefficients!

[S, U] = scatteringTransform(sf, prova_1');

f8 = figure('Visible', 'off');
plot(S{1}.signals{1}, 'x-')
grid on
saveas(f8, 'wst paper - order_coefficient_0', 'jpg')

f9 = figure('Visible', 'off');
scattergram(sf, U, 'Filterbank', 1)
saveas(f9, 'wst paper - order_coefficient_1', 'jpg')

f10 = figure('Visible', 'off');
scattergram(sf, U, 'Filterbank', 2)
saveas(f10, 'wst paper - order_coefficient_2', 'jpg')


% final step: you can replicate the classification process with just the 
% three abnormal classes. if you don't need it, comment from here to the
% finish.

n_index = find(annotations_new == 'N');
xxx = length(sx);
x_n = setdiff((1 : xxx), n_index);
sx_alt = sx(x_n, :, :);
annotations_alt = annotations_new(x_n);
ttt = datestr(clock, 'YYYY/mm/dd HH:MM:SS');
fprintf('We now begin three-classes classification with WST. Date and time: %s.\n\n', ttt)
X_WST = sx_alt(:, :, window);
n = 10;
totale = size(X_WST, 1);
ts = totale / k_fold;
start_time = tic;
y_pred = char(zeros(ts, k_fold));
y_test = char(zeros(ts, k_fold));
for i = 1 : k_fold
    idx_test = (i - 1) * ts + 1 : i * ts;
    X_test = X_WST(idx_test, :);
    y_test(:, i) = annotations_alt(idx_test);
    idx_train = setdiff((1 : totale), idx_test);
    X_train = X_WST(idx_train, :);
    y_train = annotations_alt(idx_train);
    neigh = fitcknn(X_train, y_train, ...
                    'NumNeighbors', k, ...
                    'NSMethod', 'kdtree', ...
                    'Distance', 'euclidean', ...
                    'DistanceWeight', 'inverse');
    y_pred(:, i) = predict(neigh, X_test);
end
elapsed_time = toc(start_time);
fprintf('Execution time: %.3f seconds.\n', ...
        elapsed_time)
fprintf('Average execution time per each fold: %.3f seconds.\n\n', ...
        elapsed_time / k_fold)
fprintf('Results for %d-fold cross validation using %d-Nearest Neighbours algorithm: \n\n', k_fold, k)
classificationReport(y_test, y_pred, ['F'; 'S'; 'V']);
delete(gcp('nocreate'));
