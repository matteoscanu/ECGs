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
conteggioClassi(annotations)
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
fprintf('Number of beats for each class (pre-augmentation):\n')
conteggioClassi(annotations)
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

load_augmented_0 = load('beats_rnn.mat');
df_augmented_0 = load_augmented_0.dataframe;
[classi, ~, idx] = unique(df_augmented_0.Class);
                       
conteggio = accumarray(idx, 1);
num_per_class = conteggio(1);
I = [I; df_augmented_0.Lead_I];
annotations = [annotations; df_augmented_0.Class];
augmented = [augmented; df_augmented_0.Augmented];


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
conteggioClassi(annotations_new)
fprintf('Total number of beats: %d.\n\n', num)
