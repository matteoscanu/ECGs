clear 
close all
clc
%%% load('definitivo.mat')

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

% analyze one single file

lista = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112", ...
         "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200", ...
         "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215", ...
         "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"];
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
        
% i suggest to uncomment this portion of the code just the first time you run
% it, and then comment it again, since it is much faster to load the saved 
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

totale = 360000;
I = zeros(totale, sample_points);
annotations = blanks(totale)';
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
fprintf('Numero di battiti per ciascuna classe (pre-augmentation):\n')
conteggioClassi(annotations)
fprintf('Numero totale di battiti: %d.\n\n', num)
fprintf('Tempo impiegato per importare il dataset normalmente: %.3f secondi.\n\n', tend)
%}

% in order to save you some time, the following lines allow you to save and load the dataset
% and the label of each beat: they are already centered in its r-peak.
save('dataset.mat', 'I')
save('classes.mat', 'annotations')
%{
loading_dataset = load('dataset.mat');
I = loading_dataset.I;
loading_classi = load('classes.mat');
annotations = loading_classes.annotations;
%}

% drop all q-type beats

num = length(I);
q_index = find(annotations == 'Q');
x_q = setdiff((1 : num), q_index);
I = I(x_q, :);
num = length(I);
annotations = annotations(x_q);
augmented = zeros(num, 1);
fprintf('Numero di battiti per ciascuna classe (pre-augmentation):\n')
conteggioClassi(annotations)
fprintf('Numero totale di battiti: %d.\n\n', num)
classes = unique(annotations);
%%%clear ann anntype fff k lista loading_classi loading_dataset paced prova_2
%%%clear center q_index record sss tipo x_q
