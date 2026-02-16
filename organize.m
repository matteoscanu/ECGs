% this function purpose is to take the dataset and re-order all the
% beats such that they periodically follows each other (N-V-S-F):
% in this way a manual k-fold in the classification process can
% be performed: beats will be sampled randomly, so no concerns
% about any bias of the classifier.

function [I_new, ann_new, aug_new] = organize(I, ann, aug, seed)

    I_new = zeros(size(I, 1), size(I, 2));
    ann_new = ann;
    aug_new = aug;
    num_class = length(unique(ann));
    totale = length(ann);
    epc = totale / num_class;
    rng(seed);
    for i = 1 : num_class
        temp_I = I((i - 1) * epc + 1 : i * epc, :);
        indici = randperm(totale / num_class)';
        temp_I = temp_I(indici, :);
        I_new(i : num_class : totale, :) = temp_I;
        ann_new(i : num_class : totale) = ann((i - 1) * epc + 1 : i * epc);
        aug_new(i : num_class : totale) = aug((i - 1) * epc + 1 : i * epc);
    end
end
