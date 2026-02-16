% this function performs a naive augmentation for the beats:
% it is not particularly elegant because you have to call this
% function for each class (it is okay if you work for 3-4 classes,
% but probably becomes difficult if classes are more than ten...).
% in any case it is effective.

% INPUT:
% Lead_I -> ECG data matrix
% annotation -> labels vector
% class_label -> label of the class data augmentation will be performed
% target_count -> the amount of new beats that you want to generate
% sample_points -> lenght of each beat.

% OUTPUT:
% augmented_I -> new augmented ECG data
% augmentated_ann -> new labels (all the same)
% augmented -> a vector that keeps track on which beats are augmented and
% which are original.

function [augmented_I, augmented_ann, augmented] = ...
          augmentation(Lead_I, annotations, ...
          class_label, target_count, sample_points)

    % find every member of that class

    class_index = find(annotations == class_label);
    subset_I = Lead_I(class_index, :);
    subset_ann = annotations(class_index);
    current_count = size(subset_I, 1);
    original = repmat("0", current_count, 1);
    
    if current_count >= target_count
        return
    end

    num_to_generate = target_count - current_count;
    noise_I = sqrt(0.05) .* randn(num_to_generate, sample_points);
    augmented_signals_I = zeros(num_to_generate, sample_points);
    augmented_signals_ann = repmat(class_label, num_to_generate, 1);
    now = repmat("1", num_to_generate, 1);
    
    for i = 1 : num_to_generate
        kk = randsample(current_count, 1);
        original_signal_I = subset_I(kk, :);
        augmented_signals_I(i, :) = original_signal_I + noise_I(i, :);
    end

    augmented_I = [subset_I; augmented_signals_I];
    augmented_ann = [subset_ann; augmented_signals_ann];
    augmented = [original; now];

end
