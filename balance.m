% this function is needed in order to extract the exact number of beats from each
% class. for example one class (like 'N') might have too many beats, so one can
% use this function to extract the number of beats needed.

function [new_data, new_labels, aug_new] = balance(data, labels, aug, class, num)
    
    all = find(labels == class);
    index = datasample(all, num);
    new_data = data(index, :);
    new_labels = repmat(class, num, 1);
    aug_new = aug(index);
    
end  
