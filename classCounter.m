% this file counts the beats for each class, printing the results.

function classCounter(y)
    
    y_tmp = char(y);
    [classes, ~, idx] = unique(y_tmp);
    counter = accumarray(idx, 1);
    per = (counter / sum(counter)) * 100;
    for kk = 1 : length(classes)
        fprintf('%c \t %6d \t (%.4f%%)\n', char(classes(kk)), ...
                                           counter(kk), ...
                                           per(kk))
    end
end
