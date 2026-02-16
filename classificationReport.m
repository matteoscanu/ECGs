function classificationReport(y_true, y_pred, classes)
    
    % note: this function is ideal when classes are more than two. 
    % you can use it also for binary classification but imo is a little
    % redundant

    % Compute the confusion matrix
    
    cmats = zeros(length(classes));
    for column = 1 : size(y_true, 2) % one iteration for each fold
        cmat = confusionmat(y_true(:, column), y_pred(:, column), 'Order', classes);
        cmats = cmats + cmat;
    end

    % Initialize metrics

    num_classes = length(classes);
    accuracy = zeros(num_classes, 1);
    precision = zeros(num_classes, 1);
    sensitivity = zeros(num_classes, 1);
    specificity = zeros(num_classes, 1);

    % Calculate metrics for each class

    for i = 1 : num_classes
        TP = cmats(i, i);           % True Positives
        FP = sum(cmats(:, i)) - TP; % False Positives
        FN = sum(cmats(i, :)) - TP; % False Negatives
        TN = sum(cmats, 'all') - (TP + FP + FN);
        accuracy(i) = (TP + TN) / (TP + TN + FP + FN);
        precision(i) = TP / (TP + FP);
        sensitivity(i) = TP / (TP + FN);
        specificity(i) = TN / (TN + FP);
    end

    % create a table for the report

    fprintf('\n\tConfusion matrix for all classes\n\n')
    disp(array2table(cmats, 'VariableNames', string(classes), 'RowNames', string(classes)))
    fprintf('\tClassification report\n\n')
    report = table(classes, accuracy, precision, sensitivity, specificity, ...
    'VariableNames', {'Class', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity'});
    disp(report)
    fprintf('Averages:\n')
    fprintf('Accuracy -> %.4f\n', mean(accuracy))
    fprintf('Precision -> %.4f\n', mean(precision))
    fprintf('Sensitivity -> %.4f\n', mean(sensitivity))
    fprintf('Specificity -> %.4f\n\n', mean(specificity))
end
