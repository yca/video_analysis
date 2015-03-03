svm_models = cell(1, 102);

for i=1:102
    fprintf('Training SVM for class %d\n', i);
    %libsvm is buggy about class weights for probabilistic estimates so we
    %can't do following:
    %   trlabels = 2 * (bow.trlabels == i) - 1;
    %   trdata = bow.pyramids(bow.trdata, 1);
    %   svm_models{i} = svmtrain_chi2(trlabels, trdata, '-t 5 -c 2048 -g 0.0078125 -w-1 30 -w1 3030 -b 1');
    
    trlabels_p = bow.trlabels(bow.trlabels == i);
    trlabels_n = bow.trlabels(bow.trlabels ~= i);
    trdata_p = bow.pyramids(bow.trlabels == i, :);
    trdata_n = bow.pyramids(bow.trlabels ~= i, :);
    %perm = randperm(length(trlabels_n));
    %trdata_n = trdata_n(perm(1:length(trlabels_p)), :);
    %trlabels_n = trlabels_n(perm(1:length(trlabels_p)));
    trdata_n = trdata_n(1:30:end, :);
    trlabels_n = trlabels_n(1:30:end);
    trlabels = [trlabels_p ; trlabels_n];
    trlabels = 2 * (trlabels == i) - 1;
    trdata = [trdata_p ; trdata_n];
    svm_models{i} = svmtrain_chi2(trlabels, trdata, '-t 5 -c 2048 -g 0.0078125 -b 1');
end

all_probs = zeros(2995, 102);
for i=1:102
    fprintf('Predicting SVM for class %d\n', i);
    tslabels = 2 * (bow.tslabels == i) - 1;
    tsdata = bow.pyramids(bow.tsdata, :);
    [prlabels, ~, probs] = svmpredict_chi2(tslabels, tsdata, svm_models{i}, '-b 1');
    
    pind = find(svm_models{i}.Label > 0);
    all_probs(:, i) = probs(:, pind);
end