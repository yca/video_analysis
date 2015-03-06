function res = evaluate_bow(bow, trial)

%overload caltech101 categories(remove background)
rcnt = length(find(bow.database.label == 1));
bow.database.label = bow.database.label(rcnt+1:end);
bow.database.label = bow.database.label - 1;
bow.database.path = bow.database.path(rcnt+1:end);
bow.database.nclass = bow.database.nclass - 1;
    
dfeatures = bow.select_features(50, 10000);
dict = bow.create_dict(trial.K, dfeatures);
bow.create_image_pyramids(bow.database.path, dict, trial.L);
bow.split_data3(30, 50);

res = {};

if trial.K < 2048
    bow.train2(10, bow.pyramids, 'kchi2');
    bow.predict();
    res.prlabels = bow.prlabels;

    bow.train2(10, bow.pyramids, 'kinters');
    bow.predict();
    res.prlabels_kinters = bow.prlabels;
end

%test with linear SVM(liblinear)
model = train(bow.trlabels, sparse(bow.pyramids(bow.trdata, :)), '-c 10000');
[res.prlabels_linear, ~, ~] = predict(bow.tslabels,sparse(bow.pyramids(bow.tsdata, :)), model, '-b 0');

%test with LIBSVM CHI2(remember no sparse support!)
model = svmtrain_chi2(bow.trlabels, bow.pyramids(bow.trdata, :), '-c 10000 -t 5 -g 0.0078125 -m 1000 -h 0');
[res.prlabels_libsvmchi2, ~, ~] = svmpredict_chi2(bow.tslabels, bow.pyramids(bow.tsdata, :), model, '-b 0');

%test with LIBSVM RBF(remember no sparse support!)
model = svmtrain_chi2(bow.trlabels, bow.pyramids(bow.trdata, :), '-c 10000 -t 2 -g 2 -m 1000 -h 0');
[res.prlabels_libsvmrbf, ~, ~] = svmpredict_chi2(bow.tslabels, bow.pyramids(bow.tsdata, :), model, '-b 0');

%keep results
res.bow = bow;
res.dict = dict;
