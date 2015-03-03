clear

trials = cell(1, 1000);
trial = {};
trial.ind = 1;
trial.K = 256;
trial.L = 3;
trial.rcnt = 10;
trial.acc = zeros(trial.rcnt, 102);
trial.dicts = cell(1, trial.rcnt);
trial.prlabels = cell(1, trial.rcnt);
trial.tslabels = cell(1, trial.rcnt);
trial.svm_models = cell(1, trial.rcnt);
trial.imps = cell(1, trial.rcnt);
trial.opts = cell(1, trial.rcnt);
for i=1:trial.rcnt
    bow = vlfeat_bow('images/Caltech101');
    dfeatures = bow.select_features(50, 10000);
    dict = bow.create_dict(trial.K, dfeatures);
    bow.create_image_pyramids(bow.database.path, dict, trial.L);
    bow.split_data3(30, 50);
    bow.train2(10, bow.pyramids);
    bow.predict();
    
    %keep results
    trial.acc(i, :) = get_classification_accuracy(bow.database.nclass, bow.tslabels', bow.prlabels);
    trial.dicts{i} = dict;
    trial.prlabels{i} = bow.prlabels;
    trial.tslabels{i} = bow.tslabels;
    trial.svm_models{i} = bow.svm_model;
    trial.imps{i} = bow.imp_name;
    trial.opts{i} = bow.opts;
    
    save(strcat(['trials/trial_', int2str(trial.ind), '_', int2str(i), '.mat'], 'trial'));
end

%save('trials/trial_all.mat', 'trials');

return



% clear
% 
% load('data/bow_vlfeat_D256_L3.mat');
% bow.split_data3(30, 50);
% %pyramids = bow.pyramids(bow.trdata, :);
% bow.train2(10, bow.pyramids);
% bow.predict()
% mean(get_classification_accuracy(bow.database.nclass, bow.tslabels', bow.prlabels))
% 
% return

clear

bow = vlfeat_bow('images/Caltech101');
load('dictionary/baseline-vocab.mat');
dict2 = vocab';

L = 3;
bow.create_pyramids(dict2, L);

bow.split_data3(30, 50);
bow.train2(10, bow.pyramids);
bow.predict()
mean(get_classification_accuracy(bow.database.nclass, bow.tslabels', bow.prlabels))


%save(strcat('data/bow_', bow.imp_name, '_D', int2str(size(dict2, 1)), '_L', int2str(L)), 'bow');
return

bow.split_data2(30, 50);
bow.create_image_pyramids(bow.images, dict2, 3);
bow.train(10);
bow.predict();
mean(get_classification_accuracy(bow.database.nclass, bow.tslabels, bow.prlabels))
%trdata2 = pyramids(trdata, :);
%tsdata2 = pyramids(tsdata, :);
%trlabels = labels(trdata)';
%tslabels = labels(tsdata)';
%model = svmtrain_chi2(trlabels, trdata2, '-c 10000 -t 5 -g 0.0078125');
%[prlabels, ~, ~] = svmpredict_chi2(tslabels, tsdata2, model, '-b 0');

return


clear

%bow = bow_mine1('data/Caltech101');
load('dictionary/dict_spm_256.mat');

bow = vlfeat_bow('images/Caltech101');
%load('dictionary/baseline-vocab.mat');
%dict2 = vocab';

%load('data/pyramids_mine1_256_L2.mat');
load('data/pyramids_vlfeat_256_L2.mat');
%pyramids = bow.create_image_hist(dict2, 3);

[trdata trlabels tsdata tslabels] = bow.split_data(pyramids, 15, 15);

%K = hist_isect(trdata, trdata);
%Ke = hist_isect(trdata, tsdata);
%model = svmtrain(trlabels, [(1:length(trlabels))' K], '-c 10 -t 4');
%[prlabels, ~, ~] = svmpredict(tslabels, [(1:length(tslabels))', Ke'], model, '-b 0');

model = svmtrain_chi2(trlabels, trdata, '-c 10000 -t 5 -g 0.0078125');
[prlabels, ~, ~] = svmpredict_chi2(tslabels, tsdata, model, '-b 0');


%mean(get_classification_accuracy(database.nclass, tslabels, prlabels))

