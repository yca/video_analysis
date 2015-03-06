clear
clc

addpath('/home/caglar/myfs/tasks/video_analysis/liblinear-1.96/matlab/');
addpath('/home/caglar/myfs/tasks/video_analysis/libsvm-3.20/matlab/');
addpath('/home/caglar/myfs/tasks/video_analysis/libsvm-2.9-dense_chi_square_mat');
warning('off', 'stats:pdist2:zeroPoints');

trial = {};
trial.ind = 7;
trial.rcnt = 1;


pooling = {'avg' ; 'max'};
%Krange = [128 256 512 1024 2048 3072 4096 5120 6144 7168 8192];
%Lrange = [4 3 3 3 3 2 2 2 2 2 2];
Krange = [512 1024 2048 3072 4096 5120 6144 7168 8192];
LHrange = [3 2 2 2 2 2 2 2 2];
LLrange = [1 1 1 1 1 1 1 1 1];


for pi=1:length(pooling)
    for j=1:length(Krange)
        trial.K = Krange(j);
        for k=LLrange(j):LHrange(j)
            trial.ind = trial.ind + 1;
            trial.L = k;

            for i=1:trial.rcnt
                fname = strcat(['trials/trial_', int2str(trial.ind), '_', int2str(i), '.mat']);
                if exist(fname, 'file')
                    continue
                end
                bow = bow_mine2('images/Caltech101');
                bow.pooling = pooling{pi};
                res = evaluate_bow(bow, trial);
                save(fname, 'res', '-v7.3');
                clear res;
            end
        end
    end
end

return

trial.K = 256;
trial.L = 3;
for i=1:trial.rcnt
    %bow = vlfeat_bow('images/Caltech101');
    bow = bow_mine2('images/Caltech101');
    
    res = evaluate_bow(bow, trial);
    save(strcat(['trials/trial_', int2str(trial.ind), '_', int2str(i), '.mat']), 'res');
    clear res;
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

