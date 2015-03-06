%function [u stddev u2 stddev2] = read_trials(int, cnt)
function [index] = read_trials()

if exist('trials/index.mat', 'file')
    index = load('trials/index.mat');
    return
end

trials = dir('trials/trial*.mat');
index = cell(1, length(trials));
for i=1:length(trials)
    fprintf('reading %s\n', trials(i).name);
    load(fullfile('trials/', trials(i).name));
    index{i}.K = size(res.dict, 1);
    N = size(res.bow.pyramids, 2) / index{i}.K;
    index{i}.L = log2(3 * N + 1) / 2;
    index{i}.distance = res.bow.distance;
    index{i}.pooling = res.bow.pooling;
    index{i}.filename = trials(i).name;
    index{i}.acc_chi2 = get_subacc(res, 'prlabels');
    index{i}.acc_kinters = get_subacc(res, 'prlabels_kinters');
    index{i}.acc_linear = get_subacc(res, 'prlabels_linear');
    index{i}.acc_lschi2 = get_subacc(res, 'prlabels_libsvmchi2');
    index{i}.acc_lsrbf = get_subacc(res, 'prlabels_libsvmrbf');
end

save('trials/index.mat', 'index');
return

acc = zeros(1, cnt);
acc2 = zeros(1, cnt);

for i=1:cnt
    load(strcat(['trials/trial_', int2str(int), '_', int2str(i), '.mat']));
    acc(i) = mean(res.acc);
    acc2(i) = mean(res.acc_kinters);
end

u = mean(acc);
u2 = mean(acc2);
stddev = std2(acc);
stddev2 = std2(acc2);

end

function acc = get_subacc(res, field)

acc = 0;
if isfield(res, field)
    if size(res.bow.tslabels) == size(res.(field))
        acc = mean(get_classification_accuracy(res.bow.database.nclass, res.bow.tslabels, res.(field))); 
    else
        acc = mean(get_classification_accuracy(res.bow.database.nclass, res.bow.tslabels', res.(field))); 
    end
end

end

%res.acc_kinters = get_classification_accuracy(bow.database.nclass, bow.tslabels', bow.prlabels);