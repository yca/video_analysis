classdef vlfeat_bow < bow_model
    properties
            opts
        end
    methods
        function bow = vlfeat_bow(database_path)
            bow = bow@bow_model(database_path, '*.jpg');
            bow.imp_name = 'vlfeat_v3';
            %opts = {'Step', 6 'Sizes', 5 'FloatDescriptors', true};
            bow.opts = {'Step', 8 'Sizes', 4 'FloatDescriptors', false};
            %opts = {'Step', 3 };
        end
        
        function create_sift_desc(bow)
            for iter1 = 1:length(bow.database.path)
                fpath = bow.database.path{iter1};
                dpath = strrep(fpath, 'images/', 'data/');
                dpath = strrep(dpath, '.jpg', '.mat');
                fprintf('creating descriptor for %s\n', fpath);
                im = imread(fpath);

                [drop , desc] = vl_phow(standarizeImage(im), bow.opts{:});

                feaSet.feaArr = desc;%double(desc);
                feaSet.x = drop(1, :)';
                feaSet.y = drop(2, :)';
                feaSet.width = size(im, 2);
                feaSet.height = size(im, 1);

                siftpath = fileparts(dpath);
                if ~isdir(siftpath),
                    mkdir(siftpath);
                end;
                save(dpath, 'feaSet');
            end
        end
        
        function pyramids = create_image_pyramids(bow, images, dictionary, L)
            dictSize = size(dictionary, 1);
            pyramids = zeros(length(images),dictSize*sum((2.^(0:(L-1))).^2) - dictSize);
            parfor iter1 = 1:length(images)
                ipath = images{iter1};
                im = imread(ipath);
                pyramids(iter1,:) = getImageDescriptor(dictionary', im, bow.opts);
                fprintf('Calculating spatial-pyramid for %s\n', ipath);
            end
            bow.pyramids = pyramids;
        end
        
        function features = select_features(bow, image_max, total_max)
            features = zeros(128, length(bow.database.path) * image_max);
            for iter1 = 1:length(bow.database.path)
                fprintf('loading features from %s\n', bow.database.path{iter1});
                fpath = bow.database.path{iter1};
                
                %im = standarizeImage(imread(fpath));
                %[~, fea_arr] = vl_phow(im, bow.opts{:}) ;
                
                dpath = strrep(fpath, 'images/', 'data/');
                dpath = strrep(dpath, '.jpg', '.mat');
                load(dpath);
                fea_arr = feaSet.feaArr;
                
                p = randperm(size(fea_arr, 2));
                start = (iter1 - 1) * image_max + 1;
                fin = start + image_max - 1;
                features(:, start:fin) = fea_arr(:, p(1:image_max));
            end

            features = single(vl_colsubset(features', total_max));
        end
        
        function dict = create_dict(bow, dict_size, features)
            dict = vl_kmeans(features', dict_size, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50)';
        end
        
        function train(bow, C, pyramids)
            psix = vl_homkermap(pyramids', 1, 'kchi2', 'gamma', 0.5);
            lambda = 1 / (C * length(bow.trdata));
            w = [];
            trdata = bow.trdata;
            imageClass = bow.imageClass;
            parfor ci = 1:bow.database.nclass
                perm = randperm(length(trdata));
                fprintf('Training SVM for class %d\n', ci);
                y = 2 * (imageClass(trdata) == ci) - 1;
                [w(:, ci) b(ci) info] = vl_svmtrain(psix(:, trdata(perm)), y(perm), lambda, 'Solver', 'sdca', ...
                    'MaxNumIterations', 50 / lambda, 'BiasMultiplier', 1, 'Epsilon', 1e-3);
            end
            bow.svm_model.b = b;
            bow.svm_model.w = w;
            bow.svm_model.valid = 1;
            bow.svm_model.psix = psix;
        end
        
        function train2(bow, C, pyramids, kernel)
            psix = vl_homkermap(pyramids', 1, kernel, 'gamma', 0.5);
            lambda = 1 / (C * length(bow.trdata));
            w = [];
            trdata = bow.trdata;
            trlabels = bow.trlabels;
            for ci = 1:bow.database.nclass
                perm = randperm(length(trdata));
                fprintf('Training SVM for class %d\n', ci);
                y = 2 * (trlabels == ci) - 1;
                [w(:, ci) b(ci) ~] = vl_svmtrain(psix(:, trdata(perm)), y(perm), lambda, 'Solver', 'sdca', ...
                    'MaxNumIterations', 50 / lambda, 'BiasMultiplier', 1, 'Epsilon', 1e-3);
            end
            bow.svm_model.b = b;
            bow.svm_model.w = w;
            bow.svm_model.valid = 1;
            bow.svm_model.psix = psix;
        end
        
        function train3(bow, C, pyramids)
            psix = vl_homkermap(pyramids', 1, 'kchi2', 'gamma', 0.5);
            svm = train(bow.trlabels, sparse(double(psix(:, bow.trdata))), strcat([' -s 3 -B 1 -c ', int2str(C)]), 'col');
            bow.svm_model.b = svm.w(:, end)';
            bow.svm_model.w = svm.w(:, 1:end-1)';
            bow.svm_model.valid = 1;
            bow.svm_model.psix = psix;
        end
        
        function predict(bow)
            scores = bow.svm_model.w' * bow.svm_model.psix + ...
                bow.svm_model.b' * ones(1,size(bow.svm_model.psix, 2));
            [~, pr_labels] = max(scores, [], 1);
            bow.prlabels = pr_labels(bow.tsdata);
            %bow.tslabels = bow.imageClass(bow.tsdata);
        end
    end
end

function hist = getImageDescriptor(dict, im, opts)
% -------------------------------------------------------------------------

im = standarizeImage(im) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(dict, 2) ;

% get PHOW features
[frames, descrs] = vl_phow(im, opts{:}) ;

[~, binsa] = min(vl_alldist(dict, single(descrs)), [], 1) ;

numSpatialX = [2 4];
numSpatialY = [2 4];
hists = cell(1, length(numSpatialX));
for i = 1:length(numSpatialX)
  binsx = vl_binsearch(linspace(1,width,numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([numSpatialY(i), numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(numSpatialY(i) * numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1, hists{:}) ;
hist = hist / sum(hist) ;

end

function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im2single(im) ;
if size(im,1) > 300, im = imresize(im, [300 NaN]) ; end
if size(im,2) > 300, im = imresize(im, [NaN 300]) ; end

end