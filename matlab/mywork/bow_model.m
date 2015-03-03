classdef bow_model < handle
    properties
        database    %struct containing information for image database
        pyramids    %pyramids of all images in the database
        
        trdata          %selected train data(indexes)
        trdata_files    %selected train data(filenames)
        trlabels        %selected train labels
        tsdata          %selected test data(indexes)
        tsdata_files    %selected test data(filenames)
        tslabels        %selected test labels
        svm_model
        
        prlabels    %predicted labels corresponding to selected test labels
        
        imp_name    %implementation name, for data saving, please set in your classes constructor
    end
    methods (Abstract)
        create_sift_desc(database)
        %create_pyramids(dict, L)
        create_image_pyramids(images, dict, L)
        select_features(image_max, total_max)
        create_dict(database, dict_size, features)
    end
    methods
        function bow = bow_model(database_path, suffix)
            bow.database = retr_database_dir(database_path, suffix);
            bow.svm_model.valid = 0;
        end
        
        %Splits given pyramids into train and test pairs. pyramids should
        %row vectors of data. If ts_num + tr_num is bigger than available
        %samples then ts_num will be adjusted to fit data.
        function [tr_data, tr_labels, ts_data, ts_labels] = split_data(bow, pyramids, tr_num, ts_num)
            tr_idx = [];
            ts_idx = [];

            sc_label = bow.database.label;
            nclass = bow.database.nclass;
            for jj = 1:nclass
                idx_label = find(sc_label == jj);
                num = length(idx_label);

                idx_rand = randperm(num);

                if tr_num < 1
                    tr_num = round(length(idx_rand) * tr_num);
                end
                if ts_num < 1
                    ts_num = round(length(idx_rand) * ts_num);
                end
                if ts_num + tr_num > length(idx_rand)
                    ts_num = length(idx_rand) - tr_num;
                end

                tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
                ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:tr_num+ts_num))];
            end;

            tr_data = pyramids(tr_idx, :);
            tr_labels = sc_label(tr_idx);
            ts_data = pyramids(ts_idx, :);
            ts_labels = sc_label(ts_idx);
        end
        
        function split_data3(bow, tr_num, ts_num)
            tr_idx = [];
            ts_idx = [];

            sc_label = bow.database.label;
            nclass = bow.database.nclass;
            
            for jj = 1:nclass
                idx_label = find(sc_label == jj);
                %now idx_label contains indexes of target classes wrt
                %database indexes
                num = length(idx_label);

                idx_rand = randperm(num);
                %idx_rand will be random permutation of all image indexes
                %for current class

                %treat tr_num|ts_num as percentages if they are in the
                %range [0-1]
                if tr_num < 1
                    tr_num2 = round(length(idx_rand) * tr_num);
                else
                    tr_num2 = tr_num;
                end
                if ts_num < 1
                    ts_num2 = round(length(idx_rand) * ts_num);
                else
                    ts_num2 = ts_num;
                end
                %Do not let ts_num + tr_num to be bigger than image count
                if ts_num2 + tr_num2 > length(idx_rand)
                    ts_num2 = length(idx_rand) - tr_num2;
                end

                %append train and test indexes
                tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num2))];
                ts_idx = [ts_idx; idx_label(idx_rand(tr_num2+1:tr_num2+ts_num2))];
            end;

            
            bow.trdata = tr_idx;
            %bow.trdata_files = bow.database.path{tr_idx};
            bow.trlabels = bow.database.label(tr_idx);
            bow.tsdata = ts_idx;
            %bow.tsdata_files = bow.database.path{ts_idx};
            bow.tslabels = bow.database.label(ts_idx);
        end
        
        function [images, imageClass, selTrain, selTest] = split_data2(bow, tr_num, ts_num)
            images = {} ;
            imageClass = {} ;
            
            classes = dir('images/Caltech101');
            classes = classes([classes.isdir]);
            classes = {classes(3:102+2).name};
            for ci = 1:length(classes)
                ims = dir(fullfile('images/Caltech101', classes{ci}, '*.jpg'))' ;
                ims = vl_colsubset(ims, tr_num + ts_num);
                ims = cellfun(@(x)fullfile('images/Caltech101', classes{ci},x),{ims.name},'UniformOutput',false) ;
                images = {images{:}, ims{:}} ;
                imageClass{end+1} = ci * ones(1,length(ims)) ;  
            end
            
            selTrain = find(mod(0:length(images)-1, tr_num + ts_num) < tr_num) ;
            selTest = setdiff(1:length(images), selTrain) ;
            imageClass = cat(2, imageClass{:}) ;
            
            bow.imageClass = imageClass;
            bow.images = images;
            bow.trdata = selTrain;
            bow.tsdata = selTest;
            
        end
        
        function pyramids = get_pyramids(bow, images, dict, L)
            if length(bow.pyramids) == 0
                bow.create_pyramids(dict, L);
            end
            
            indexes = zeros(1, length(images));
            data = zeros(length(images), size(bow.pyramids, 2));
            for i = 1:length(images)
                ind = strfind(bow.database.path, images{i});
                ind = find(not(cellfun('isempty', ind)));
                data(i, :) = bow.pyramids(ind, :);
                indexes(i) = ind;
            end
            pyramids = data;
        end
    end
end

function [database] = retr_database_dir(rt_data_dir, suffix)
%=========================================================================
% inputs
% rt_data_dir   -the rootpath for the database. e.g. '../data/caltech101'
% outputs
% database      -a tructure of the dir
%                   .path   pathes for each image file
%                   .label  label for each image file
% written by Jianchao Yang
% Mar. 2009, IFP, UIUC
%=========================================================================

fprintf('dir the database...');
subfolders = dir(rt_data_dir);

database = [];

if nargin < 2
    suffix = '*.mat';
end

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') && ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(rt_data_dir, subname, suffix));
        c_num = length(frames);
                    
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        for jj = 1:c_num,
            c_path = fullfile(rt_data_dir, subname, frames(jj).name);
            database.path = [database.path, c_path];
        end;    
    end;
end;
disp('done!');

end