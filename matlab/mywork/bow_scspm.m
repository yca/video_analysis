classdef bow_scspm < vlfeat_bow
    properties
        gamma = 0.15;
    end
    methods
        function bow = bow_scspm(database_path)
            bow = bow@vlfeat_bow(database_path);
            bow.imp_name = 'scspm';
        end
        
        function dict = create_dict(bow, dict_size, features)
            %nBases -> dict_size, X -> features, beta=1e-5, gamma=0.15,
            %num_iters=50
            [B, ~, ~] = reg_sparse_coding(features', dict_size, eye(dict_size), 1e-5, bow.gamma, 50);
            dict = B;
        end
        
        function pyramids = create_image_pyramids(bow, images, dictionary, L)
            dictSize = size(dictionary, 1);
            pyramids = zeros(length(images),dictSize*sum((2.^(0:(L-1))).^2));
            
            for iter1 = 1:length(images)
                fpath = images{iter1};
                fpath = strrep(fpath, 'images/', 'data/');
                fpath = strrep(fpath, '.jpg', '.mat');
                load(fpath);
                
                pyramids(iter1, :) = sc_pooling(feaSet, dictionary, pyramid, bow.gamma)';
            end
        end
    end
end