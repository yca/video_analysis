classdef bow_mine2 < vlfeat_bow
    properties
        distance
        pooling
    end
    methods
        function bow = bow_mine2(database_path)
            bow = bow@vlfeat_bow(database_path);
            bow.imp_name = 'mine2_v1';
            
            bow.distance = 'L2';
            bow.pooling = 'avg';
        end
        
        function dist_mat = get_distance(bow, feaSet, dictionary)
            if strcmp(bow.distance, 'L2')
                dist_mat = sp_dist2(feaSet.feaArr', dictionary);
            elseif strcmp(bow.distance, 'cosine')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'cosine');
                dist_mat(isnan(n2)) = 1;
            elseif strcmp(bow.distance, 'L1')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'minkowski', 1);
            elseif strcmp(bow.distance, 'L0.75')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'minkowski', 0.75);
            elseif strcmp(bow.distance, 'L0.5')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'minkowski', 0.5);
            elseif strcmp(bow.distance, 'L0.25')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'minkowski', 0.25);
            elseif strcmp(bow.distance, 'hamming')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'hamming');
            elseif strcmp(bow.distance, 'cityblock')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'cityblock');
            elseif strcmp(bow.distance, 'chebychev')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'chebychev');
            elseif strcmp(bow.distance, 'mahalanobis')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'mahalanobis');
            elseif strcmp(bow.distance, 'jaccard')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'jaccard');
            elseif strcmp(bow.distance, 'correlation')
                dist_mat = pdist2(feaSet.feaArr', double(dictionary), 'correlation');
            else
                dist_mat = sp_dist2(feaSet.feaArr', dictionary);
            end
        end
        
        function pyramid = pool_pyramid_avg(bow, texton_ind, wid, hgt, L, dictSize)
            binsHigh = 2^(L-1);
            for i=1:binsHigh
                for j=1:binsHigh

                    % find the coordinates of the current bin
                    x_lo = floor(wid/binsHigh * (i-1));
                    x_hi = floor(wid/binsHigh * i);
                    y_lo = floor(hgt/binsHigh * (j-1));
                    y_hi = floor(hgt/binsHigh * j);

                    texton_patch = texton_ind.data( (texton_ind.x > x_lo) & (texton_ind.x <= x_hi) & ...
                                                    (texton_ind.y > y_lo) & (texton_ind.y <= y_hi));

                    % make histogram of features in bin
                    pyramid_cell{1}(i,j,:) = hist(texton_patch, 1:dictSize)./length(texton_ind.data);
                end
            end

            % compute histograms at the coarser levels
            num_bins = binsHigh/2;
            for l = 2:L
                pyramid_cell{l} = zeros(num_bins, num_bins, dictSize);
                for i=1:num_bins
                    for j=1:num_bins
                        pyramid_cell{l}(i,j,:) = ...
                        pyramid_cell{l-1}(2*i-1,2*j-1,:) + pyramid_cell{l-1}(2*i,2*j-1,:) + ...
                        pyramid_cell{l-1}(2*i-1,2*j,:) + pyramid_cell{l-1}(2*i,2*j,:);
                    end
                end
                num_bins = num_bins/2;
            end

            % stack all the histograms with appropriate weights
            pyramid = [];
            for l = 1:L-1
                pyramid = [pyramid pyramid_cell{l}(:)' .* 2^(-l)];
            end
            pyramid = [pyramid pyramid_cell{L}(:)' .* 2^(1-L)];
        end
        
        function pyramid = pool_pyramid_max(bow, texton_ind, wid, hgt, L, dictSize)
            binsHigh = 2^(L-1);
            for i=1:binsHigh
                for j=1:binsHigh

                    % find the coordinates of the current bin
                    x_lo = floor(wid/binsHigh * (i-1));
                    x_hi = floor(wid/binsHigh * i);
                    y_lo = floor(hgt/binsHigh * (j-1));
                    y_hi = floor(hgt/binsHigh * j);

                    texton_patch = texton_ind.data( (texton_ind.x > x_lo) & (texton_ind.x <= x_hi) & ...
                                                    (texton_ind.y > y_lo) & (texton_ind.y <= y_hi));

                    % make histogram of features in bin
                    h = hist(texton_patch, 1:dictSize);
                    pyramid_cell{1}(i,j,:) = h > 1;
                end
            end

            % compute histograms at the coarser levels
            num_bins = binsHigh/2;
            for l = 2:L
                pyramid_cell{l} = zeros(num_bins, num_bins, dictSize);
                for i=1:num_bins
                    for j=1:num_bins
                        % find the coordinates of the current bin
                        x_lo = floor(wid/num_bins * (i-1));
                        x_hi = floor(wid/num_bins * i);
                        y_lo = floor(hgt/num_bins * (j-1));
                        y_hi = floor(hgt/num_bins * j);

                        texton_patch = texton_ind.data( (texton_ind.x > x_lo) & (texton_ind.x <= x_hi) & ...
                                                    (texton_ind.y > y_lo) & (texton_ind.y <= y_hi));
                        h = hist(texton_patch, 1:dictSize);
                        pyramid_cell{l}(i,j,:) = h > 1;
                    end
                end
                num_bins = num_bins/2;
            end

            % stack all the histograms with appropriate weights
            pyramid = [];
            for l = 1:L-1
                pyramid = [pyramid pyramid_cell{l}(:)'];
            end
            pyramid = [pyramid pyramid_cell{L}(:)'];
        end
        
        function pyramids = create_image_pyramids(bow, images, dictionary, L)
            dictSize = size(dictionary, 1);
            pyramids = zeros(length(images),dictSize*sum((2.^(0:(L-1))).^2));
            
            for iter1 = 1:length(images)
                fpath = images{iter1};
                fpath = strrep(fpath, 'images/', 'data/');
                fpath = strrep(fpath, '.jpg', '.mat');
                load(fpath);
                feaSet.feaArr = double(feaSet.feaArr);
                
                dist_mat = bow.get_distance(feaSet, dictionary);
                
                [~, min_ind] = min(dist_mat, [], 2);
                texton_ind.data = min_ind;
                texton_ind.x = feaSet.x;
                texton_ind.y = feaSet.y;
                wid = feaSet.width;
                hgt = feaSet.height;
                if strcmp(bow.pooling, 'max')
                    pyramid = bow.pool_pyramid_max(texton_ind, wid, hgt, L, dictSize);
                elseif strcmp(bow.pooling, 'avg')
                    pyramid = bow.pool_pyramid_avg(texton_ind, wid, hgt, L, dictSize);
                end

                pyramids(iter1,:) = pyramid;

                fprintf('Calculating spatial-pyramid for %s\n', fpath);
            end
            bow.pyramids = pyramids;
        end
    end
end
                
function n2 = sp_dist2(x, c)
% DIST2	Calculates squared distance between two sets of points.
% Adapted from Netlab neural network software:
% http://www.ncrg.aston.ac.uk/netlab/index.php
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the  squared distance from the Ith row of X to the
%	Jth row of C.
%
%
%	Copyright (c) Ian T Nabney (1996-2001)

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
  ones(ndata, 1) * sum((c.^2)',1) - ...
  2.*(x*(c'));

% Rounding errors occasionally cause negative entries in n2
if any(any(n2<0))
  n2(n2<0) = 0;
end

end