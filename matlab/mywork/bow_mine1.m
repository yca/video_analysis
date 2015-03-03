classdef bow_mine1 < bow_model
    methods
        function bow = bow_mine1(database_path)
            bow = bow@bow_model(database_path, '*.mat');
        end
        %creates sift descriptors for the entire database and saves in
        %target folder
        function create_sift_desc(bow)
        end
        
        function create_image_pyramids(images, dict, L)
        end
        
        %creates bow descriptors for all images in the database
        function pyramids = create_pyramids(bow, dictionary, L)
            dictSize = size(dictionary, 1);
            binsHigh = 2^(L-1);
            pyramid_all = zeros(length(bow.database.path),dictSize*sum((2.^(0:(L-1))).^2));

            %H_all = zeros(dictSize, length(database.path));
            for iter1 = 1:length(bow.database.path)
                fpath = database.path{iter1};
                load(fpath);

                %dist_mat = sp_angle2(feaSet.feaArr', dictionary);
                dist_mat = sp_dist2(feaSet.feaArr', dictionary);
                [~, min_ind] = min(dist_mat, [], 2);
                texton_ind.data = min_ind;
                texton_ind.x = feaSet.x;
                texton_ind.y = feaSet.y;
                wid = feaSet.width;
                hgt = feaSet.height;

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

                pyramid_all(iter1,:) = pyramid;

                fprintf('Calculating spatial-pyramid for %s\n', fpath);
            end
            
            pyramids = pyramid_all;

        end
        
        %clusters images to create bow dictionary
        function dict = create_dict(bow, dict_size, image_max, total_max)
            features = zeros(128, length(bow.database.path) * image_max);
            for iter1 = 1:length(bow.database.path)
                fpath = database.path{iter1};
                load(fpath);
                p = randperm(size(feaSet.feaArr, 2));
                start = (iter1 - 1) * image_max + 1;
                fin = start + image_max - 1;
                features(:, start:fin) = feaSet.feaArr(:, p(1:image_max));
                fprintf('loading features from %s\n', bow.database.path{iter1});
            end

            dict = calculate_dict(features', dict_size, total_max);
        end
    end
end

function dict = calculate_dict(features, dict_size, ndata_max)

ndata = size(features,1);    
if (ndata > ndata_max)
    fprintf('Reducing to %d descriptors\n', ndata_max);
    p = randperm(ndata);
    features = features(p(1:ndata_max),:);
end

centers = zeros(dict_size, size(features,2));

options = zeros(1,14);
options(1) = 1; % display
options(2) = 1;
options(3) = 0.1; % precision
options(5) = 1; % initialization
options(14) = 100; % maximum iterations

dict = sp_kmeans(centers, features, options);

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

function [centres, options, post, errlog] = sp_kmeans(centres, data, options)
% KMEANS	Trains a k means cluster model.
% Adapted from Netlab neural network software:
% http://www.ncrg.aston.ac.uk/netlab/index.php
%
%	Description
%	 CENTRES = KMEANS(CENTRES, DATA, OPTIONS) uses the batch K-means
%	algorithm to set the centres of a cluster model. The matrix DATA
%	represents the data which is being clustered, with each row
%	corresponding to a vector. The sum of squares error function is used.
%	The point at which a local minimum is achieved is returned as
%	CENTRES.  The error value at that point is returned in OPTIONS(8).
%
%	[CENTRES, OPTIONS, POST, ERRLOG] = KMEANS(CENTRES, DATA, OPTIONS)
%	also returns the cluster number (in a one-of-N encoding) for each
%	data point in POST and a log of the error values after each cycle in
%	ERRLOG.    The optional parameters have the following
%	interpretations.
%
%	OPTIONS(1) is set to 1 to display error values; also logs error
%	values in the return argument ERRLOG. If OPTIONS(1) is set to 0, then
%	only warning messages are displayed.  If OPTIONS(1) is -1, then
%	nothing is displayed.
%
%	OPTIONS(2) is a measure of the absolute precision required for the
%	value of CENTRES at the solution.  If the absolute difference between
%	the values of CENTRES between two successive steps is less than
%	OPTIONS(2), then this condition is satisfied.
%
%	OPTIONS(3) is a measure of the precision required of the error
%	function at the solution.  If the absolute difference between the
%	error functions between two successive steps is less than OPTIONS(3),
%	then this condition is satisfied. Both this and the previous
%	condition must be satisfied for termination.
%
%	OPTIONS(14) is the maximum number of iterations; default 100.
%
%	Copyright (c) Ian T Nabney (1996-2001)

[ndata, data_dim] = size(data);
[ncentres, dim] = size(centres);

if dim ~= data_dim
  error('Data dimension does not match dimension of centres')
end

if (ncentres > ndata)
  error('More centres than data')
end

% Sort out the options
if (options(14))
  niters = options(14);
else
  niters = 100;
end

store = 0;
if (nargout > 3)
  store = 1;
  errlog = zeros(1, niters);
end

% Check if centres and posteriors need to be initialised from data
if (options(5) == 1)
  % Do the initialisation
  perm = randperm(ndata);
  perm = perm(1:ncentres);

  % Assign first ncentres (permuted) data points as centres
  centres = data(perm, :);
end
% Matrix to make unit vectors easy to construct
id = eye(ncentres);

% Main loop of algorithm
for n = 1:niters

  % Save old centres to check for termination
  old_centres = centres;
  
  % Calculate posteriors based on existing centres
  d2 = sp_dist2(data, centres);
  % Assign each point to nearest centre
  [minvals, index] = min(d2', [], 1);
  post = id(index,:);

  num_points = sum(post, 1);
  % Adjust the centres based on new posteriors
  for j = 1:ncentres
    if (num_points(j) > 0)
      centres(j,:) = sum(data(find(post(:,j)),:), 1)/num_points(j);
    end
  end

  % Error value is total squared distance from cluster centres
  e = sum(minvals);
  if store
    errlog(n) = e;
  end
  if options(1) > 0
    fprintf(1, 'Cycle %4d  Error %11.6f\n', n, e);
  end

  if n > 1
    % Test for termination
    if max(max(abs(centres - old_centres))) < options(2) & ...
        abs(old_e - e) < options(3)
      options(8) = e;
      return;
    end
  end
  old_e = e;
end

% If we get here, then we haven't terminated in the given number of 
% iterations.
options(8) = e;
if (options(1) >= 0)
  disp('Warning: Maximum number of iterations has been exceeded');
end

end