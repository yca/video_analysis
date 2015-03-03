function D = chi2_kernel(X,Y)
    D = zeros(size(X,1),size(Y,1));
    for i=1:size(Y,1)
        fprintf('%d of %d\n', i, size(Y, 1));
        d = bsxfun(@minus, X, Y(i,:));
        s = bsxfun(@plus, X, Y(i,:));
        D(:,i) = sum(d.^2 ./ (s/2+eps), 2);
    end
    D = 1 - D;
end