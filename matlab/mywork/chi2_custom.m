function res=chi2_custom(x,y)
a=size(x);
b=size(y);
res = zeros(a(1,1), b(1,1));
for i=1:a(1,1)
    fprintf('%d of %d\n', i, a(1, 1));
    for j=1:b(1,1)
        resHelper = chi2_ireneHelper(x(i,:), y(j,:));
        res(i,j) = resHelper;
    end
end
function resHelper = chi2_ireneHelper(x,y)
a=(x-y).^2;
b=(x+y);
resHelper = 1-sum(2*a./(b + eps));