
function [a R] = simple_least_squares(x,y)

%calculate optimal a
a = sum(x.*y)./sum(x.*x);

%calculate predict error R
y_pred = a*x;
R = norm(y-y_pred)^2;