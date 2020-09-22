function R = test_error(w, Z, y)
% w weight col vector
% Z input matrix
% y label col vector
R = (w'*Z-y')*(w'*Z-y')';
end