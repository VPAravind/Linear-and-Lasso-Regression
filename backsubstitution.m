function x = backsubstitution(A,b)
% A is an upper-triangular matrix
% b is a vector
% want to find x s.t. A*x = b

N = length(A);
x = zeros(N,1);
for i=N:-1:1
    if A(i,i) == 0
        x(i) = 1;
        continue
    end
    x(i) = b(i) - A(i,i+1:N)*x(i+1:N);
    x(i) = x(i)/A(i,i);
end