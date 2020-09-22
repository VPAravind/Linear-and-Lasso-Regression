function [w, R] = solve_weight(Z, y)
M = Z*Z';
b = Z*y;
w = linear_solve(M, b);
R = (w'*Z-y')*(w'*Z-y')';
end