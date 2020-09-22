function x = linear_solve(A,b)

[Ap bp] = make_triangular(A,b);
x = backsubstitution(Ap,bp);

