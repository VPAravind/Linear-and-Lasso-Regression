function Y_pred = predict(w_final, minP)
testdata = importdata('testinputs.txt');
% K = int32(10);
[N2,~] = size(testdata);
N2 = int32(N2);
% % fold = idivide(N,K,'floor');
X = testdata(:,1:8);
% Y = traindata(:,9);
Xt1 = sqrt(X(:,1:4));
Xt2 = sqrt(X(:,6:8));
Xt3 = X(:,5);
Zt1= cat(2,Xt1,Xt3,Xt2);
Zt1 = Zt1';
Z_test_pred = [];

for i = 1:minP
    Z_test_pred = [Z_test_pred;Zt1.^i];
end
constant2 = ones(1,N2);
Z_test_pred = [Z_test_pred;constant2];
Y_pred = w_final' * Z_test_pred;
end
