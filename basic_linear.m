clear;

%data preprocessing
traindata = importdata('traindata.txt');
K = int32(10);
[N,M] = size(traindata);
N = int32(N);

%matrix and vector construction
X_inp = traindata(:,1:8);
Y = traindata(:,9);
X_part1 = sqrt(X_inp(:,1:4));
X_part3 = sqrt(X_inp(:,6:8));
X_part2 = X_inp(:,5);
Z1= cat(2,X_part1,X_part2,X_part3);

%polynomial selection and cross_validation
[opt_poly, min_test_error, min_train_error] = k_fold_cross(Z1, Y, N);

Z1 = Z1';
Z = [];

%using selected polynomial to construct a new basis expansion matrix Z
for i = 1:opt_poly
    Z = [Z;Z1.^i];
end

%adding the column for the intercept in weight matrix to the basis expanison matrix
constant = ones(1,N);
Z = [Z;constant];

%calculating the final Weight vector and the LSE
[W, R] = solve_weight(Z,Y);

%Taking the average of error to get MSE
mean_error = R / double(N);

fprintf('optimum Polynomial order:%-4dminimum Train Error:%-15.4fminimum Test Error:%-15.4fMean Error:%-10.4f\n',opt_poly,min_train_error,min_test_error,mean_error);

%Call predict function to get the predicted values
Y_pred = predict(W, opt_poly);

[~,Mp] = size(Y_pred);

%write predicted value out to linear_prediction.txt 
fid = fopen('linear_prediction.txt' , 'w+');
fprintf(fid,'%3d \r\n',Y_pred);
fclose(fid);


