clear;
traindata = importdata('traindata.txt');
testdata = importdata('testinputs.txt');
K = int32(10);
[N,~] = size(traindata);
N = int32(N);
fold = idivide(N,K,'floor');
X = traindata(:,1:8);
test = testdata(:,1:8);
Y = traindata(:,9);
X1 = sqrt(X(:,1:4));
X2 = sqrt(X(:,6:8));
X3 = X(:,5);
Z1= cat(2,X1,X3,X2);
X1 = sqrt(test(:,1:4));
X2 = sqrt(test(:,6:8));
X3 = test(:,5);
Ztest= cat(2,X1,X3,X2);
minTesterror = +Inf;
order = zeros(1,8);

for count = 0:(3^8-1)
    back = count;
    for o = 1:8
        order(o) = mod(back,3)+1;
        back = idivide(int32(back),int32(3),'floor');
    end
    test_R = 0;
    tR = 0;
    for k = 1:K
        Z = [];
        Z_test = [];
        testZ1 = Z1(fold*(k-1)+1:fold*k,:)';
        trainZ1 = [Z1(1:fold*(k-1),:);Z1(fold*k+1:N,:)]';
        testY = Y(fold*(k-1)+1:fold*k);
        trainY = Y([1:fold*(k-1),fold*k+1:N]);
        [N1, M1] = size(trainZ1);
        [N2, M2] = size(testZ1);
        for o = 1:8
            p = order(o);
            trainZ2 = trainZ1(o,:);
            for i = 1:p
                Z = [Z;trainZ2.^i];
            end
        end
        for o = 1:8
            p = order(o);
            testZ2 = testZ1(o,:);
            for i = 1:p
                Z_test = [Z_test;testZ2.^i];
            end
        end
        constant1 = ones(1,M1);
        constant2 = ones(1,M2);
        Z = [Z;constant1];
        Z_test = [Z_test;constant2];
        [w, R] = solve_weight(Z,trainY);
        tR = tR + R/M1;
        test_R = test_R + test_error(w, Z_test, testY)/M2;       
    end    
 
    test_R = test_R / 10;
    tR = tR / 10;
    if(test_R<=minTesterror)
            minTesterror = test_R;
            minOrder = order;
            minTR = tR;
    end  
end

Z = [];
Z1 = Z1';
for o = 1:8
    p = minOrder(o);
    trainZ2 = Z1(o,:);
    for i = 1:p
        Z = [Z;trainZ2.^i];
    end
end
constant = ones(1,N);
Z = [Z;constant];
[w_final, R_final] = solve_weight(Z,Y);
averageError = R_final / double(N);
fprintf('minTR:%-15.4fminTesterror:%-15.4faverageError:%-10.4f\n',minTR,minTesterror,averageError);
fid=fopen('lasso_predicted.txt','w+');

%Call predict function to get the predicted values
Z = [];
Ztest = Ztest';
for o = 1:8
    p = minOrder(o);
    trainZ2 = Ztest(o,:);
    for i = 1:p
        Z = [Z;trainZ2.^i];
    end
end
constant = ones(1,size(Ztest,2));
Z = [Z;constant];

Y_pred = w_final'*Z;

%write predicted value out to linear_prediction.txt 
fid = fopen('customizedlinear_prediction.txt' , 'w+');
fprintf(fid,'%3d \r\n',Y_pred);
fclose(fid);