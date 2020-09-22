function [minP, min_test_error, min_train_error] = k_fold_cross(Z1, Y, N)
    groups = int32(10);
    minP = 0;
    min_test_error = +Inf;
    testError = [];
    fold = idivide(N,groups,'floor');
    
    for polynomial = 1:8
        
        train_R = 0;
        test_R = 0;
        
        for groups = 1:groups

            Z = [];
            Z_test = [];

            %Generating train and test for current fold
            testZ1 = Z1(fold*(groups-1)+1:fold*groups,:)';
            trainZ1 = [Z1(1:fold*(groups-1),:);Z1(fold*groups+1:N,:)]';
            testY = Y(fold*(groups-1)+1:fold*groups);
            trainY = Y([1:fold*(groups-1),fold*groups+1:N]);

            %Size of training set
            [N1, M1] = size(trainZ1);

            %Size of test set
            [N2, M2] = size(testZ1);

            %Generating basis expansion for current polynomial selection
            for i = 1:polynomial
                Z = [Z;trainZ1.^i];
            end

            for i = 1:polynomial
                Z_test = [Z_test;testZ1.^i];
            end

            %adding the intercept columns for basis expansion matrix Z
            constant1 = ones(1,M1);
            Z = [Z;constant1];

            %adding the intercept columns for basis expansion matrix Z_test
            constant2 = ones(1,M2);
            Z_test = [Z_test;constant2];
            [w, R] = solve_weight(Z,trainY);

            %MSE for training data selected from the current fold
            train_R = train_R + R/M1;

            %Calculating the test MSE for the current fold
            test_R = test_R + test_error(w, Z_test, testY)/M2;  
        end

        %get the average test error for all 10 folds
        test_R = test_R / 10;
        train_R = train_R / 10;
        
        %choose minimum error value and choose best polynomial for min
        %error
        
        if(test_R<=min_test_error)
                min_test_error = test_R;
                min_train_error = train_R;
                minP = polynomial;
        end
        testError = [testError, train_R];
    end
end