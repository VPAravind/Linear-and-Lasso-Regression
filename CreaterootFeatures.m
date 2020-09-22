function [Z] = CreaterootFeatures(X,level)
Z = [];
for i = 1:level
    new_features = X.^(i);
    Z = [Z,new_features];
end
end