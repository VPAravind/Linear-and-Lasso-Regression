function [A b] = make_triangular(A,b)

[N,~] = size(A);
for i=1:N
    flag = 0;
    if A(i,i) == 0
        for k = i+1:N
            if A(k,i)~=0
                temp = A(k,:);
                A(k,:) = A(i,:);
                A(i,:) = temp;
                flag = 1;
                break
            end
        end
        if flag == 0
            continue
        end
    end
    for j=i+1:N
        factor = -A(j,i)/A(i,i);
        A(j,:) = A(j,:) + factor*A(i,:);
        b(j)   = b(j)   + factor*b(i);
    end
end
