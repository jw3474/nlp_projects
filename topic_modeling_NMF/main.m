% Each row in nyt data.txt corresponds to a single document. 
% It gives the index of words appearing in that document and the number of times they appear. 
% It uses the format “idx:cnt” with commas separating each unique word in the document. 
% Any index that doesn’t appear in a row has a count of zero for that word. 
% The vocabulary word corresponding to each index is given in the corresponding row of nyt vocab.dat.


%% Process data and get matrix x
% I use python to process nyt_data.txt and it will return a X.mat which
% stores sparse version of matrix X.
% Please run process_data.py before running this section, thanks.
load('X', 'X');

%% Run NMF and plot the objective function verses iteration
obj = zeros(100,1);
W = 1+(2-1)*rand(3012,25);
H = 1+(2-1)*rand(25,8447);
for i = 1:100
    purple = X./((W*H) + 10^(-16));
    w_tmp = W';
    w_row_sum = sum(w_tmp,2);
    for j = 1:size(w_tmp,1)
        w_tmp(j, :) = w_tmp(j, :) ./ w_row_sum(j);
    end
    H = H.*(w_tmp*purple);
    purple = X./((W*H) + 10^(-16));
    h_tmp = H';
    h_column_sum = sum(h_tmp,1);
    for j = 1:size(h_tmp,2)
        h_tmp(:, j) = h_tmp(:, j) ./ h_column_sum(j);
    end
    W = W.*(purple*h_tmp);
    WH = W*H;
    %obj(i) = -sum(sum(X.*log(WH+10^(-16))-WH));
    obj(i) = sum(sum(X.*log(1./(WH+10^(-16)))+WH));
end
figure
plot(obj);
title('Objective as a function of iteration');

%% List the 10 words having the largest weight and show the weight
normalized_w = W;
w_column_sum = sum(normalized_w,1);
for i = 1:size(normalized_w,2)
    normalized_w(:, i) = normalized_w(:, i) ./ w_column_sum(i);
end
[sorted_w_matrix, I_w] = sort(normalized_w,1,'descend');