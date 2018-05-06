function [ X ] = normalize_mat( X )

%%
% X \in R{d x N} ,N is sample num,d is dimension of feature

%% 归一化
X = X-repmat(mean(X),[size(X,1) 1]);                % 去中心
X = X./( repmat(sqrt(sum(X.*X)), [size(X,1),1]) );  % 归一化

end

