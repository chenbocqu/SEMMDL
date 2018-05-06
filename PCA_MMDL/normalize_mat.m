function [ X ] = normalize_mat( X )

%%
% X \in R{d x N} ,N is sample num,d is dimension of feature

%% ��һ��
X = X-repmat(mean(X),[size(X,1) 1]);                % ȥ����
X = X./( repmat(sqrt(sum(X.*X)), [size(X,1),1]) );  % ��һ��

end

