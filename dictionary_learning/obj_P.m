function [F,G] = obj_P( M, X, B, S,lambda )

%% 为了使用正交约束
% 传入的时候 M=Pt
% 返回的时候 P=Mt

% F = trace( M'*X - B*S )

XXtM    = X*X'*M;
BSXt    = B*S*X';

%% F 为值，G为梯度
F       = trace( (1-lambda)*M'*XXtM ) - trace(2*M*BSXt );
G       = 2*(1-lambda)*XXtM - 2* BSXt';

return