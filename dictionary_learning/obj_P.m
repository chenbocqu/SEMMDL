function [F,G] = obj_P( M, X, B, S,lambda )

%% Ϊ��ʹ������Լ��
% �����ʱ�� M=Pt
% ���ص�ʱ�� P=Mt

% F = trace( M'*X - B*S )

XXtM    = X*X'*M;
BSXt    = B*S*X';

%% F Ϊֵ��GΪ�ݶ�
F       = trace( (1-lambda)*M'*XXtM ) - trace(2*M*BSXt );
G       = 2*(1-lambda)*XXtM - 2* BSXt';

return