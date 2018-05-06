function P = update_P( P, X, B, S,lambda )

%% This code solves the following problem:
% 
%    minimize_P   ||X - B*S||^2 + lambda*||X-P'PX||^2
%    subject to   P'P = I

opts.record = 0; %
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

M = P';

[M, out]= OptStiefelGBB(M, @obj_P, opts, X, B, S,lambda ); 

P = M';

return