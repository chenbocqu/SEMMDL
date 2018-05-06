
clear all; clc;
warning off;


reco_rates= [];

%%
load YaleB_DR_DAT;
dname = 'Yale B';

%% 训练样本集
tr_dat = Test_DAT(:,1:64);
trls = testlabels(:,1:64);

tt_dat=Train_DAT(:,1:40);
ttls=trainlabels(:,1:40);

clear Train_DAT Test_DAT trainlabels testlabels;


%% 算法初始化过程，初始字典D，类标Fisher矩阵U，降维后的训练数据和测试数据
lambda1         = 2e-3;

param.rdim      = 9; % 169
param.f         = 10;
param.max_iters = 5; % 30
param.lambda1   = lambda1;
param.lambda2   = 1.2;
param.lambda3   = 1e-6;
param.theta     = 5; % 7 - > 97.205%
param.draw      = false;

tic;
[ model ] = pca_mmdl( tr_dat, trls, param );
toc;

%% encode the testing data
fprintf('\n\n---------- Testing --------------\n\n');

B           = model.B;
U           = model.U;
b           = model.b;

class_list  = model.class_list;
tt_dat      = model.P * tt_dat;
tt_dat      = normalize_mat(tt_dat);

temp        = inv(B'*B+lambda1*eye(size(B,2)))*B';
S_test      = temp*tt_dat;

[ttls_pred, ~]  = li2nsvm_multiclass_fwd(S_test', U, b, class_list);
reco_rate       =  (sum(ttls_pred'==ttls))/length(ttls);

disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
reco_rates = [reco_rates; roundn(reco_rate*100,-4) ];

[M,eigvalue] = PCA(S_test',2);
ld_test = M'*S_test;

pos= find(ttls ==1 );
neg= find(ttls ==2 );

h1=plot(ld_test(1,pos),ld_test(2,pos),'r+');hold on;
h2=plot(ld_test(1,neg),ld_test(2,neg),'g*');hold on;
grid on;

% h2=plot(dataset(neg,demension1),dataset(neg,demension2),'g*');
% h3=plot(model.SVs(:,demension1),model.SVs(:,demension2),'o');

