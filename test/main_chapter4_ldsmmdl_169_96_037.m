
% ����2��
% 169ά  - 96.037%
% 300ά  - 96.646%
% 400ά  - 96.697%

% ����5��
% 169ά  - XX%
% 300ά  - 96.697%
% 400ά  - XX%

%% script for testing the discriminative dictionary learning code - SEMMDL

clear all; clc;
warning off;

%% load toolkits
addpath('.\large_scale_svm');
addpath('.\dictionary_learning');
addpath('.\FOptM');
addpath('.\PCA_MMDL');
%% load dataset
addpath('.\data');

reco_rates= [];

for dataset=[1]
    
    if (dataset == 1)
        load YaleB_DR_DAT
        num_atom_per_class = 10; % f
        dname = 'Extended Yale B dataset';
        
    elseif (dataset == 2)
        load AR_DR_DAT
        num_atom_per_class = 5;
        dname = 'AR dataset';
        
    elseif (dataset == 3)
        load MNIST
        num_atom_per_class = 30;
        dname = 'MNIST dataset';
        
    elseif (dataset == 4)
        load USPS
        num_atom_per_class = 30;
        dname = 'USPS dataset';
    end
    
    tr_dat = Train_DAT;
    tt_dat = Test_DAT;
    trls = trainlabels;
    ttls = testlabels;
    
    if(dataset == 3)
        ttls    = ttls + 1;
        trls    = trls + 1;
    end
    
    clear Train_DAT Test_DAT trainlabels testlabels;
    
    %% �㷨��ʼ�����̣���ʼ�ֵ�D�����Fisher����U����ά���ѵ�����ݺͲ�������
    lambda1         = 0.03;
    
    param.rdim      = 300; % 169--96.037%
    param.f         = num_atom_per_class;
    param.max_iters = 5; % 30
    param.lambda1   = lambda1;
    param.lambda2   = 0.7;
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
    
    %% ����
    class_list  = model.class_list;
    tt_dat      = normalize_mat(tt_dat);
    
    S_test        = inv(B'*B+lambda1*eye(size(B,2)))*B'*model.P * tt_dat;
    
    [ttls_pred, ~]  = li2nsvm_multiclass_fwd(S_test', U, b, class_list);
    reco_rate       =  (sum(ttls_pred'==ttls))/length(ttls);
    
    disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
%     fprintf(['\nThe Recognition Rate on the ', dname, ' is ', num2str(roundn(reco_rate,-4)),' !\n']);
    reco_rates = [reco_rates; roundn(reco_rate*100,-4) ];
    
%     plot_dict(model);
    
end

reco_rates