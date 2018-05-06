
%% 使用SVGDL同样的参数
% 根据不同的迭代次数的精度

clear all;clc;
warning off;

%% load toolkits
addpath('.\large_scale_svm');
addpath('.\dictionary_learning');
addpath('.\FOptM');
addpath('.\PCA_MMDL');
%% load dataset
addpath('.\data');
for dataset=[1:2]
    
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
    
    %% 算法初始化过程，初始字典D，类标Fisher矩阵U，降维后的训练数据和测试数据
    lambda1         = 3e-2;
    
    param.rdim      = 300; % 169--96.037%
    param.f         = num_atom_per_class;
    param.max_iters = 20; % 30
    param.lambda1   = lambda1;
    param.lambda2   = 2e-3;
    param.lambda3   = 1e-6;
    param.theta     = 8; % 7 - > 97.205%
    param.draw      = false;
    
    tic;
    [ model ] = semmdl( tr_dat, trls,tt_dat,ttls, param );
    toc;
    
    reco_iters{dataset} = model.reco_rates;
    
end
save 'semmdl_reco_iters' reco_iters
