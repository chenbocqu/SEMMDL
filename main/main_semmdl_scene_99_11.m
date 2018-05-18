
clear   all; 
warning off;
clc;

%% load toolkits
addpath('.\..');
addpath('.\..\large_scale_svm');
addpath('.\..\dictionary_learning');
addpath('.\..\FOptM');
addpath('.\..\SEMMDL');

%% load dataset
addpath('.\..\data');

for dataset=[5]
    
    if (dataset == 1)
        load    YaleB_DR_DAT
        f       = 10; % f
        dname   = 'Extended Yale B dataset';
        
    elseif (dataset == 2)
        load    AR_DR_DAT
        f       = 5;
        dname   = 'AR dataset';
        
    elseif (dataset == 3)
        load    MNIST
        f       = 30;
        dname   = 'MNIST dataset';
        
    elseif (dataset == 4)
        load    USPS
        f       = 30;
        dname   = 'USPS dataset';
        
    elseif (dataset == 5)
        load    scene15
        f       = 10;
        dname   = 'Scene15';
        
    elseif (dataset == 6)
        load    caltech101
        f       = 5;
        dname   = 'Caltech101';
    end
    
    tr_dat  = Train_DAT;
    tt_dat  = Test_DAT;
    trls    = trainlabels;
    ttls    = testlabels;
    
    if(dataset == 3)
        ttls    = ttls + 1;
        trls    = trls + 1;
    end
    
    clear Train_DAT Test_DAT trainlabels testlabels;
    
    %% 算法模型参数
    lambda1         = 3e-2;
    
    param.rdim      = 300;      % 降维
    param.f         = f;        % 每类原子个数
    param.max_iters = 3;        % 最大迭代次数
    param.lambda1   = lambda1;  % 
    param.lambda2   = 0.5;      %
    param.lambda3   = 1e-6;     %
    param.theta     = 5;        % theta
    param.draw      = false;    % 是否绘制曲线
    
    tic;
    [ model ] = semmdl( tr_dat, trls,tt_dat,ttls, param );
    toc;
    
    reco_iters{dataset} = model.reco_rates;
    
end

save 'semmdl_reco_iters' reco_iters
