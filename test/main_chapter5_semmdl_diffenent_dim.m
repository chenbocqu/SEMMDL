
%% 使用SVGDL同样的参数

% 迭代2次
% 169维  - 96.037%
% 300维  - 96.646%
% 400维  - 96.697%

% 迭代5次
% 169维  - XX%
% 300维  - 96.697%
% 400维  - XX%

% 迭代20次
% AR 93.991%

%% script for testing the discriminative dictionary learning code - SEMMDL

clear all;clc;
warning off;

%% load toolkits
addpath('.\large_scale_svm');
addpath('.\dictionary_learning');
addpath('.\FOptM');
addpath('.\PCA_MMDL');
%% load dataset
addpath('.\data');

% dims = 20:20:400;
dims = 340:20:400;
% dims = 300;

for dataset=[2]
    
    reco_rates= [];
    
    for dim = dims
        
        fprintf('\ndim = %d\n',dim);
        
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
        lambda1         = 2e-3;
        
        param.rdim      = dim; % 169--96.037%
        param.f         = num_atom_per_class;
        param.max_iters = 20; % 30
        param.lambda1   = lambda1;
        param.lambda2   = 2.5;
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
        
        %% 测试
        class_list  = model.class_list;
        tt_dat      = normalize_mat(tt_dat);
        tic;
        S_test        = inv(B'*B+lambda1*eye(size(B,2)))*B'*model.P * tt_dat;
        
        [ttls_pred, ~]  = li2nsvm_multiclass_fwd(S_test', U, b, class_list);
        t = toc;
        
        time = t/length(ttls)
        
        reco_rate       =  (sum(ttls_pred'==ttls))/length(ttls);
        
        disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
        %     fprintf(['\nThe Recognition Rate on the ', dname, ' is ', num2str(roundn(reco_rate,-4)),' !\n']);
        reco_rates = [reco_rates, reco_rate ];
        
        %     plot_dict(model);
        
    end
    
    recos{dataset} = reco_rates;
    
end

save 'semmdl_recos' recos;
reco_rates
recos
