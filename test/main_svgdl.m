
%% script for testing the discriminative dictionary learning code - mmdl

clear all; clc;
warning off;

%% load toolkits
addpath('.\large_scale_svm');

%% load dataset
addpath('.\data');

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
        num_atom_per_class = 9;
        dname = 'MNIST dataset';
        
    elseif (dataset == 4)
        load USPS
        num_atom_per_class = 10;
        dname = 'USPS dataset';
        %         %Test 1
        %         load session1_05_1_netural_14
        %         Train_DAT = double(DAT(:,labels<=60));
        %         trainlabels = labels(:,labels<=60);
        %         clear DAT labels;
        %         load session3_05_1_netural_10o
        %         Test_DAT = double(DAT(:,labels<=60));
        %         testlabels = labels(:,labels<=60);
        %         clear DAT labels;
        %         %Test 2
        %         load session1_05_1_smile_14
        %         Train_DAT = double(DAT(:,labels<=60));
        %         trainlabels = labels(:,labels<=60);
        %         clear DAT labels;
        %         load session3_05_1_smile_10o
        %         Test_DAT = double(DAT(:,labels<=60));
        %         testlabels = labels(:,labels<=60);
        %         clear DAT labels;
        %
        %         num_atom_per_class = 14;
        %         dname = 'Multi-PIE dataset';
    end
    
    tr_dat = Train_DAT;
    tt_dat = Test_DAT;
    trls = trainlabels;
    ttls = testlabels;
    
    clear Train_DAT Test_DAT trainlabels testlabels;
    
    
    %% reduce the dimension
    
    rdim = 300;
    
    Vt = Eigenface_f(tr_dat,rdim);
    tr_dat = Vt'*tr_dat;
    tt_dat = Vt'*tt_dat;
    
    %% ¹éÒ»»¯
    tr_dat = tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [size(tr_dat,1),1]) );
    tt_dat = tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [size(tt_dat,1),1]) );
    
    %% set parameters
    lambda1    =   2e-3;
    lambda2    =   1e-6;
    max_iter   =   25;
    theta      =   5;
    
    %% initialize sub-dictionaries via pca
    fprintf('\n------------------------Initializing Dictionary------------------------\n\n');
    Dini = [];
    num_class   = length(unique(trls));
    num_atom_ci = num_atom_per_class;
    fprintf('class:');
    for ci = 1:num_class
        tr_dat_ci           =    tr_dat(:,trls==ci);
        [Dini_ci,~,mean_ci] =    Eigenface_f(tr_dat_ci,num_atom_ci-1);
        Dini_ci             =    [Dini_ci mean_ci./norm(mean_ci)];
        Dini                =    [Dini Dini_ci];
        fprintf('%d', ci);
        if ~mod(ci, 20),
            fprintf('.\n      ');
        else
            fprintf('.');
        end;
    end
    fprintf('\n\nInitialization Is Done!')
    
    %% run algorithm
    fprintf('\n\n----------------------------Algorithm SVGDL ----------------------------\n\n');
    [D,Z,U,b,class_list]  = svgdl(tr_dat,trls,Dini,lambda1,lambda2,theta,max_iter);
    fprintf('\n\nSVGDL Model Training Is Completed!')
    
    %% encode the testing data
    fprintf('\n\n--------------------------------Testing--------------------------------\n\n');
    P = inv(D'*D+lambda1*eye(size(D,2)))*D';
    Z_test = P*tt_dat;
    
    [ttls_pred, ~] = li2nsvm_multiclass_fwd(Z_test', U, b, class_list);
    reco_rate  =  (sum(ttls_pred'==ttls))/length(ttls);
    disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
%     fprintf(['\nThe Recognition Rate on the ', dname, ' is ', num2str(roundn(reco_rate,-4)),' !\n']);
end