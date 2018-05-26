
clear all;clc;
warning off;

%% load toolkits
addpath('.\large_scale_svm');
addpath('.\dictionary_learning');
addpath('.\FOptM');
addpath('.\SEMMDL');

%% load dataset
addpath('.\data');

max = 0.0;
best_l1 = 0; 
best_l2 = 0;
best_l3 = 0;

cnt = 1;

Lambda1     = 1e-3 : 0.005 : 0.05;
Lambda2     = 0.5 : 0.1 :1.5;

len_i       = length(Lambda1);
len_j       = length(Lambda2);

Accurcy     = zeros(len_i,len_j);      

for i         = 1 : len_i
    for j         = 1 : len_j
        for lambda3     = [1e-6] %[1e-6 5e-6 1e-5 2e-5 5e-5 8e-5 1e-4 2e-4 5e-4 1e-3]
            
            %% choose data set
            for dataset=[5]
                
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
                    
                elseif (dataset == 5)
                    load scene15_f100
                    num_atom_per_class = 10;
                    dname = 'Scene15';
                    
                elseif (dataset == 6)
                    load caltech101_f5
                    num_atom_per_class = 5;
                    dname = 'caltech101_f5';
                    
                elseif (dataset == 7)
                    load caltech101_f10
                    num_atom_per_class = 5;
                    dname = 'caltech101_f10';
                    
                elseif (dataset == 8)
                    load caltech101_f15
                    num_atom_per_class = 5;
                    dname = 'caltech101_f15';
                    
                elseif (dataset == 9)
                    load caltech101_f20
                    num_atom_per_class = 5;
                    dname = 'caltech101_f20';
                    
                elseif (dataset == 10)
                    load caltech101_f25
                    num_atom_per_class = 5;
                    dname = 'caltech101_f25';
                    
                elseif (dataset == 11)
                    load caltech101_f30
                    num_atom_per_class = 5;
                    dname = 'caltech101_f30';
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
                
                %% para setting ... 
                lambda1         = Lambda1(i);
                lambda2         = Lambda2(j);
                
                param.rdim      = 300; % 169--96.037%
                param.f         = num_atom_per_class;
                param.max_iters = 1; % 30
                param.lambda1   = lambda1;
                param.lambda2   = lambda2;
                param.lambda3   = lambda3;
                
                param.theta     = 5; % 7 - > 97.205%
                param.draw      = false;
                
                %% Running SEMMDL model ...
                tic;
                [ model ] = semmdl( tr_dat, trls,tt_dat,ttls, param );
                toc;
                
                %% Result ...
                reco = model.reco_rates(end);
                Accurcy(i,j) = reco;
                
                %% Pick up the best ret!
                if reco > max
                    
                    best_l1 = lambda1;
                    best_l2 = lambda2;
                    best_l3 = lambda3;
                    
                    max     = reco;
                end
                
                %% Print info
                fprintf ('\nChoose best progrom running ..., Iter = %d\n',cnt);
                fprintf ('\nreco = %f, at l1 = %f, l2 = %f, l3 = %f',reco,lambda1,lambda2,lambda3);
                fprintf ('\nmax  = %f, at l1 = %f, l2 = %f, l3 = %f\n\n',max,best_l1,best_l2,best_l3);
                
                %% Increase iter ...
                cnt     = cnt + 1;
                
            end
        end
    end
end

fprintf ('\nChoose best progrom EDN');
fprintf ('\nmax  = %f, at l1 = %f, l2 = %f, l3 = %f\n\n',max,best_l1,best_l2,best_l3);

[X,Y] = meshgrid(Lambda1,Lambda2);
mesh(X, Y, Accurcy');

hold on
plot3(best_l1,best_l2,max,'bo','markerfacecolor', [ 0, 0, 0 ])

xlabel('\lambda_1');
ylabel('\lambda_2');
zlabel('accury');



