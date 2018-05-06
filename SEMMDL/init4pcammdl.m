function [ Pini,Bini,Uini,bini ] = init4pcammdl( tr_data,tr_label,rdim,num_atom_ci )

%%
% X     归一化后的训练数据

%% 归一化
X       = normalize_mat(tr_data);

%% 1.初始化投影矩阵P
M       = Eigenface_f(tr_data,rdim); % P = M'
P       = M';

%% 低维数据
P_X     = P*X;

%% 2.初始化字典B
disp('Init dictionary ... ');

num_class   = length(unique(tr_label));
Dini        = [];

for ci = 1:num_class
    
    tr_dat_ci           =    P_X(:,tr_label==ci);
    [Dini_ci,~,mean_ci] =    Eigenface_f(tr_dat_ci,num_atom_ci-1);
    Dini_ci             =    [Dini_ci mean_ci./norm(mean_ci)];      % 每一类的初始字典为降维和均值
    
    Dini                =    [Dini Dini_ci];
    
    fprintf('%d', ci);  
    if ~mod(ci, 20),
        fprintf('.\n      ');
    else
        fprintf('.');
    end
end
fprintf('\n');

m           = size(P_X,1);
n           = length(y);
K           = size(Dini,2);

class_list  = unique(y,'stable');
class_num   = length(class_list);
class_space = 1;
class_idx   = zeros(n, 1);

% define the label matrix Y_label for c two-class classification problems (one-vs-all)
Y           = zeros(n, class_num);
for i = 1 : n
    for j = 1 : class_space
        if y(i) == class_list(j) 
            class_idx(i) = j;
        end
    end
    if class_idx(i) == 0
        class_space = class_space + 1;
        class_idx(i) = class_space;
    end
    Y(i, class_idx(i)) = 1;
end

Y_label = sign(Y-0.5);

% initialize Z U b
Sinit       = zeros(K,n);
Uinit       = zeros(m,class_num);
binit       = zeros(1,class_num);


%% output
Pini    = P;
Bini    = Dini;



end

