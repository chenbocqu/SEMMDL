function [ model ] = semmdl( tr_data, trls, tt_dat, ttls, para )

%% copy parameters
rdim        = para.rdim;        % ��άά��
Max_iters   = para.max_iters;   % ����������
num_atom_ci = para.f;           % ÿ���ֵ�ԭ�Ӹ���
lambda1     = para.lambda1;     % lambda1 ||S||_1
lambda2     = para.lambda2;     % PCAԼ���������lambda2 ||X-P^tPX||^2
lambda3     = para.lambda3;     % lambda3 SVM
theta       = para.theta;       % SVM ����

if isfield(para,'VERBOSE')
    VERBOSE = para.VERBOSE;
else
    VERBOSE = false;
end

if isfield(para,'draw')
    draw    = para.draw;
else
    draw    = false;
end

if isfield(para,'tol')
    tol     = para.tol;
else
    tol     = 1e-3;
end

%% ��ʼ�� ͶӰ����P & �ֵ�B
M       = Eigenface_f(tr_data,rdim); % P = M'
P       = M';

X       = normalize_mat(tr_data);
y       = trls;
tau     = 1/theta; 

%% 
P_X     = P*X;

%% 
num_class   = length(unique(trls));
Dini        = [];

%% ��ʼ���ֵ�
fprintf('Initing Dictionary >>> \n      ');
for ci = 1:num_class
    
    tr_dat_ci           =    P_X(:,trls==ci);
    [Dini_ci,~,mean_ci] =    Eigenface_f(tr_dat_ci,num_atom_ci-1);
    Dini_ci             =    [Dini_ci mean_ci./norm(mean_ci)];      % ÿһ��ĳ�ʼ�ֵ�Ϊ��ά�;�ֵ
    
    Dini                =    [Dini Dini_ci];
    
    fprintf('%d', ci);  
    if ~mod(ci, 20),
        fprintf('.\n      ');
    else
        fprintf('.');
    end
end
fprintf('\nIniting End ! \n');

%% MMDL
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

disp('Start algorithm SEMMDL ... ');

%% 
k           = 0;
rel_deltaD  = 1;
J           = [];

S           = Sinit;
U           = Uinit;
b           = binit;
B           = Dini;

reco_rates  = []; % ���������е��ٻ���

%% Start iterating ...
while  k < Max_iters && rel_deltaD > tol
    
    k=k+1;
    
    fprintf('\nIteration: %i, ', k);
    
   %% step 1 , �̶�P,[w,b],���ֵ�ѧϰ
    P_X         = P * X;    % �����ݽ�ά
    
   %% step 1.1 , update S
    Pk          = inv(B'*B+lambda1*eye(size(B',1)));
    
    if k~=1
        for i = 1 : n
            Y_labelki = S(:,i)'*U + b;                          % Ԥ���Y
            loss_idx  = find( Y_labelki.*Y_label(i,:) < 1 ) ;   % Ԥ������id
            if isempty(loss_idx)
                S(:,i) = Pk*B'*P_X(:,i);
            else
                Yi_idx = Y_label(i,loss_idx);
                Uk_idx = U(:,loss_idx);
                bk_idx = b(loss_idx);
                
                ski = B'*P_X(:,i)+2*lambda3*theta*(Uk_idx*Yi_idx'-Uk_idx*bk_idx');
                Tki = inv(eye(size(Uk_idx,2))+2*lambda3*theta*Uk_idx'*Pk*Uk_idx);
                
                S(:,i)=(Pk-2*lambda3*theta*Pk*Uk_idx*Tki*Uk_idx'*Pk)*ski;
                % equivalent to Zk(:,i)=inv(Dk'*Dk+lambda1*eye(size(Dk',1))+2*lambda3*theta*Uk_idx*Uk_idx')*ski based on Woodbury formula, but more efficient.
            end
        end
    else
        S = Pk * B' * P_X;
    end
    
    %% step 1.2 , update B
    
    % ����ǰһ���ֵ�
    B_pre       = B;  
    B           = l2ls_learn_basis_dual(P_X, S, 1);
    
    rel_deltaD  = norm(B(:)-B_pre(:))/norm(B(:));
    
    if  VERBOSE
        fprintf('\b  relative change of D = %g', rel_deltaD);
    end
    
    %% step 2 , update Uk bk
    [U, b, ~] = li2nsvm_multiclass_lbfgs(S',y, tau);
    
    %% step 3 , update P
    P           = update_P( P, X, B, S,lambda2 );

    %% compute hing loss
    hinge_loss      = zeros(1,n);
    for i = 1 : n
        Y_labelki = S(:,i)'*U + b;                          % Ԥ���Y
        loss_idx  = find( Y_labelki.*Y_label(i,:) < 1 ) ;   % Ԥ������id
        if isempty(loss_idx)
            continue;
        else      
            hinge_loss(n) = norm (1- Y_labelki(loss_idx).*Y_label(i,loss_idx),2)^2;
        end
    end
    hinge_loss_sum = sum (hinge_loss);
    
    %% print lost J
    j           = norm(P*X-B*S,2)^2 + lambda1*norm(S,2) + lambda2*norm(X-P'*P*X,2)^2 +...
                  2 * lambda3 * ( norm(U'*U,2) + theta * hinge_loss_sum);
    J           = [J,j];
   
    %% ����
    tt_dat      = normalize_mat(tt_dat);
    
    S_test      = inv(B'*B+lambda1*eye(size(B,2)))*B'*P * tt_dat;
    
    [ttls_pred, ~]  = li2nsvm_multiclass_fwd(S_test', U, b, class_list);
    reco_rate       =  (sum(ttls_pred'==ttls))/length(ttls);
    
    reco_rates = [reco_rates,reco_rate ];
    
    fprintf(['\nSEMMDL :iter = ',num2str(k),';\nTest accuracy = ',num2str(roundn(reco_rate*100,-3)), '%;\nJ = ', num2str(j) '.\n']);
    
end


if draw
   
   %% ��������ͼ���ӵڶ�����ʼ����������
   figure;
   plot     ( J(:,2:end),'bo-',...
                         'MarkerEdgeColor','b',...
                         'MarkerFaceColor','w',...
                         'MarkerSize',4);
                     
   xlabel   ( '��������' );
   ylabel   ( '�Ż�Ŀ��J' );
   
   grid on;
   
   %% ���ӻ��ֵ� (BΪ��ά�ֵ䣬P' * B Ϊ�ع��ĸ�ά�ֵ�)
   figure;ImD=displayPatches(B);      % ���ӻ��ֵ�
   imagesc(ImD); colormap('gray');
   
end

%% ������
model.P     = P;
model.B     = B;
model.U     = U;
model.b     = b;
model.J     = J;

model.class_list = class_list;
model.reco_rates = reco_rates;

end

