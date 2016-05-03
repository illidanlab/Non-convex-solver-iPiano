addpath(genpath('../../../'))

configureFile;


%% set parameters: 
dim  = 100;
task = 10;
samp = 500;
datatype = 'sp'; 

dataDir = configurePara.inDataDir;
dataname = sprintf('Syn_sp_mtl_dim%d_task%d_samp%d.mat',dim,task,samp);
realdata = strcat(dataDir, dataname);
load_data = load(realdata);
 
 
% Load data
Xtest  = load_data.Xtest;
Ytest  = load_data.Ytest;
Xtrain = load_data.Xtrain;
Ytrain = load_data.Ytrain;

d = size(Xtrain{1}, 2);
K = size(Ytrain,1);
r = configurePara.MTIL_L_Ln.tunedParas.rank(1);
lambdaB = configurePara.MTIL_L_Ln.tunedParas.lambdaB(1);
lambdaq = configurePara.MTIL_L_Ln.tunedParas.lambdaq(1);
lambdaW = configurePara.MTIL_L_Ln.tunedParas.lambdaW(1);

inputs = struct(...
        'X',    Xtrain, ...
        'Y',    Ytrain, ...
        'd',    d,      ...
        'K',    K,      ...
        'r',    r,      ...
  'lambdaB',    lambdaB,...
  'lambdaq',    lambdaq,...
  'lambdaW',    lambdaW ...
);

%         'W',    W,      ... 
%         'B',    B,      ...
%         'q',    q,      ...

%%% optimal initialization
% timeflag1 = '27-Mar-201611-24';  % test
% dataload = load(sprintf(strcat(configurePara.resultDir,'%s_%s_%s'), 'MTIL_L_Lc', timeflag1  ,dataname));
% cvx_rmse = cell2mat(dataload.RMSE);
% cvx_idx  = find(cvx_rmse==min(min(cvx_rmse)));
% cvx_W_all    = dataload.W_all;
% cvx_Q_all    = dataload.Q_all;
% W_ini = cvx_W_all{cvx_idx};
% Q_ini = cvx_Q_all{cvx_idx};
% [B_ini,q_ini,F] = block_coordinate(Q_ini, r ,FISTA_OPT);
% grad_W_vec = reshape(W_ini,d*K,1);
% grad_B_vec = reshape(B_ini,d*r,1);
% grad_q_vec = reshape(q_ini,r*r*K,1);
% x0 = [grad_W_vec;grad_B_vec;grad_q_vec];

%%% random initialization
x0 = rand(d*K+r*r*K+d*r, 1);    


% solution
tic;
[x,output] = argmin_MTIL_S_Ln(x0, Xtrain, Ytrain, inputs, options);
toc;

 plot(output.funcVal)
 
 output.funcVal(end)











% x1 = [-0.5:0.01:0.5];
% x2 = [0:0.01:1];

% [X,Y] = meshgrid(-0.5:0.5, 0:1);
% Z = 100.*(Y - X.^2).^2 + (1 - X).^2;

%   Solving Dual Problems Using a Coevolutionary Optimization Algorithm 5.3
% figure
% surf(X,Y,Z);
% az =  0;
% el = 10;
% view(az, el);
% colorbar


 