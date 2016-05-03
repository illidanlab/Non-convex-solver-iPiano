% Non-convex multi-task 

function [x,output] = argmin_MTIL_S_Ln(x0, Xtrain, Ytrain, inputs, options)

X = Xtrain;
Y = Ytrain;
% W_ini = inputs.W;
% B_ini = inputs.B;
% q_ini = inputs.q;
d = inputs.d;
K = inputs.K;
r = inputs.r;
lambdaB = inputs.lambdaB;
lambdaq = inputs.lambdaq;
lambdaW = inputs.lambdaW;


% grad_W_vec = reshape(W_ini,d*K,1);
% grad_B_vec = reshape(B_ini,d*r,1);
% grad_q_vec = reshape(q_ini,r*r*K,1);
% x0 = [grad_W_vec;grad_B_vec;grad_q_vec];

gradF_solve    =  @(x) gradF(x,X,Y,K,d,r,lambdaB,lambdaq);
valueF_solve   =  @(x) valueF(x,X,Y,d ,K, r,lambdaB,lambdaq);
proximal_solve =  @(x, t) proximal(x,t,lambdaW, d,K);
valueG_solve   =  @(x) valueG(x, lambdaW,d,K);

[x, output] = nmiPiano(gradF_solve ,valueF_solve, proximal_solve, valueG_solve, x0, options);
% [x, output] = iPiano(gradF_solve ,valueF_solve, proximal_solve, valueG_solve, x0, options);
end

 

function y = gradF(z,X,Y,K,d,r,lambdaB,lambdaq)

     W = reshape(z(1:d*K),d,K);
     q = reshape(z(d*K+1:d*K+r*r*K),r,r,K);
     
     B = reshape(z(d*K+1+r*r*K:end),d,r);
     
     
    [grad_W,f] = ncvBCDgradW(X,Y,W,q,B,d,K,r);
    grad_W_vec = reshape(grad_W,d*K,1);

    
    [grad_q,f] = ncvBCDgradqreg(X,Y,W,q,B,d,K,r,lambdaq);
    grad_q_vec = reshape(grad_q,r*r*K,1);
         
    [grad_B,f] = ncvBCDgradBreg(X,Y,W,q,B,d,K,r,lambdaB);
    grad_B_vec = reshape(grad_B,d*r,1);
    
    y = [grad_W_vec;grad_q_vec;grad_B_vec];

end

function y = valueF(z,X,Y,d ,K, r,lambdaB,lambdaq)

     W = reshape(z(1:d*K),d,K);
     q = reshape(z(d*K+1:d*K+r*r*K),r,r,K);
     B = reshape(z(d*K+1+r*r*K:end),d,r);
     y = smooth_funcvalueNcvreg(X,Y,W,q,B,d,K,r,lambdaB,lambdaq);
 
end
 
 
function y = proximal(x, t, para,d,K)
   W = reshape(x(1:d*K),d,K);
   
   W_new = prox_tr(W,para*t);
   
   x(1:d*K) = W_new(:);
   
   y = x;
end


function y = valueG(x, para,d,K)
   
    W = reshape(x(1:d*K),d,K);
    y = trace_norm(W) * para;

end
 


