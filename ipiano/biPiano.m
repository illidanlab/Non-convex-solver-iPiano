% IPIANO Algorithm 4 nmiPiano & Algorithm 2 ciPiano implementation.

function [x, output] = biPiano(gradF ,valueF, proximal, valueG, x0, options) %checked
% INPUT: 
% gradF gradient of non-convex smooth component, 
% valueF function value of non-convex smooth component
% proximal: proximal mapping, map the point to the feasible set.
% valueG function value of convex smooth component
% input variables x.

 
funcVal = [];
x_points = [];

% ============== Process options ==============
maxIter      = options.maxIter;
ftol         = options.ftol;
lip_const    = options.lip_const; % When you choose alg 2, you need to specify the lip const.
alg_version  = options.alg_version; 
% alg_version = 2 corespond to algorithm 2 in iPiano paper.(fixed lipschitz constant/steps size)
% alg_version = 4 corespond to algorithm 4 in iPiano paper. (line search.)
% alg_version = 5 corespond to algorithm 5 in iPiano paper. (line search.)

% flags of optimization algorithms .
bFlag        = options.bFlag; % 0  this flag tests whether the gradient step only changes a little
tFlag        = options.tFlag; % 3  the termination criteria
beta         = options.beta; % step of inertial step size.  

% initialize a starting point
xk   = x0; % x_k
xk_1 = x0; % x_{k-1}
 
% initialize other variables.
L     = 0.1; % initial L  
eta   = 1.1; % increment of L
eta2  = 0.9; % decreasement of L.
c2    = 1e-6; % c2 in paper algorithm 3.

 

iter = 0;
Lk = L;
 
% algorithm 5  parameters
sigma_k = c2 + 1e-6; % >= c2
b  =  (sigma_k + Lk/2)/(c2 + Lk/2);
beta_k = (b - 1)/(b-0.5);



%%% Debug
sigmas = [];


while iter < maxIter



    % compute smooth noncvx function value and gradients of the search point
    fg_yk  = gradF  (xk); % gradient
    fv_yk  = valueF (xk); % function value
 
            
   while true

    b  =  (sigma_k + Lk/2)/(c2 + Lk/2);
    beta_k =  (b - 1)/(b-0.5);
    alpha = 2*(1-beta_k)/(Lk + 2*c2);

    yk  = xk - alpha * fg_yk + beta_k*(xk - xk_1);
    xk_new = proximal(yk,  alpha);

    Fv_plyk = valueF(xk_new);
    delta   = xk_new - xk;
    q_apro  = fv_yk + delta'* (fg_yk + Lk/2 * delta);


    if (Fv_plyk   <= q_apro)
        break;
    end
    Lk = Lk * eta;


   end
 
 
    % update current and previous solution.
    xk_1 = xk;
    xk = xk_new;

    % concatenate function value  

    funcVal = cat(1, funcVal, Fv_plyk + valueG(xk_new));
    x_points = cat(2, x_points, xk);
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end

    % test stop condition.
    switch(tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= ftol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= ftol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= ftol)
                break;
            end
        case 3
            if iter>= maxIter
                break;
            end
    end

    % update other variables.
    iter = iter + 1;
 

end

x = xk;


  
  output = struct( ...
    'x'       , x       ,...
    'funcVal'  ,   funcVal  ,...
    'x_points'      , x_points ... 
    );


end

 