% IPIANO Algorithm 4 nmiPiano & Algorithm 2 ciPiano implementation.

function [x, output] = iPiano(gradF ,valueF, proximal, valueG, x0, options)
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
 
c2    = 1e-8; % c2 in paper algorithm 3.
c1    = 1e-8;
step  = 10000;

iter = 0;
Lk = L;
 
 


%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%
 
Lk = estimate_lip(xk);
beta_k = 0.5;
alpha = 2*(1-beta_k)/Lk; 
alpha_k = alpha;  % alpha_0

if alpha_k < c1
    disp(sprintf('Cannot choose alpha_k= %f > c1 = %f',alpha_k, c1))
end
    first_part = 1/alpha_k - Lk/2;
    second_part = beta_k/alpha_k;
    delta_k = first_part - second_part/2; 
    gamma_k = first_part - second_part;

while alpha_k > c1

    first_part = 1/alpha_k - Lk/2;
    second_part = beta_k/alpha_k;
    delta_k = first_part - second_part/2; 
    gamma_k = first_part - second_part;

    if delta_k >= gamma_k && gamma_k >= c2  
        break;
    end

    alpha_k = alpha_k - (alpha_k - c1)/step;

end 

%%%%%%%%%%%%% End of Initialization %%%%%%%%%%%%%%%%%%%
% Find alpha_0 iteratively with fixed beta_0 and estimate L_0

 


while iter < maxIter



    % compute smooth noncvx function value and gradients of the search point
    fg_yk  = gradF  (xk); % gradient
    fv_yk  = valueF (xk); % function value

    Lk = estimate_lip(xk); % Every iteration, estimate the lipschtiz constant 
    
    iter_Lip = 0;
    while true %start line search 
 
        Lk = Lk * eta;
        delta_k_1 = delta_k;
        
        b  =  (delta_k + Lk/2)/(c2 + Lk/2);
        beta_k =  (b - 1)/(b-0.5);
        beta = (b - 1)/(b-0.5);
        while beta_k > 0
            
            take = false; % take current beta_k or not
            
            alpha = 2*(1-beta_k)/(Lk );    %%%%%%%%%%%% alpha = 2*(1-beta_k)/(Lk) ??
            alpha_k = alpha;
            while alpha_k > c1
      
                first_part = 1/alpha_k - Lk/2;
                second_part = beta_k/alpha_k;

                delta_k = first_part - second_part/2; 
                gamma_k = first_part - second_part;

                if delta_k <= delta_k_1 && delta_k >= gamma_k && gamma_k >= c2  %%%%%%%%%%%%delta_k < delta_k_1 ??
                    take = true;
                    break;
                end
                alpha_k = alpha_k - (alpha - c1)/step;
            end
            
            if take
                break;
            end
            
            
            beta_k = beta_k - beta/step;
            
        end
        
       %%%%%%%%%%%%  check lipschitz constant %%%%%%%%%%%%%%%%% 
        yk  = xk - alpha_k * fg_yk + beta_k*(xk - xk_1);
        xk_new = proximal(yk,  alpha_k);

        Fv_plyk = valueF(xk_new);
        delta   = xk_new - xk;
        q_apro  = fv_yk + delta'* (fg_yk + Lk/2 * delta);
        if (Fv_plyk   <= q_apro)
            break;
        else if (iter_Lip > 5000)
                disp('Cannot find local Lipschtiz constant')
            end
        end
        %%%%%%%%%%%%  check lipschitz constant %%%%%%%%%%%%%%%%% 
        iter_Lip = iter_Lip + 1;
        

    end  % end of line search 
          
    
%     Lk_1 = Lk;
 
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


    function Lip = estimate_lip(x0) % Estimate lipschitz constant from current point xk
        fg_xk_est  = gradF  (x0); % gradient
        fv_xk_est  = valueF (x0); % function value
        yk_est     = x0 -   fg_xk_est;
        xk_new_est = proximal(yk_est,  1);
        fv_xk_new_est  = valueF (xk_new_est); % function value


        Lip = abs(fv_xk_new_est - fv_xk_est)/sqrt(sum((x0-xk_new_est).^2));
    end

end



 