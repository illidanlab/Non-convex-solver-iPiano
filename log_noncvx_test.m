% testing problem: f(x) = 1/2 \sum_i log(1 + mu(x_i  - u_i)^2 )
% g(x) = lambda |x|_1

function [x,output] = log_noncvx_test(x0, mu, u, rho1, options)

gradF_solve    =  @(x) gradF(x, mu, u);
valueF_solve   =  @(x) valueF(x, mu, u);
proximal_solve =  @(x, t) proximal(x,t,rho1);
valueG_solve   =  @(x) valueG(x, rho1);

[x, output] =  ciPiano(gradF_solve ,valueF_solve, proximal_solve, valueG_solve, x0, options);

end



function y = gradF(x, mu, u)

    y = mu * (x - u)./(1 + mu*(x - u).^2);

end

function y = valueF(x, mu, u)

    y = 0.5 * sum( log(1+ mu*(x-u).^2));

end
 
 
function y = proximal(x, t, para)

   y = max(0, abs(x)- para*t).*sign(x);

end


function y = valueG(x, para)

   y = sum(abs(x))*para;

end
 