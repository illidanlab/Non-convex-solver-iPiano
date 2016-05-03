
options = struct(...
                 'maxIter', 1000,...
                 'ftol',    1e-6,... 
                 'bFlag',   0, ...    % 0  this flag tests whether the gradient step only changes a little
                 'tFlag',   3, ...    % 3  the termination criteria
                 'beta',    0.75,  ...   % inertial step size.
                 'alg_version', 5, ... % choose algorithm version  1: fixed lipschitz constant 2: line search
                 'lip_const', 100 ... %  Need to specify the lip_const for ciPiano
);

addpath(genpath('./'))
mu = 100;
u  = [1,1]';
x0 = [ -2,  2]';
rho1 = 1;
[x, output] = log_noncvx_test(x0, mu, u, rho1, options);

hFig = figure;
% set(hFig, 'Position', [50,50, 1000,400]);
% subplot(1,2,1);
plot( output.funcVal);

 
xpoints = output.x_points;

%% Contour of non-convex objective function
xmin = -2;
xmax = 2;
rho1 = 1;

figx = linspace(xmin,xmax);
figy = linspace(xmin,xmax);
[X,Y] = meshgrid( figx,figy);
Z = 0.5 *  (log(1+ mu*((X-u(1)).^2)) + log(1+ mu*((Y - u(2)).^2))) + rho1 *(abs(X) + abs(Y));
 
% subplot(1,2,2);
% contour(X,Y,Z,80,'ShowText','on');
contour(X,Y,Z,50);
hold on;
plot(0,1,'c*','MarkerSize',10,'MarkerEdgeColor','r', 'MarkerFaceColor','r');
plot(1,0,'c*','MarkerSize',10,'MarkerEdgeColor','r', 'MarkerFaceColor','r');
plot(1,1,'c*','MarkerSize',10,'MarkerEdgeColor','r', 'MarkerFaceColor','r');
plot(0,0,'c*','MarkerSize',10,'MarkerEdgeColor','r', 'MarkerFaceColor','r');
hold on;

a = linspace(7,7,size(xpoints,2));
scatter( xpoints(1,:),xpoints(2,:),a,'filled','MarkerEdgeColor',[0 0 0],...
              'MarkerFaceColor',[0 0 0]);

% print('./contour_log.pdf','-dpdf');
print(sprintf('./clean_beta%d_startpoint%d_%d.pdf',options.beta*100,x0(1), x0(2)),'-dpdf')
%%%%%%%%%%%%%%%%
% surf(X,Y,Z);
% % colormap([0  0  0])
% view([-0.2,-1,1.5]);
% xlabel('x');
% ylabel('y')
% 
% % print('./3dplot_log.pdf','-dpdf');

