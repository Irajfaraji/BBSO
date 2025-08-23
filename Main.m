%--------------------------------------------------------------------------
% Boxelder Bug Search Optimization (BBSO)
% Based on the paper: "Boxelder Bugs Search Optimization: A Novel Reliable Tool for Optimizing Engineering Problems Through Bio-Inspired Ecology of Boxelder Bugs"
% Authors: Iraj Faraji Davoudkhani, Hossein Shayeghi, Abdollah Younesi
% Neural Computing and Applications (2025)  ISSN: 0941-0643 ,
% https://doi.org/.
% e-mail : faraji.iraj@gmail.com
%--------------------------------------------------------------------------

clear all;
close all;
clc;

disp('Boxelder Bug Search Optimization (BBSO)');

% Select Benchmark Function (e.g., 'F1' for Sphere)
Function_name = 'F1';
Max_iteration = 300000;  % Maximum Iterations (Adjust based on Evaluations)
nPop = 100;            % Population Size

% Load Function Details
[lb, ub, dim, fobj] = Get_Functions_details(Function_name);

%% Run BBSO
[Best_score, Best_pos, cg_curve] = BBSO(nPop, Max_iteration, lb, ub, dim, fobj);


% Visualization
figure(1);
subplot(1, 2, 1);
func_plot(Function_name);
title('Test Function');
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name, '(x_1, x_2)']);
grid off;

subplot(1, 2, 2);
semilogy(cg_curve, 'Color', 'b','linewid',2);
title('Convergence Curve');
xlabel('Iteration');
ylabel('Best Score');
axis tight;
grid off;
box on;
legend('BBSO');
set(gcf,'position',[250,200,800,400]) 

% Display Results
display(['The best solution obtained by BBSO is: ', num2str(Best_pos)]);
display(['The best optimal value found by BBSO is: ', num2str(Best_score)]);