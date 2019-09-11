%% Steepest Descent
close all
clear all
clc

% Define paramaters
m1 = [-10, -10]';
m2 = [10, 10]';
A1 = [1, 0.5; 0.5, 1];
A2 = [1, -0.5; -0.5, 1];

f1 = @(x1)J(x1,m1,m2,A1,A2);
f2 = @(x2)J_gradient(x2,m1,m2,A1,A2);

x0 = [8, 6]';
N = 200;
beta = 0.05;

[x_min, J_min] = steepest_descent(x0, f1, f2, N, beta);

%% Simplex
close all
clear all
clc

% Define paramaters
m1 = [-10, -10]';
m2 = [10, 10]';
A1 = [1, 0.5; 0.5, 1];
A2 = [1, -0.5; -0.5, 1];

f1 = @(x1)J(x1,m1,m2,A1,A2);

x0 = [8, 6]';
N = 1000;
beta = 0.05;

[x_min, J_min] = simplex(x0, f1, N, beta);