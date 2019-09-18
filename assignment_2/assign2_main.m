%% Task A and B
clc
clear all
close all

% Set how many dimensions observations should have, 
% n_dim = 2 will correspond to the underlying model
n_dim = 10;

% Define paramaters
a = 0.5;
b = 2;
c1 = 2;
c2 = 3;

% Set how many train examples should be generated
N_train = 200;

% Generate 10 dimensional vectors of observations
X_train = normrnd(0, 2, n_dim, N_train);

% Generate the response, only dependent on x1 and x2
Y_train = zeros(N_train, 1);
C = [c1 c2]';
func1 = @(x)f(x, a, b, C);

for i=1:N_train
    
    Y_train(i) = func1(X_train(1:2, i));
    
end




func = @(theta)J_for_f(X_train, Y_train, theta);
func_gradient = @(theta)J_for_f_gradient(X_train, Y_train, theta);

% Set parameters for steepest descent
N = 10000;
beta = 0.05;

% Set starting guess for parameters
x0 = ones(12,1);
% x0(1) = a + 1;
% x0(2) = b + 1;
% x0(3) = c1 + 1;
% x0(4) = c2 + 1;

% Run steepest descent
[x_min, J_min] = steepest_descent(x0, func, func_gradient, N, beta);

figure

[x_min_simp, J_min_simp] = simplex(x0, func, N, beta);

% Extract "optimised" parameters
%a_min = x_min(1);
%b_min = x_min(2);
%C_min = x_min(3:end);

% Compare to matlabs steepest descent
options = optimoptions('fminunc','SpecifyObjectiveGradient',true);
REFERENCE = fminunc(func, x0, options);

%% Task C
clc
clear all
close all

% Define paramaters
a = 0.5;
b = 2;
c1 = 2;
c2 = 3;

% Set how many dimensions observations should have, 
% n_dim = 2 will correspond to the underlying model
n_dim = 10;

C = [c1 c2]';
func1 = @(x)f(x, a, b, C);

% Set how many train examples should be generated
N_train = 200;

% Set number of sets
N_sets = 100;

D = cell(1, 100);

for j=1:N_sets
    % Generate 10 dimensional vectors of observations
    X_train = normrnd(0,2,n_dim,N_train);

    % Generate the response, only dependent on x1 and x2
    Y_train = zeros(N_train, 1);
    

    for i=1:N_train

        Y_train(i) = func1(X_train(1:2, i));

    end
    
   D{j} = [X_train' Y_train]; 
    
end

% Collect model parameter estimates
a_vector = zeros(1, 100);
b_vector = zeros(1, 100);
c_matrix = zeros(n_dim, 100);

% a_vector_own = zeros(1, 100);
% b_vector_own = zeros(1, 100);
% c_matrix_own = zeros(n_dim-2, 100);

a_vector_simp = zeros(1, 100);
b_vector_simp = zeros(1, 100);
c_matrix_simp = zeros(n_dim, 100);

for i=1:N_sets
    
    D_train = D{i};
    X_train = D_train(:, 1:end-1)';
    Y_train = D_train(:, end);
    
    func = @(theta)J_for_f(X_train, Y_train, theta);
    func_gradient = @(theta)J_for_f_gradient(X_train, Y_train, theta);

    % Set parameters for steepest descent
    N = 10000;
    beta = 0.05;

    % Set starting guess for parameters
    X_size = size(X_train);
    d = X_size(1) + 2;
    x0 = ones(d, 1);

    % Run steepest descent
    %[x_min_own, J_min_own] = steepest_descent(x0, func, func_gradient, N, beta);
    %options = optimoptions('fminunc','SpecifyObjectiveGradient',true);
    %x_min = fminunc(func, x0, options);
    x_min = fminsearch(func, x0);
    
    [x_min_simp, J_min_simp] = simplex(x0, func, N, beta);
    
    %figure
    close all
    

    % Extract "optimised" parameters
    a_min = x_min(1);
    b_min = x_min(2);
    C_min = x_min(3:end);
    
%     a_min_own = x_min_own(1);
%     b_min_own = x_min_own(2);
%     C_min_own = x_min_own(3:end);
    
    a_min_simp = x_min_simp(1);
    b_min_simp = x_min_simp(2);
    C_min_simp = x_min_simp(3:end);
    
    close all;
    
    a_vector(i) = a_min;
    b_vector(i) = b_min;
    c_matrix(:, i) = C_min;
    
%     a_vector_own(i) = a_min_own;
%     b_vector_own(i) = b_min_own;
%     c_matrix_own(:, i) = C_min_own;
    
    a_vector_simp(i) = a_min_simp;
    b_vector_simp(i) = b_min_simp;
    c_matrix_simp(:, i) = C_min_simp;
%     % Compare to matlabs steepest descent
%     options = optimoptions('fminunc','SpecifyObjectiveGradient',true);
%     REFERENCE = fminunc(func, x0, options);
end

% Calculate means for parameters
a_mean = mean(a_vector);
b_mean = mean(b_vector);
c1_mean = mean(c_matrix(:, 1));
c2_mean = mean(c_matrix(:, 2));

% Calculate means for parameters
a_mean_simp = mean(a_vector_simp);
b_mean_simp = mean(b_vector_simp);
c1_mean_simp = mean(c_matrix_simp(1, :));
c2_mean_simp = mean(c_matrix_simp(2, :));

% Plot means
means = [a_mean, a_mean_simp; b_mean, b_mean_simp; c1_mean, c1_mean_simp; c2_mean, c2_mean_simp];
labels={'a'; 'b'; 'c1'; 'c2'};
bar(means)
set(gca,'xticklabel',labels)
legend('matlab', 'simplified')
title('Means for 100 parameter estimations for a, b, c1 and c2')

% Plot parameters
figure

subplot(2,2,1)
histogram(a_vector_simp);
title('Histrogram of 100 estimates of a using simplified simplex')
xlabel('Value of a')
ylabel('Frequency')

subplot(2,2,2)
histogram(b_vector_simp);
title('Histrogram of 100 estimates of b using simplified simplex')
xlabel('Value of b')
ylabel('Frequency')

subplot(2,2,3)
histogram(c_matrix_simp(1, :));
title('Histrogram of 100 estimates of c1 using simplified simplex')
xlabel('Value of c1')
ylabel('Frequency')

subplot(2,2,4)
histogram(c_matrix_simp(2, :));
title('Histrogram of 100 estimates of c2 using simplified simplex')
xlabel('Value of c2')
ylabel('Frequency')

figure

subplot(2,2,1)
histogram(a_vector);
title('Histrogram of 100 estimates of a using matlabs simplex')
xlabel('Value of a')
ylabel('Frequency')

subplot(2,2,2)
histogram(b_vector);
title('Histrogram of 100 estimates of b using matlabs simplex')
xlabel('Value of b')
ylabel('Frequency')

subplot(2,2,3)
histogram(c_matrix(1, :));
title('Histrogram of 100 estimates of c1 using matlabs simplex')
xlabel('Value of c1')
ylabel('Frequency')

subplot(2,2,4)
histogram(c_matrix(2, :));
title('Histrogram of 100 estimates of c2 using matlabs simplex')
xlabel('Value of c2')
ylabel('Frequency')
    
%% Task C b)
    
% Generate big data set for testing performance
N_big = 10e4;
X_big = normrnd(0, 2, n_dim, N_big);
Y_big = zeros(N_big, 1);
for i=1:N_big
    
    Y_big(i) = func1(X_big(1:2, i));
    
end

% "True performance" for matlab simplex
RMSE_vector = zeros(N_sets, 1);

for i=1:N_sets
    a = a_vector(i);
    b = b_vector(i);
    C = c_matrix(:, i);
    theta = [a;b;C];
    
    RMSE = sqrt(J_for_f(X_big, Y_big, theta));
    RMSE_vector(i) = RMSE;
end

% "True performance" for simplified simplex
RMSE_vector_simp = zeros(N_sets, 1);

for i=1:N_sets
    
    a = a_vector_simp(i);
    b = b_vector_simp(i);
    C = c_matrix_simp(:, i);
    theta = [a;b;C];
    
    RMSE = sqrt(J_for_f(X_big, Y_big, theta));
    RMSE_vector_simp(i) = RMSE;
end

figure

subplot(2,1,1)
histogram(RMSE_vector);
title('Histrogram of performance for 100 models optimised with matlabs simplex')
xlabel('RMSE')
ylabel('Frequency')

subplot(2,1,2)
histogram(RMSE_vector_simp);
title('Histrogram of performance for 100 models optimised with simplified simplex')
xlabel('RMSE')
ylabel('Frequency')

%% Task D

% Repeat C with added noise

for j=1:N_sets
    D_train = D{j};
    Y_train = D_train(:, end);
    for i=1:N_train
        
        % Original Y
        Y = Y_train(i);
        
        % Noise
        epsilon = normrnd(0, Y/10);
        
        % Add noise
        Y_train(i) = Y + epsilon;

    end
    
 
   D{j} = [X_train' Y_train]; 
    
end

% Collect model parameter estimates
a_vector = zeros(1, 100);
b_vector = zeros(1, 100);
c_matrix = zeros(n_dim, 100);

% a_vector_own = zeros(1, 100);
% b_vector_own = zeros(1, 100);
% c_matrix_own = zeros(n_dim-2, 100);

a_vector_simp = zeros(1, 100);
b_vector_simp = zeros(1, 100);
c_matrix_simp = zeros(n_dim, 100);

for i=1:N_sets
    
    D_train = D{i};
    X_train = D_train(:, 1:end-1)';
    Y_train = D_train(:, end);
    
    func = @(theta)J_for_f(X_train, Y_train, theta);
    func_gradient = @(theta)J_for_f_gradient(X_train, Y_train, theta);

    % Set parameters for steepest descent
    N = 10000;
    beta = 0.05;

    % Set starting guess for parameters
    X_size = size(X_train);
    d = X_size(1) + 2;
    x0 = ones(d, 1);

    % Run steepest descent
    %[x_min_own, J_min_own] = steepest_descent(x0, func, func_gradient, N, beta);
    %options = optimoptions('fminunc','SpecifyObjectiveGradient',true);
    %x_min = fminunc(func, x0, options);
    x_min = fminsearch(func, x0);
    
    [x_min_simp, J_min_simp] = simplex(x0, func, N, beta);
    
    %figure
    close all
    

    % Extract "optimised" parameters
    a_min = x_min(1);
    b_min = x_min(2);
    C_min = x_min(3:end);
    
%     a_min_own = x_min_own(1);
%     b_min_own = x_min_own(2);
%     C_min_own = x_min_own(3:end);
    
    a_min_simp = x_min_simp(1);
    b_min_simp = x_min_simp(2);
    C_min_simp = x_min_simp(3:end);
    
    close all;
    
    a_vector(i) = a_min;
    b_vector(i) = b_min;
    c_matrix(:, i) = C_min;
    
%     a_vector_own(i) = a_min_own;
%     b_vector_own(i) = b_min_own;
%     c_matrix_own(:, i) = C_min_own;
    
    a_vector_simp(i) = a_min_simp;
    b_vector_simp(i) = b_min_simp;
    c_matrix_simp(:, i) = C_min_simp;
%     % Compare to matlabs steepest descent
%     options = optimoptions('fminunc','SpecifyObjectiveGradient',true);
%     REFERENCE = fminunc(func, x0, options);
end

% Calculate means for parameters
a_mean = mean(a_vector);
b_mean = mean(b_vector);
c1_mean = mean(c_matrix(:, 1));
c2_mean = mean(c_matrix(:, 2));

% Calculate means for parameters
a_mean_simp = mean(a_vector_simp);
b_mean_simp = mean(b_vector_simp);
c1_mean_simp = mean(c_matrix_simp(1, :));
c2_mean_simp = mean(c_matrix_simp(2, :));

% Plot means
means = [a_mean, a_mean_simp; b_mean, b_mean_simp; c1_mean, c1_mean_simp; c2_mean, c2_mean_simp];
labels={'a'; 'b'; 'c1'; 'c2'};
bar(means)
set(gca,'xticklabel',labels)
legend('matlab', 'simplified')
title('Means for 100 parameter estimations for a, b, c1 and c2')

% Plot parameters
figure

subplot(2,2,1)
histogram(a_vector_simp);
title('Histrogram of 100 estimates of a using simplified simplex')
xlabel('Value of a')
ylabel('Frequency')

subplot(2,2,2)
histogram(b_vector_simp);
title('Histrogram of 100 estimates of b using simplified simplex')
xlabel('Value of b')
ylabel('Frequency')

subplot(2,2,3)
histogram(c_matrix_simp(1, :));
title('Histrogram of 100 estimates of c1 using simplified simplex')
xlabel('Value of c1')
ylabel('Frequency')

subplot(2,2,4)
histogram(c_matrix_simp(2, :));
title('Histrogram of 100 estimates of c2 using simplified simplex')
xlabel('Value of c2')
ylabel('Frequency')

figure

subplot(2,2,1)
histogram(a_vector);
title('Histrogram of 100 estimates of a using matlabs simplex')
xlabel('Value of a')
ylabel('Frequency')

subplot(2,2,2)
histogram(b_vector);
title('Histrogram of 100 estimates of b using matlabs simplex')
xlabel('Value of b')
ylabel('Frequency')

subplot(2,2,3)
histogram(c_matrix(1, :));
title('Histrogram of 100 estimates of c1 using matlabs simplex')
xlabel('Value of c1')
ylabel('Frequency')

subplot(2,2,4)
histogram(c_matrix(2, :));
title('Histrogram of 100 estimates of c2 using matlabs simplex')
xlabel('Value of c2')
ylabel('Frequency')
    
%% Task D b)
    
% Generate big data set for testing performance, add noise to prevois set
for i=1:N_big
    
    Y = Y_big(i);
    
    epsilon = normrnd(0, Y/10);
    
    Y_big(i) = Y +  epsilon;
    
end

% "True performance" for matlab simplex
RMSE_vector = zeros(N_sets, 1);

for i=1:N_sets
    a = a_vector(i);
    b = b_vector(i);
    C = c_matrix(:, i);
    theta = [a;b;C];
    
    RMSE = sqrt(J_for_f(X_big, Y_big, theta));
    RMSE_vector(i) = RMSE;
end

% "True performance" for simplified simplex
RMSE_vector_simp = zeros(N_sets, 1);

for i=1:N_sets
    
    a = a_vector_simp(i);
    b = b_vector_simp(i);
    C = c_matrix_simp(:, i);
    theta = [a;b;C];
    
    RMSE = sqrt(J_for_f(X_big, Y_big, theta));
    RMSE_vector_simp(i) = RMSE;
end

figure

subplot(2,1,1)
histogram(RMSE_vector);
title('Histogram of performance for 100 models optimised with matlabs simplex')
xlabel('RMSE')
ylabel('Frequency')

subplot(2,1,2)
histogram(RMSE_vector_simp);
title('Histogram of performance for 100 models optimised with simplified simplex')
xlabel('RMSE')
ylabel('Frequency')

%% D c)
N_cv = 400;
n_dim = 10;

X_cv = normrnd(0,2,n_dim,N_cv);

Y_cv = zeros(100, 1);

for i=1:N_cv
    
    Y = f(X_cv(1:2, i), a, b, [c1;c2]);
    
    epsilon = normrnd(0, Y/10);
    
    Y_cv(i) = Y + epsilon;
end

RMSE_vector_simp = zeros(100, 1);

RMSE_vector = zeros(100, 1);

counter = 1;
for i=1:50
    
    cv_partitions = crossvalind('kfold', Y_cv, 2);
    
    DX = cell(1,2);
    DY = cell(1,2);
    
    cv_X1 = X_cv(:, cv_partitions == 1);
    cv_X2 = X_cv(:, cv_partitions == 2);
    
    DX{1} = cv_X1;
    DX{2} = cv_X2;
    
    cv_Y1 = Y_cv(cv_partitions == 1);
    cv_Y2 = Y_cv(cv_partitions == 2);
    
    DY{1} = cv_Y1;
    DY{2} = cv_Y2;
    
    for k=1:2
        X_train = DX{k};
        Y_train = DY{k};
        
        X_test = DX{3 - k};
        Y_test = DY{3 - k};
        
        
        d = n_dim + 2;
        % Set starting guess for parameters
        x0 = ones(d, 1);


        func = @(theta)J_for_f(X_train, Y_train, theta);


        x_min = fminsearch(func, x0);

        [x_min_simp, J_min_simp] = simplex(x0, func, N, beta);

        close all

        % Extract "optimised" parameters
        a_min = x_min(1);
        b_min = x_min(2);
        C_min = x_min(3:end);

        a_min_simp = x_min_simp(1);
        b_min_simp = x_min_simp(2);
        C_min_simp = x_min_simp(3:end);

        close all;
        

         % RMSE for simplified simplex
         a = a_min_simp;
         b = b_min_simp;
         C = C_min_simp;
         theta = [a;b;C];

         RMSE = sqrt(J_for_f(X_test, Y_test, theta));
         RMSE_vector_simp(counter) = RMSE;
         
         
         % For matlab simplex
         a = a_min;
         b = b_min;
         C = C_min;
         theta = [a;b;C];

         RMSE = sqrt(J_for_f(X_test, Y_test, theta));
         RMSE_vector(counter) = RMSE;
         
         counter = counter + 1;
    end
       
end

figure

subplot(2,1,1)
histogram(RMSE_vector);
title('Histrogram of performance for 100 models optimised with matlabs simplex using CV')
xlabel('RMSE')
ylabel('Frequency')

subplot(2,1,2)
histogram(RMSE_vector_simp);
title('Histrogram of performance for 100 models optimised with simplified simplex using CV')
xlabel('RMSE')
ylabel('Frequency')

