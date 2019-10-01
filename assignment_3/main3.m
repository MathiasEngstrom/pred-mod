%% Task A and B
clc
clear all
close all

% Set how many dimensions observations should have, 
% n_dim = 2 will correspond to the underlying model
n_dim = 10;

% Define paramaters
a = 0.5;
b = [2; 2];
c = [2;3];

% Set how many train examples should be generated
N_train = 400;

% Generate 10 dimensional vectors of observations
X_train = unifrnd(0, 6, [n_dim, N_train]);

% Generate the response, only dependent on x1 and x2
Y_train = zeros(N_train, 1);
response_func = @(x)f3(x, a, b, c);

for i=1:N_train
    
    Y_train(i) = response_func(X_train(1:2, i));
    
end

X_train = X_train(1:n_dim,:);

func = @(theta)J3(X_train, Y_train, theta);
func_gradient = @(theta)J3_gradient(X_train, Y_train, theta);

% Set parameters for steepest descent
N = 50000;
beta = 0.05;

% Set starting guess for parameters
x0 = ones(n_dim*2+1,1);
x0(1) = 0.25;
x0(2) = 1;
x0(3) = 1;
x0(4) = 4;
x0(5) = 4;

% Run steepest descent
[x_min, J_min] = steepest_descent(x0, func, func_gradient, N, beta, true);

%% Task C
clc
clear all
close all

% Define paramaters
a = 0.5;
b = [2; 2];
c = [2;3];
n_dim = 10;
% Set how many train examples should be generated
N_train = 400;

% Set number of sets
N_sets = 100;

D = cell(1, 100);

for j=1:N_sets
    % Generate 10 dimensional vectors of observations
    X_train = unifrnd(0, 6, [n_dim, N_train]);
    X_train = X_train(1:n_dim, :);
    % Generate the response, only dependent on x1 and x2
    Y_train = zeros(N_train, 1);
    

    for i=1:N_train

        Y_train(i) = f3(X_train(1:2, i), a, b, c);

    end
    
   D{j} = [X_train' Y_train]; 
    
end

% Collect model parameter estimates
a_vector = zeros(1, 100);
b_matrix = zeros(n_dim, 100);
c_matrix = zeros(n_dim, 100);

for i=1:N_sets
    
    D_train = D{i};
    X_train = D_train(:, 1:end-1)';
    Y_train = D_train(:, end);
    
    func = @(theta)J3(X_train, Y_train, theta);
    func_gradient = @(theta)J3_gradient(X_train, Y_train, theta);

    % Set parameters for steepest descent
    N = 50000;
    beta = 0.05;

    % Set starting guess for parameters
    X_size = size(X_train);

    x0 = ones(n_dim*2+1, 1);
    x0(1) = 0.25;
    x0(2) = 1;
    x0(3) = 1;
    x0(4) = 4;
    x0(5) = 4;
    % Run steepest descent
    [x_min, J_min] = steepest_descent(x0, func, func_gradient, N, beta, false);
    
    % Extract "optimised" parameters
    a_min = x_min(1);
    b_min = x_min(2:n_dim+1);
    C_min = x_min(n_dim+2:end);
    
    %close all;
    
    a_vector(i) = a_min;
    b_matrix(:, i) = b_min;
    c_matrix(:, i) = C_min;
end

% Calculate means for parameters
a_mean = mean(a_vector);
b1_mean = mean(b_matrix(:, 1));
b2_mean = mean(b_matrix(:, 2));
c1_mean = mean(c_matrix(:, 1));
c2_mean = mean(c_matrix(:, 2));

% Plot means
means = [a_mean, 0.5; b1_mean, 2; b2_mean, 2; c1_mean, 2; c2_mean, 3];
labels={'a'; 'b1'; 'b2'; 'c1'; 'c2'};
bar(means)
set(gca,'xticklabel',labels)
legend('estimated', 'actual')
title('Means for 100 parameter estimations for a, b1, b2, c1 and c2')

% Plot parameters
figure

subplot(3,2,1)
histogram(a_vector);
title('Histrogram of 100 estimates of a using steepest descent')
xlabel('Value of a')
ylabel('Frequency')

subplot(3,2,2)
histogram(b_matrix(1, :));
title('Histrogram of 100 estimates of b1 using steepest descent')
xlabel('Value of b1')
ylabel('Frequency')

subplot(3,2,3)
histogram(b_matrix(2, :));
title('Histrogram of 100 estimates of b2 using steepest descent')
xlabel('Value of b2')
ylabel('Frequency')

subplot(3,2,4)
histogram(c_matrix(1, :));
title('Histrogram of 100 estimates of c1 using steepest descent')
xlabel('Value of c1')
ylabel('Frequency')

subplot(3,2,5)
histogram(c_matrix(2, :));
title('Histrogram of 100 estimates of c2 using steepest descent')
xlabel('Value of c2')
ylabel('Frequency')


%% Task C b)
a = 0.5;
b = [2; 2];
c = [2;3];   
% Generate big data set for testing performance
N_big = 10e4;
X_big = unifrnd(0, 6, [n_dim, N_big]);
Y_big = zeros(N_big, 1);
for i=1:N_big
    
    Y_big(i) = f3(X_big(1:2, i), a, b, c);
    
end

% "True performance"
RMSE_vector = zeros(N_sets, 1);

for i=1:N_sets
    a = a_vector(i);
    b = b_matrix(:, i);
    c = c_matrix(:, i);
    theta = [a;b;c];
    
    RMSE = sqrt(J3(X_big, Y_big, theta));
    RMSE_vector(i) = RMSE;
end

figure

histogram(RMSE_vector);
title('Histrogram of performance for 100 models optimised with steepest descent')
xlabel('RMSE')
ylabel('Frequency')

%% Task D

% Repeat C with added noise, using the same data frame
D_noise = cell(1, 100);
for j=1:N_sets
    D_train_orig = D{j};
    
    Y_train_orig = D_train_orig(:, end);
    X_train_orig = D_train_orig(:, 1:end-1)';
    Y_train_noise = zeros(size(Y_train_orig));
    for i=1:N_train
        
        % Original Y
        Y = Y_train_orig(i);
        
        % Noise
        epsilon = normrnd(0, Y/10);
        
        % Add noise
        Y_train_noise(i) = Y + epsilon;

    end
    
 
   D_noise{j} = [X_train_orig' Y_train_noise]; 
    
end

% Collect model parameter estimates
a_vector = zeros(1, 100);
b_matrix = zeros(n_dim, 100);
c_matrix = zeros(n_dim, 100);



for i=1:N_sets
    
    D_train_noise = D_noise{i};
    X_train_noise = D_train_noise(:, 1:end-1)';
    Y_train_noise = D_train_noise(:, end);
    
    func = @(theta)J3(X_train_noise, Y_train_noise, theta);
    func_gradient = @(theta)J3_gradient(X_train_noise, Y_train_noise, theta);

    % Set parameters for steepest descent
    N = 3000;
    beta = 0.05;

    % Set starting guess for parameters
    x0 = ones(2*n_dim + 1 , 1);
    x0(1) = 0.25;
    x0(2) = 1;
    x0(3) = 1;
    x0(4) = 4;
    x0(5) = 4;
    
    % Run steepest descent
    [x_min, J_min] = steepest_descent(x0, func, func_gradient, N, beta, true);  
    
    
    % Extract "optimised" parameters
    a_min = x_min(1);
    b_min = x_min(2:n_dim+1);
    C_min = x_min(n_dim+2:end);
    
    a_vector(i) = a_min;
    b_matrix(:, i) = b_min;
    c_matrix(:, i) = C_min;
    
end

% Calculate means for parameters
a_mean = mean(a_vector);
b1_mean = mean(b_matrix(:, 1));
b2_mean = mean(b_matrix(:, 2));
c1_mean = mean(c_matrix(:, 1));
c2_mean = mean(c_matrix(:, 2));

figure

% Plot means
means = [a_mean, 0.5; b1_mean, 2; b2_mean, 2; c1_mean, 2; c2_mean, 3];
labels={'a'; 'b1'; 'b2'; 'c1'; 'c2'};
bar(means)
set(gca,'xticklabel',labels)
legend('steepest descent', 'actual')
title('Means for 100 parameter estimations for a, b1, b2, c1 and c2')

% Plot parameters
figure

subplot(3,2,1)
histogram(a_vector);
title('Histrogram of 100 estimates of a using steepest descent')
xlabel('Value of a')
ylabel('Frequency')

subplot(3,2,2)
histogram(b_matrix(1, :));
title('Histrogram of 100 estimates of b1 using steepest descent')
xlabel('Value of b1')
ylabel('Frequency')

subplot(3,2,3)
histogram(b_matrix(2, :));
title('Histrogram of 100 estimates of b2 using steepest descent')
xlabel('Value of b2')
ylabel('Frequency')

subplot(3,2,4)
histogram(c_matrix(1, :));
title('Histrogram of 100 estimates of c1 using steepest descent')
xlabel('Value of c1')
ylabel('Frequency')

subplot(3,2,5)
histogram(c_matrix(2, :));
title('Histrogram of 100 estimates of c2 using steepest descent')
xlabel('Value of c2')
ylabel('Frequency')
    
%% Task D b)
a = 0.5;
b = [2; 2];
c = [2;3];

% "True performance" 
RMSE_vector = zeros(N_sets, 1);

for i=1:N_sets
    a = a_vector(i);
    b = b_matrix(:, i);
    c = c_matrix(:, i);
    theta = [a;b;c];
    
    RMSE = sqrt(J3(X_big, Y_big, theta));
    RMSE_vector(i) = RMSE;
end

figure

histogram(RMSE_vector);
title('Histogram of performance for 100 models optimised with steepest descent')
xlabel('RMSE')
ylabel('Frequency')

%% D c) 
a = 0.5;
b = [2; 2];
c = [2;3];


N_cv = 400;
n_dim = 10;

X_cv = normrnd(0,2,n_dim,N_cv);

Y_cv = zeros(100, 1);

for i=1:N_cv
    
    Y = f3(X_cv(1:2, i), a, b, c);
    
    epsilon = normrnd(0, Y/10);
    
    Y_cv(i) = Y + epsilon;
end

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
        
        
        % Set starting guess for parameters
        x0 = ones(2*n_dim + 1 , 1);
        x0(1) = 0.25;
        x0(2) = 1;
        x0(3) = 1;
        x0(4) = 4;
        x0(5) = 4;


        func = @(theta)J3(X_train, Y_train, theta);
        func_gradient = @(theta)J3_gradient(X_train, Y_train, theta);

        x_min = steepest_descent(x0, func, func_gradient, N, beta, false);

        % Extract "optimised" parameters
        a_min = x_min(1);
        b_min = x_min(2:n_dim+1);
        C_min = x_min(n_dim+2:end);
         
        a = a_min;
        b = b_min;
        c = C_min;
        theta = [a;b;c];

        RMSE = sqrt(J3(X_test, Y_test, theta));
        RMSE_vector(counter) = RMSE;
         
        counter = counter + 1;
    end
       
end

figure

histogram(RMSE_vector);
title('Histogram of performance for 100 models optimised with steepest descent using CV')
xlabel('RMSE')
ylabel('Frequency')


