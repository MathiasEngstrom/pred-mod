%% Task A
clc
clear all
close all

% Set how many dimensions observations should have, 
% n_dim = 2 will correspond to the underlying model
n_dim = 2;

% Define paramaters
a = 0.5;
b = [2; 2];
c = [2;3];

% Set how many train examples should be generated
N_train = 1000;
N_test = 10e4;
N_tot = N_test + N_train;

% Generate 10 dimensional vectors of observations U[0,6]xU[0,6]
X_tot = unifrnd(0, 6, [n_dim, N_tot]).*unifrnd(0, 6, [n_dim, N_tot]);

% Generate the response, only dependent on x1 and x2
Y_tot = zeros(N_tot, 1);
response_func = @(x)f3(x, a, b, c);

for i=1:N_train
    
    Y_tot(i) = response_func(X_tot(1:2, i));
    
end

X_train = X_tot(:, 1:N_train);
Y_train = Y_tot(1:N_train);

X_test = X_tot(:, N_train+1:end);
Y_test = Y_tot(N_train+1:end);


%%%%%%%%%%%%%%%%%%%%%%%%% Batch-mode active learning %%%%%%%%%%%%%%%%%%%%%%

% Number of batches B
B = 96;
t_max = 10;

% Create training set S

% ---------------------------- 1. ----------------------------------------%
% Perform k-means on training set X_train with k = B 

[cluster_labels, C, SUMD, distances] = kmeans(X_train', B);


size_X = size(X_train); 
n_examples = size_X(2); % Number of x-vectors
d = size_X(1);          % Dimension of x

% Vector to remember what experiments we've done (known y-values)
known_y_idx = zeros(1, t_max*B); 

% Dataset S(performed experiments), input S_X and response S_Y
S_X = zeros(d, B);
S_Y = zeros(1, B);

%---------------------------------- 2. -----------------------------------%
% For each cluster, choose one point, either
% (i) point closest to cluster centroid
% (ii) max Y of cluster

for i=1:B
    
    cluster_index = find(cluster_labels == i);
    cluster_X = X_train(:, cluster_index);
    cluster_Y = Y_train(cluster_index);
    
    % (i) point closest to cluster centroid
    %select_index = find(distances(:, i) == min(distances(:, i)));
    %select_index = select_index(1);
    
    %X_train(:, select_index);
    %Y_train(select_index);
    
    % (ii) max Y of cluster
    select_index = find(cluster_Y == max(cluster_Y));
    select_index = select_index(1);
    
    S_X(:, i) = cluster_X(:, select_index);
    S_Y(i) = cluster_Y(select_index);

    % Points in S correspond to 'known' points(experiments performed)
    % and should not be added again
    known_y_idx(i) = select_index;
end

% % TEST
% all_index = maxk(Y_train, B);
% for i=1:B
%     select_index = find(Y_train == all_index(i));
%     known_y_idx = select_index;
%     S_X(:, i) = X_train(:, select_index);
%     S_Y(i) = Y_train(select_index);
% end
% %END TEST

% -------------------------------- 3. ----------------------------------- %
%Let t = 1

t = 1;
P = 20; % Number of commitee members

% Plotting
rmse_values = zeros(t_max-1, 1);
function_plot = figure; % function plot handle
train_scatter = figure; % train dataset plot handle
% End plotting

% ------------------------------- 4. ------------------------------------ %
% The algorithm

while t < t_max
    
    % Plot the train datasets for each t
    figure(train_scatter);
    subplot(3, 3, t);
    scatter3(S_X(1,:), S_X(2,:), S_Y);
    title("t=" + t)
    xlabel('x1');
    ylabel('x2');
    % End plotting

    % --------------------- a. + b. ------------------------- %
    % a. Create P different bootstrap datasets Sp_star, p=1, 2 ... P
    % b. Use the sets to create P different models f_p(x)
    
    n_examples = length(S_Y); % Current number of examples in S
    bootstrap_size = n_examples; % Create bootstrap sets of same size as S
    f_p = cell(P, 1); % Save P = 20 models f_p(x) in a cell
    
    % Create P bootrsp sets from S, and train ANN:s for each
    for p = 1:P
        
        % Sample indecies from S with replacement = true
        bootstrap_idx = randsample(n_examples, bootstrap_size, true);
        
        % Select specified data points 
        Sp_star_X = S_X(:, bootstrap_idx);
        Sp_star_Y = S_Y(bootstrap_idx);

        f_p{p} = train_NN(Sp_star_X, Sp_star_Y);
        
    end
    
    % Plot the mean function of all commitee members f(x) = mean(f_p(x))
    % on the interval [0,6]
    x1=0:0.5:6;
    x2=x1;
    Y = zeros(numel(x1), numel(x2));
    
    % For i rows and j columns in the matrix Y
    for i = 1:numel(x1)
        for j = 1:numel(x2)
            
            f_x = zeros(P, 1); % Store each of P = 20 models output
            
            % Iterate over all models 
            for p = 1:P
                model = f_p{p};                 % select model p
                f_x(p) = model([x1(i); x2(j)]); % use model p to estimate y
            end
            
            f_x_mean = mean(f_x);

            Y(i, j) = f_x_mean;
        end
    end
    
    figure(function_plot);
    subplot(3, 3, t);
    mesh(x1, x2, Y)
    xlabel('x1')
    ylabel('x2')
    colorbar
    title('f(x) = mean(f_p(x))')
    % End plotting
    
    % Plot RMSE based on external test data in X_test and Y_test
    f_x_rmse = zeros(P, length(Y_test));
    
    % Iterate over all models
    for p = 1:P
        model = f_p{p};
        f_x_rmse(p, :) = model(X_test); % Input all x values at once
    end
    f_x_rmse_mean = mean(f_x_rmse);
    error = 0;
    for i=1:length(Y_test)
        error = error + (Y_test(i) - f_x_rmse_mean(i))^2;
    end
    rmse = sqrt(error)/length(Y_test);
    rmse_values(t) = rmse;
    % end RMSE and plotting
    
    % ---------------------------- c. -------------------------- %
    % Perform k-means clustering of the vectors xn in DTR using 
    % k = B*(t + 1)
    
    k = B*(t+1);
    [cluster_labels, C, SUMD, distances] = kmeans(X_train', k);
    
    % Theese y values are known and should not be included
    known_y = known_y_idx(known_y_idx ~= 0);
    known_labels = unique(cluster_labels(known_y)); % Clusters which have known y:s
    
    % Get all cluster labels, and remove those that have known y:s in them
    clusters = unique(cluster_labels);
    clusters(known_labels) = [];
    
    n_clusters = length(clusters);
    cluster_sizes = zeros(2, n_clusters);

    % Iterate over clusters, calculating their sizes
    for i=1:n_clusters

        label = clusters(i);
        c_size = length(cluster_labels(cluster_labels  == label));
        cluster_sizes(:, i) = [label; c_size];
    
    end
    
    clusters_sorted = sortrows(cluster_sizes.',2,'desc').';
    
    % ------------------ e. -----------------------%
    % Initialise empty index set to collect B new points to add to S based
    % on their variance in y_hat_p
    
    X_t = zeros(B, 1);
    
    % ------------------ f. ------------------------%
    % Iterate over B largest clusters
    for b = 1:B
        
        label = clusters_sorted(1, b); % Get the B biggest clusters
        index = find(cluster_labels == label); % Index in original X_train
        S_b = X_train(:, index); 
        size_Sb = size(S_b);
        
        
        y_hat_n = zeros(P, size_Sb(2)); % Collect P predictions for n x
        
        % Iterate over models
        for p = 1:P
            
            model = f_p{p};
            
            %Iterate over examples x_n in cluster b
            for n = 1:size_Sb(2)
                x_n = S_b(:, n);
                y_hat = model(x_n);
                y_hat_n(p, n) = y_hat;
            end
            
            v_n = var(y_hat_n); % Calculate variance for each column n
            max_v_idx = find(v_n == max(v_n));
            X_t(b) = index(max_v_idx(1));
            
        end
        
    end
    
    for b=1:B
        x_idx = X_t(b);
        S_X = [S_X, X_train(:, x_idx)];
        S_Y = [S_Y, Y_train(x_idx)];
    end
    
    t = t +1;
    
end

% Plotting
figure
plot(1:t_max-1, rmse_values');
title('RMSE over t')
xlabel('t');
ylabel('RMSE');