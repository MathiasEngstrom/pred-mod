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

% Perform k-means on training set X_train with k = B
[cluster_labels, C, SUMD, distances] = kmeans(X_train', B);

% Create training set S
size_X = size(X_train);
n_examples = size_X(2);
d = size_X(1);

known_y_idx = zeros(1, t_max*B);

S_X = zeros(d, B);
S_Y = zeros(1, B);

for i=1:B
    
    cluster_X_index = find(cluster_labels == i);
    
    %cluster_X = X_train(:, cluster_labels == i);
    %cluster_Y = Y_train(cluster_labels == i);
    %cluster_size = size(cluster_X);
    
    select_index = find(distances(:, i) == min(distances(:, i)));
    select_index = select_index(1);
    
    S_X(:, i) = X_train(:, select_index);
    S_Y(i) = Y_train(select_index);
    
    
    known_y_idx(i) = select_index;
end

t = 1;
P = 20;

while t < t_max
    
    n_examples = length(S_Y);
    bootstrap_size = n_examples;
    %S_star = cell(n_examples);
    f_p = cell(n_examples);
    
    for p = 1:P
        
        bootstrap_idx = randsample(n_examples, bootstrap_size, true);
        Sp_star_X = S_X(:, bootstrap_idx);
        Sp_star_Y = S_Y(:, bootstrap_idx);
        %S_star{p} = Sp_star;
        f_p{p} = train_NN(Sp_star_X, Sp_star_Y);
        
    end
    
    k = B*(t+1);
    [cluster_labels, C, SUMD, distances] = kmeans(X_train', k);
    
    known_y = known_y_idx(known_y_idx ~= 0);
    known_labels = cluster_labels(known_y);
    
    clusters = unique(cluster_labels);
    clusters(known_labels) = [];
    
    n_clusters = length(clusters);
    cluster_sizes = zeros(n_clusters, 2);

    for i=1:n_clusters

        label = clusters(i);
        c_size = length(cluster_labels(cluster_labels  == label));
        cluster_sizes(i, :) = [label; c_size];
    
    end
    
    clusters_sorted = sortrows(cluster_sizes.',2,'desc').';
    X_t = zeros(t_max, 1);
    
    for b = 1:B
        
        label = clusters_sorted(b);
        index = cluster_labels(cluster_labels == label);
        S_b = X_train(:, index);
        size_Sb = size(S_b);
        
        y_hat_n = zeros(P, size_Sb(2));
        for p = 1:P
            
            model = f_p{p};
            
            for n = 1:size_Sb(2)
                x_n = S_b(:, n);
                y_hat = model(x_n);
                y_hat_n(p, n) = y_hat;
            end
            
            v_n = var(y_hat_n);
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