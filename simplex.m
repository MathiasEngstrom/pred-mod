function [x_min, J_min] = simplex(x0, func, N, beta)
%SIMPLEX Find minima of function using a simplified simplex algorithm
%   Detailed explanation goes here

    % 1. Guess alfa paramter
    alfa = 2;

    % 2. Create a simplex dataset S
    % Get size of vector x and determine dimensions d, n simplex points is d+1
    x0_dimensions = size(x0);
    d = x0_dimensions(1);
    n_points = d + 1;

    S = zeros(d, n_points);
    S(:, 1) = x0;

    % 3.1 Produce random points to start the algorithm, make sure no row is equal
    % to avoid getting all points on a line
    while ~all(diff(S'))

        for i = 2:n_points
            S(:,i) = x0.*rand(d, 1) - 10*rand();
        end
    end
    
    % 3.2 Calculate J for the starting points
    J_for_S = zeros(1, n_points);
    for i = 1:n_points
        point = S(:,i);
        J_for_S(i) = func(point);
    end
    
    % 3.4 Find the maximum value in J_for_S
    JM = max(J_for_S);
    
    % 3.5 Find the index of the point in S that corresponds to the maximum
    max_index = find(J_for_S == max(J_for_S));
    
    % 3.6 Get the point xM
    xM = S(:, max_index);
    
    % 4. Calculate centroid between the points that were not worst
    S_M = S(:, [1:max_index-1 max_index+1:end]);
    c = mean(S_M, 2);
    
    % Iterate over t
    t = 1;
    
    % Count number of function evaluations n
    n = 0;
    
    % Store alfa and J values in vectors, use NaN to exclude non set values
    % from the plot
    J_vector = NaN(1, N);
    alfa_vector = NaN(1, N);
    
    % Store the initial values calculated outside the loop
    J_vector(t) = JM;
    alfa_vector(t) = alfa;
    
    % Predefine iteration vector for plotting
    t_vector = 0:1:N-1;
    
    % Arbitraly set convergance criteria to 10 to enter while loop
    conv_crit = 10;
    thresh = 10e-10;
    x0 = xM;
    
    % Continue until converged
    while conv_crit > thresh
        
        % 5.1 Calculate the new point
        x1 = c + alfa*(c - xM);

        % 5.2 Calculate J for the new point
        J1 = func(x1);

        % 5.2.1 We use JM as reference
        J0 = JM;

        % 6a The point gives a larger J, reduce step size
        if J1 > J0
            
            % Reduce step size by reducing alfa, repeat step 4 and 5
            alfa = (1 - beta)*alfa;

        % 6b. The point gives a smaller J, set new point, increase step
        % size and proceed
        elseif J1 < J0

            % Replace xM with the new point x1
            S(:, max_index) = x1;
            
            % Store J
            J_vector(t + 1) = J1;
            
            % Replace the corresponding value for x1 in J_for_S
            J_for_S(max_index) = J1;
            
            % Store alfa
            alfa_vector(t + 1) = alfa;
            
            % Increase alfa
            alfa = (1 + beta)*alfa;

            % Increase iteration index t
            t = t + 1;
            
            %----- Repeat steps 3.4 to 4 with the updated set S -------%
            
            % 3.4 Find the maximum value in J_for_S
            JM = max(J_for_S);

            % 3.5 Find the index of the point in S that corresponds to 
            % the maximum
            max_index = find(J_for_S == JM);

            % 3.6 Get the point xM
            xM = S(:, max_index);

            % 4. Calculate centroid between the points that were not worst
            S_M = S(:, [1:max_index-1 max_index+1:end]);
            c = mean(S_M, 2);
            
            % Check if we have converged
            conv_crit = norm(x1-x0)/norm(x0);
            x0 = x1;
            
        end
        
        % Increase evaluation counter n
        n = n + 1;

        % Terminate if max iterations reached
        if n >= N
            disp('Reached max iterations')
            break
        end
        
    end
    
    subplot(2,1,1)
    plot(t_vector, alfa_vector);
    title('Alfa over number of steps t')
    xlabel('t')
    ylabel('Alfa')
    
    subplot(2,1,2)
    plot(t_vector, J_vector);
    title('J(x) over time')
    xlabel('t')
    ylabel('J(x)')
    
    % Set output
    x_min = x1; 
    J_min = J1;     
    n
    
end