function [x_min, J_min] = simplex(x0, func, N, beta)
%SIMPLEX Find minima of function using a simplified simplex algorithm
%   Detailed explanation goes here

    % Start guess for alfa
    alfa0 = 2;

    % Get size of vector x and determine dimensions d, n simplex points is d+1
    x0_dimensions = size(x0);
    d = x0_dimensions(1);
    n_points = d + 1;

    S = zeros(d, n_points);
    S(:, 1) = x0;

    % Produce random points to start the algorithm, make sure no row is equal
    % to avoid getting all points on a line
    while ~all(diff(S'))

        for i = 2:n_points
            S(:,i) = x0.*rand(d, 1);
        end
    end


    % Arbitraly set convergance criteia to 10 to enter while loop
    conv_criteria = 10;
    threshold = 10e-4;
    
    % Calculate J for the strating points
    J_for_S = zeros(1, n_points);
    for i = 1:n_points
        point = S(:,i);
        J_for_S(i) = func(point);
    end
    
    J0 = J_for_S(1);
    
    % Store t, J and alfa values in vectors
    t_vector = 0:1:N-1;
    J_vector = NaN(1, N);
    alfa_vector = NaN(1, N);
    
    t = 0;
    while conv_criteria > threshold
        % Find the index in S of the point with the largest value for J(x)
        max_index = find(J_for_S == max(J_for_S));
        %J_max = J_for_S(max_index);
        xM = S(max_index);
        
        % Calculate centroid between the points that were not worst
        S_without_max = S(:, [1:max_index-1 max_index+1:end]);
        centroid = sum(S_without_max, 2)/d;
        
        % The new point is calculated
        x1 = centroid + alfa0*(centroid - xM);
        
        % J(x1) is calculated
        J1 = func(x1);
        
        % Everything is fine, increase step length
        if J1 < J0
            alfa1 = (1 + beta)*alfa0;
            
            % Store values
            alfa_vector(t+1) = alfa0;
            J_vector(t+1) = J0;
            
            % Update indecies
            alfa0 = alfa1;
            t = t + 1;
            
            % Overwrite the old 'worst' point with the new
            S(:, max_index) = x1;
            
            % Overwrite the old 'worst' value for J with the new
            J_for_S(max_index) = J0;
        else
            % Decrease step length until J1 < J0
        
            % Decrease alfa
            alfa1 = (1 - beta)*alfa0;
            
            % Store values
            alfa_vector(t+1) = alfa0;
            J_vector(t+1) = J1;
            
            % Update indecies
            alfa0 = alfa1;
            %t = t + 1;
        end
        
        % Check if we have converged
        conv_criteria = norm(x1-x0)/norm(x0);
        
        x0 = x1;
        J0 = J1;
        
        % Plot results
        subplot(2,1,1)
        plot(t_vector, alfa_vector);
        title('Alfa over time')
        xlabel('Time')
        ylabel('Alfa')
        
        subplot(2,1,2)
        plot(t_vector, J_vector);
        title('J(x) over time')
        xlabel('Time')
        ylabel('J(x)')
        
        linkdata on
        
        % Terminate if max iterations reached
        if t > N
            disp('Reached max iterations')
            break
        end    
    end
    
    % Update output
    x_min = x1; 
    J_min = J1;
end