function [x_min, J_min] = steepest_descent(x0, func, func_gradient, N, beta)

    % Calculate gradient
    g = func_gradient(x0);
    
    % Start guess for alfa
    alfa0 = 0.01*norm(x0)/norm(g);
    
    % Counter
    t = 0;
    
    % Threshold
    thresh = 10^-5;
    
    % Store t, J and alfa values in vectors
    t_vector = 0:1:N-1;
    J_vector = NaN(1, N);
    alfa_vector = NaN(1, N);
    
    while norm(g) > thresh

        % 1. Calculate J0 for x0
        J0 = func(x0);

        % 2. Calculate the gradient in the point x0
        g = func_gradient(x0);
        disp(g)
        % 3. Calculate x1
        x1 = x0 - alfa0*g;

        % 4. Calculate the new function value J1
        J1 = func(x1);

        % 5a. If J1 is less than J0, proceed with larger step
        if J1 < J0
            alfa1 = (1 + beta) * alfa0;
        end

        % 5b. If the new value is larger, reduce step size
        while J1 > J0

            % Reduce alfa
            alfa1 = (1 - beta) * alfa0;

            % Calculate new x1 based on shorter step
            x1 = x0 - alfa0*g;

            % Calculate new J1
            J1 = func(x1);

        end
        
        % Store values
        alfa_vector(t+1) = alfa0;
        J_vector(t+1) = J0;
        
        % Update counter
        t = t+1;
        
        
        % 6. Update indicies
        alfa0 = alfa1;
        x0 = x1;
        

        
        % Set output to current x and J
        x_min = x1;
        J_min = J1;
        
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
        
        % Terminate if max iterations have been reached
        if t > N
            disp("Maximum evaluations reached")
            break
        end  
    end
end