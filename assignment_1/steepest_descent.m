function [x_min, J_min] = steepest_descent(x0, func, func_gradient, N, beta)
        
    % Counters
    t = 0;
    n = 0;
    
    % Initiate x1 and J1
    x1 = NaN;
    J1 = NaN;
    
    % Threshold
    thresh = 10e-10;
    
    % Store t, J and alfa values in vectors
    t_vector = 0:1:N-1;
    J_vector = NaN(1, N);
    alfa_vector = NaN(1, N); 
    
   % 1. Calculate J0 for x0
   J0 = func(x0);

   % 2. Calculate the gradient in the point x0
   g0 = func_gradient(x0);
   %g0 = g/norm(g);
   thresh = thresh;%/norm(g);
   
   
   % Start guess for alfa
   alfa0 = 0.01*norm(x0)/norm(g0);
    
    while norm(g0) > thresh
        
        % 3. Calculate x1
        x1 = x0 - alfa0*g0;

        % 4. Calculate the new function value J1
        J1 = func(x1);

        % 5a. If J1 is less than J0, proceed with larger step
        if J1 < J0
            
            alfa1 = (1 + beta) * alfa0;
            
            % Store values
            alfa_vector(t+1) = alfa0;
            J_vector(t+1) = J0;
            
            % Update alfa
            alfa0 = alfa1;
            
            % Update gradient
            g1 = func_gradient(x1);
            %g1 = g/norm(g);
            
            % Update indecies
            x0 = x1;
            t = t + 1;
            J0 = J1;
            g0 = g1;
        else
            
            % 5b. If the new value is larger, reduce step size
            
            % Reduce alfa
            alfa1 = (1 - beta) * alfa0;

            % Update alfa
            alfa0 = alfa1;
            
        end
       
        % Terminate if max iterations have been reached
        if n >= N
            disp("Maximum evaluations reached")
            break
        end
        
        
        n = n + 1;
  
    end
    
    subplot(2,1,1)
    plot(t_vector, alfa_vector);
    title('Alfa number of steps t')
    xlabel('t')
    ylabel('Alfa')
    
    subplot(2,1,2)
    plot(t_vector, J_vector);
    title('J(x) number of steps t')
    xlabel('t')
    ylabel('J(x)')
    
    % Display info
    %alfa0
    %n
    %t
    %norm(g0)
    
    % Set output to current x and J
    x_min = x1;
    J_min = J1;
end