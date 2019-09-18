function [g] = J_for_f_gradient(X, Y, theta)
%J_for_f_gradient Calculates gradient for J_for_f
%   Detailed explanation goes here
% Parameter a should be in position 1 in theta etc..
a = theta(1);
b = theta(2);
% C is the rest of the parameters
C = theta(3:end);

% Sum over n to N
N = length(Y);

% Get dimensions of X
X_size = size(X);
d = X_size(1) + 2;

% Save all (y - y_hat)*dy_hat/d_theta in diff and sum afterwards
diff = zeros(d, N);

for i=1:N
    
    % Current x_n
    x = X(:, i);
    
    % Current y_hat_n
    y_hat = f(x, a, b, C);
    
    % Current y_n
    y = Y(i);
    
    % Partial derivative dy_hat_n/da
    d_a = y_hat/a;
    
    % Partial derivative dy_hat_n/db
    d_b = -y_hat*norm(x - C)^2;
    
    % Partial derivative dy_hat_n/dC
    d_C = 2*b*(x - C)*y_hat;
    
    % Gradient dy_hat_n/d_theta
    d_theta = [d_a; d_b; d_C];
    
    % Collect each value and sum afterwards
    diff(:, i) = (y - y_hat)*d_theta;
end

% Gradient dJ/d_theta
g = (-2/N)*sum(diff, 2);


end

