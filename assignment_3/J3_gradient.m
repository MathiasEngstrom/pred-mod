function [g] = J3_gradient(X, Y, theta)
%J_for_f_gradient Calculates gradient for J_for_f
%   Detailed explanation goes here
% Parameter a should be in position 1 in theta etc..

n_parameters = length(theta);
n_b = (n_parameters - 1)/2;

a = theta(1);
b = theta(2:n_b+1);
% c is the rest of the parameters
c = theta(n_b+2:end);

% Sum over n to N
N = length(Y);

% Get dimensions of X
%X_size = size(X);
%d = X_size(1)*2 + 1;

% Save all (y - y_hat)*dy_hat/d_theta in diff and sum afterwards
%diff = zeros(d, N);
error = 0;
for i=1:N
    
    % Current x_n
    x = X(:, i);
    
    % Current y_hat_n
    y_hat = f3(x, a, b, c);
    
    % Current y_n
    y = Y(i);
    
    % Partial derivative dy_hat_n/da
    d_a = y_hat/a;
    
    % Partial derivative dy_hat_n/db
    d_b = -y_hat*(x - c).^2;
    
    % Partial derivative dy_hat_n/dc
    d_c = 2*b.*(x - c)*y_hat;
    
    % Gradient dy_hat_n/d_theta
    d_theta = [d_a; d_b; d_c];
    
    % Collect each value and sum afterwards
    error = error + (y - y_hat)*d_theta;
end

% Gradient dJ/d_theta
g = (-2/N)*error;


end

