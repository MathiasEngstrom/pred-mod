function [J] = J3(X, Y, theta)
%J_FOR_F Error function for f, takes matrix X of training data and vector Y
% of corresponding y values
%   Detailed explanation goes here
n_parameters = length(theta);
n_b = (n_parameters - 1)/2;


a = theta(1);

b = theta(2:n_b+1);

c = theta(n_b+2:end);

N = length(Y);

error = 0;
 

for i=1:N
    
    x = X(:,i);
    y_hat = f3(x, a, b, c);
    
    
    y = Y(i);
    
    error = error + (y - y_hat)^2;
    
end

J = (1/N)*error;

end

