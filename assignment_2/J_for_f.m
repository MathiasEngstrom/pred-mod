function [J] = J_for_f(X, Y, theta)
%J_FOR_F Error function for f, takes matrix X of training data and vector Y
% of corresponding y values
%   Detailed explanation goes here
a = theta(1);

b = theta(2);

C = theta(3:end);

N = length(Y);

diff = zeros(N, 1);
 

for i=1:N
    
    x = X(:,i);
    y_hat = f(x, a, b, C);
    
    
    y = Y(i);
    
    diff(i) = (y - y_hat)^2;
    
end

J = (1/N)*sum(diff);

%g = J_for_f_gradient(X, Y, theta);
end

