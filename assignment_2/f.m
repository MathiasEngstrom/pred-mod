function [y] = f(X, a, b, C)
%F Respone function
%   Takes a vector X
y = a*exp(-b*norm(X-C)^2);

end

