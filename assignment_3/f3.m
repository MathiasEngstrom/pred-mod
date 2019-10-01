function [y] = f3(X, a, b, c)
%F3 Response function
%   Takes a vector X
B = diag(b);
y = a*exp(-(X-c)'*B*(X-c));
end