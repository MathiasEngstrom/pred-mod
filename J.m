function [J] = J(x, m1, m2, A1, A2)
%J Some form of error function to be minimised
%   m1, m2 and x should be Nx1 column vectors; A1 and A2 NxN matrices
J = (x - m1)' * A1 * (x - m1) * (x - m2)' * A2 * (x - m2);
end

