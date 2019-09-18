function [g] = J_gradient(x, m1, m2, A1, A2)
%J_GRADIENT Analytical gradient if function J
%   m1, m2 and x should be Nx1 column vectors; A1 and A2 NxN matrices
g = (A1 + A1')*(x - m1)*(x - m2)'*A2*(x - m2) + (A2 + A2')*(x - m2)*(x - m1)'*A1*(x-m1);
end

