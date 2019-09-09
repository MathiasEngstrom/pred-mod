close all
clear all
clc

% Define paramaters
m1 = [-10, -10]';
m2 = [10, 10]';
A1 = [1, 0.5; 0.5, 1];
A2 = [1, -0.5; -0.5, 1];

x1 = -20:1:20;
x2 = x1';

f = zeros(size(length(x1)));

for i=1:length(x1)
    for j=1:length(x1)
        x = [x1(i);x2(j)];
       
        f(i, j) = J(x, m1, m2, A1, A2);
       
   
    end
end

surf(x1, x2, f)