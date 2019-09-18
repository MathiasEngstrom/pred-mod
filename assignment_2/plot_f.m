close all
clear all
clc

% Define paramaters
a = 0.5;
b = 2;
c1 = 2;
c2 = 3;
C = [c1;c2];

x1 = -2:0.1:7;
x2 = x1';

y = zeros(size(length(x1)));

for i=1:length(x1)
    for j=1:length(x1)
        x = [x1(i);x2(j)];
       
        y(i, j) = f(x, a, b, C);
       
   
    end
end

surf(x1, x2, y)
