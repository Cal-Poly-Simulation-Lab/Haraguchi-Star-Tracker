close all;
clear;
clc;

sa = [0, 1, -5, 1, 1;
      1, 3, 0, -1, 1;
      2, 0, 1, 4, 1];

for i = 1:5
    sa(:,i) = sa(:,i) / norm(sa(:,i));
end
sb = [0.9082, 0.567, -0.2821, 0.751, 0.9261;
      0.3185, 0.3732, 0.7163, -0.3303, -0.2053;
      0.2715, -.7343, 0.6382, 0.5718, -0.3166];
w = [1 / 0.01^2; 1 / 0.0325^2; 1 / 0.055^2; 1 / 0.0775^2; 1 / 0.1^2];

B = zeros(3,3);
lam0 = 0;
for i = 1:5
    B = B + w(i) * sa(:,i) * sb(:,i)';
    lam0 = lam0 + w(i);
end
B = B';

K12 = [B(2,3) - B(3,2);
       B(3,1) - B(1,3);
       B(1,2) - B(2,1)];
K22 = trace(B);

S = B + B';

a = K22^2 - trace(adjoint(S));
b = K22^2 + K12' * K12;
c = det(S) + K12' * S * K12;
d = K12' * (S*S) * K12;

lam = newtonRaphson(lam0, a, b, c, d, K22);
alpha = lam^2 - K22^2 + trace(adjoint(S));
beta = lam - K22;
gamma =(lam + K22) * alpha - det(S);
x = (alpha * eye(3) + beta * S + (S*S)) * K12;

p = inv((lam + K22) * eye(3) - S) * K12;
q = 1 / sqrt(1 + p' * p) * [p; 1]

% q = 1 / sqrt(gamma^2 + x' * x) * [x; gamma]
C = q2C(q)

C_book = [0.4153, 0.4472, 0.7921;
          -0.7562, 0.6537, 0.0274;
          -0.5056, -0.6104, 0.6097];
C2q(C_book)

function [lam] = newtonRaphson(lam, a, b, c, d, K22)
h = func(lam, a, b, c, d, K22) / funcPrime(lam, a, b, c);
while abs(h) >= 0.0001
    h = func(lam, a, b, c, d, K22) / funcPrime(lam, a, b, c);
    lam = lam - h ;
end
end

function [x] = func(lam, a, b, c, d, K22)
x = lam^4 - (a + b) * lam^2 - c * lam + (a * b + c * K22 - d);
end

function [x] = funcPrime(lam, a, b, c)
x = 4 * lam^3 - 2 * (a + b) * lam - c;
end

function [C] = q2C(q)
epsilon = q(1:3);
eta = q(4);
C = (eta^2 - epsilon' * epsilon) * eye(3) + 2 * (epsilon * epsilon') - 2 * eta * crossMatrix(epsilon);
end

function [X_x] = crossMatrix(X)
X_x = [0, -X(3), X(2); X(3), 0, -X(1); -X(2), X(1), 0];
end

function [q] = C2q(C)
q = zeros(4,1);
q(4) = 0.5 * sqrt(1 + trace(C));
if q(4) ~= 0
    q(1) = 0.25 * (C(2,3) - C(3,2)) / q(4);
    q(2) = 0.25 * (C(3,1) - C(1,3)) / q(4);
    q(3) = 0.25 * (C(1,2) - C(2,1)) / q(4);
else
    q(1) = sqrt((1 + C(1,1)) / 2);
    q(2) = sqrt((1 + C(2,2)) / 2);
    q(3) = sqrt((1 + C(3,3)) / 2);
end
end

% q = [-0.17808894; 0.42039834; -0.38820679; 0.80052798]
% q2C(q)

% d = 0;
% p = 0;
% product = [];
% determ = [];
% idx = [];
% for a = 0:360
%     idx = [idx; a];
%     M = [sin(a) * cos(p) - cos(a) * sin(d) * sin(p), -sin(a) * sin(p) - cos(a) * sin(d) * cos(p), -cos(a) * cos(d);
%          -cos(a) * cos(p) - sin(a) * sin(d) * sin(p), cos(a) * sin(p) - sin(a) * sin(d) * cos(p), -sin(a) * cos(d);
%          cos(a) * sin(p), cos(a) * cos(p), -sin(d)];
%     % M1 = [cos(a - pi/2), -sin(a - pi/2), 0;
%     %       sin(a - pi/2), cos(a - pi/2), 0;
%     %       0, 0, 1];
%     % M2 = [1, 0, 0;
%     %       0, cos(d + pi/2), -sin(d + pi/2);
%     %       0, sin(d + pi/2), cos(d + pi/2)];
%     % M3 = [cos(p), -sin(p), 0;
%     %       sin(p), cos(p), 0;
%     %       0, 0, 1];
%     % M = M1 * M2 * M3;
%     product = [product; sum(M.' * M, "all")];
%     determ = [determ; det(M)];
% end
% 
% figure
% subplot(2,1,1)
% plot(idx, determ)
% subplot(2,1,2)
% plot(idx, product)
