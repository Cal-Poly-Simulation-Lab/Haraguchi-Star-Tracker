close all; clear; clc;

% for big dipper image
ra = 195;
dec = 55;
roll = 180;
unit = [cosd(dec) * cosd(ra);
        cosd(dec) * sind(ra);
        sind(dec)];

q = [0.58072572; 0.75641404; 0.23865935; 0.18340433];
C = q2C(q);
C_true = [0.433, 0.4356, 0.7891;
          -0.75, 0.6597, 0.0474;
          -0.5, -0.6124, 0.6124];
C_error = C * C_true';
phi = acosd(0.5 * (trace(C_error) - 1))

[yaw, pitch, roll] = C2euler(C);
% rotate z into inertial
body = [0; 0; 1];
res = C * body
check = C * unit


function [C] = q2C(q)
epsilon = q(1:3);
eta = q(4);
C = (eta^2 - epsilon' * epsilon) * eye(3) + 2 * (epsilon * epsilon') - 2 * eta * crossMatrix(epsilon);
end

function [X_x] = crossMatrix(X)
X_x = [0, -X(3), X(2); X(3), 0, -X(1); -X(2), X(1), 0];
end

function [yaw, pitch, roll] = C2euler(C)
yaw = atan2d(C(2,3), C(3,3));
pitch = -asind(C(1,3));
roll = atan2d(C(1,2), C(1,1));
end
