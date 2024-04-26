function [q, C_ba] = quest(Sa, Sb, sig, tol)

lam_0 = 0;
for k = 1:length(sig)
    lam_0 = lam_0 + 1/sig(k)^2;
end

B = zeros(3,3);
for k = 1:length(sig)
    B = B + 1/sig(k)^2*Sa(:,k)*Sb(:,k)';
end
B = B';
S = B + B';

k22 = trace(B);
k12 = [B(2,3)-B(3,2); B(3,1)-B(1,3); B(1,2)-B(2,1)];
K11 = S - k22*eye(3);

a = k22^2 - trace(adjoint(S));
b = k22^2 + k12'*k12;
c = det(S) + k12'*S*k12;
d = k12'*S^2*k12;

f = @(x) x^4 - (a + b)*x^2 - c*x + (a*b + c*k22 - d);
fprime = @(x) 4*x^3 - 2*(a + b)*x - c;

lam = newtons(f, fprime, lam_0, tol);

alpha = lam^2 - k22^2 + trace(adjoint(S));
beta = lam - k22;
gamma = (lam + k22)*alpha - det(S);

x = (alpha*eye(3) + beta*S + S^2)*k12;
q = 1/(sqrt(gamma^2 + norm(x)^2))*[x; gamma];

C_ba = quat2C(q);

end

function x_1 = newtons(f, fprime, x_0, TOL)
    e = 2*TOL;
    
    while abs(e) > TOL
        x_1 = x_0 - f(x_0)/fprime(x_0);
        e = x_1 - x_0;
        x_0 = x_1;
    end
end

function C = quat2C(q) 
epsilon = q(1:3);
eta = q(4);
C = (eta^2 - epsilon' * epsilon) * eye(3) + 2 * (epsilon * epsilon') - 2 * eta * crossMatrix(epsilon);
end

function [X_x] = crossMatrix(X)
X_x = [0, -X(3), X(2); X(3), 0, -X(1); -X(2), X(1), 0];
end