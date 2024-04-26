function [q, C_ba] = q_method(Sa, Sb, sig)

Bt = zeros(3,3);
for k = 1:length(sig)
    Bt = Bt + 1/sig(k)^2*Sa(:,k)*Sb(:,k)';
end
B = Bt';

k22 = trace(B);
k12 = [B(2,3)-B(3,2); B(3,1)-B(1,3); B(1,2)-B(2,1)];
K11 = B + B' - k22*eye(3);
K = [K11 k12; k12' k22];

[P, Lam] = eig(K);

%lam_bar = Lam(4,4);
q = P(:,4);

C_ba = quat2C(q);