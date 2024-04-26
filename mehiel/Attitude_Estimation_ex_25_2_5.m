%% Aero 421 - q-Method, QUEST, and TRIAD Example 25.2.5
% Eric Mehiel
% Aero 421
% Cal Poly, Aerospace Engineering

clear
clc
close all

% Cba = Czd(60)*Cyd(-30)*Cxd(45);

Sa = [0 1 -5 1 1;
      1 3 0 -1 1;
      2 0 1 4 1];

for i = 1:size(Sa, 2)
    Sa(:,i) = Sa(:,i)/norm(Sa(:,i));
end

sig = [.01; .0325; .055; .0775; .1];

Sb = [.9082 .5670 -.2821 .7510 .9261;
      .3185 .3732 .7163 -.3303 -.2053;
      .2715 -.7343 .6382 .5718 -.3166];

% Cba_t_est = triad(Sa(:,1), Sa(:,2), Sb(:,1), Sb(:,2));

% [q, Cba_qm_est] = q_method(Sa, Sb, sig);

[q_Q, Cba_Q_est] = quest(Sa, Sb, sig, .01)

% J_t = J(Sa, Sb, sig, Cba_t_est)
% J_qm = J(Sa, Sb, sig, Cba_qm_est)
% J_Q = J(Sa, Sb, sig, Cba_Q_est)
% 
% function c = J(Sa, Sb, sig, Cba)
%     c = 0;
%     for k = 1:size(Sa,2)
%         r = Sb(:,k) - Cba*Sa(:,k);
%         c = c + 1/sig(k)^2*(r')*r;
%     end
% end
