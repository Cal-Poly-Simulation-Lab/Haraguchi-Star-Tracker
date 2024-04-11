%% star field generation in matlab to test

close all; clear; clc

catalog = readtable('bs5_brief.csv');

ra_hour = catalog.RA;
ra_min = catalog.m;
ra_sec = catalog.s;
dec_dir = catalog.DecDir;
dec_deg = catalog.DecDeg;
dec_min = catalog.DecM;
dec_sec = catalog.DecS;

ra_all = zeros(9096,1);
dec_all = zeros(9096,1);
for i = 1:9096
    ra_all(i) = hours2rad(ra_hour(i), ra_min(i), ra_sec(i));
    if strcmp(dec_dir(i), 'N')
        dec_all(i) = minsec2rad(dec_deg(i), dec_min(i), dec_sec(i));
    elseif strcmp(dec_dir(i), 'S')
        dec_all(i) = -1 * minsec2rad(dec_deg(i), dec_min(i), dec_sec(i));
    end
end

% parameters for function input
q_ECI_b = [0; 0; 0; 1];
q_b_st = [0; 0; 0; 1];
FOV = 40;
f = 1.4;
h = 600;
w = 1024;

imageGeneration(ra_all, dec_all, q_ECI_b, q_b_st, FOV, f, h, w);


%% functions

function [rad] = hours2rad(hr, min, s)
    hr = hr + (min / 60);
    hr = hr + (s / 3600);
    rad = hr * 15 * pi / 180;
end

function [rad] = minsec2rad(deg, min, s)
    deg = deg + (min / 60);
    deg = deg + (s / 3600);
    rad = deg * pi / 180;
end

function [] = imageGeneration(ra_all, dec_all, q_ECI_b, q_b_st, FOV, f, h, w)
% unit vector of star tracker boresight angle in ECI
% assume boresight is aligned with x body frame
alpha = 2 * pi;
delta = 0;
phi = 0; % no roll

M = [sin(alpha) * cos(phi) - cos(alpha) * sin(delta) * sin(phi), -sin(alpha) * sin(phi) - cos(alpha) * sin(delta) *  cos(phi), -cos(alpha) * cos(delta);
     -cos(alpha) * cos(phi) - sin(alpha) * sin(delta) * sin(phi), cos(alpha) * sin(phi) - sin(alpha) * sin(delta) * cos(phi), -sin(alpha) * cos(delta);
     cos(alpha) * sin(phi), cos(alpha) * cos(phi), -sin(delta)];

FOV = 40 * pi / 180;
R = sqrt(2 * FOV^2) / 2;

ra_min = alpha - R / cos(delta);
ra_max = alpha + R / cos(delta);
dec_min = delta - R;
dec_max = delta + R;

ra = [];
dec = [];
for i = 1:9096
    if dec_all(i) > dec_min && dec_all(i) < dec_max
        if ra_all(i) > ra_min || ra_all(i) < (ra_max - 2 * pi)
            ra = [ra; ra_all(i)];
            dec = [dec; dec_all(i)];
        end
    end
end

num_stars = length(ra);
disp("number of stars found = " + num_stars)

F = [f, 0, w/2; 0, f, h/2; 0, 0, 1];

u_star_ECI = zeros(num_stars, 3);
u_star_st = zeros(num_stars, 3);
r = zeros(num_stars, 1);
c = zeros(num_stars, 1);
for i = 1:num_stars
    % u_star_ECI(i,1) = cos(ra(i)) * cos(dec(i));
    % u_star_ECI(i,2) = sin(ra(i)) * cos(dec(i));
    % u_star_ECI(i,3) = sin(dec(i));
    % u_star_st(i,:) = (M' * u_star_ECI')';

    x = [cos(dec(i)) * cos(ra(i));
         cos(dec(i)) * sin(ra(i));
         sin(dec(i))];
    X = F * M' * x;
    c(i) = floor(X(1));
    r(i) = floor(X(2));
end

img = zeros(h, w);
for i = 1:num_stars
    img(r,c) = 255;
end

figure
imshow(img)

end
