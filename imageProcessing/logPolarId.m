close all; clear; clc;

% read in image
img = imread("sampleStellarium.png");
img = rgb2gray(img);
figure
imshow(img,[])

% identify key points
H = fspecial('gaussian', 5, 1);
% img_fil = imfilter(img, H);
[r, c] = harris(img, 1.5, 6, 0.05, 0.1);
figure
imshow(img, [])
hold on
for i = 1:length(r)
    x = c(i);
    y = r(i);
    viscircles([x, y], 8, 'EdgeColor', 'r', 'LineWidth', 1);
    % rectangle('Position', [x-5 y-5 10 10], 'EdgeColor', 'r', 'LineWidth', 1);
end
hold off

features = logPolar(img, [r, c]);

% pick brightest star


function [row, col] = harris(img, sigma, window, alpha, thresh_percent)
% inputs 
%   img = grayscale image
%   sigma = sigma value for gaussian window
%   window = window size 
%   alpha = constant 0.04-0.06
%   R_thresh = threhsold value for detected corners 
% output
%   row, col = detected corner locations 
% gradient using sobel
Sx = [-1, -2, -1; 0, 0, 0; 1, 2, 1];
Sy = Sx';
Ix = double(imfilter(img, Sx));
Iy = double(imfilter(img, Sy));
% second moment matrix
Ixx = Ix .* Ix;
Ixy = Ix .* Iy;
Iyy = Iy .* Iy;
W = fspecial('gaussian', round(window * sigma + 1), sigma);
M11 = imfilter(Ixx, W);
M12 = imfilter(Ixy, W);
M22 = imfilter(Iyy, W);
% harris corner response function
R = (M11 .* M22 - M12 .* M12) - alpha * (M11 + M22).^2; 
R_thresh = max(max(R)) * thresh_percent;
% nonmaxima suprression and find local max
Lmax = (R==imdilate(R, strel('disk', 19)) & R > R_thresh);
[row, col] = find(Lmax);
end

function [features] = logPolar(img, detected_pts)
[row,col] = size(img);
[n,~] = size(detected_pts);
figure
hold on
for i = 1:n
    x = detected_pts(i,1);
    y = detected_pts(i,2);
    r = log(sqrt(x^2 + y^2));
    theta = atan(y/x);
    if x < 0
        theta = pi + theta;
    elseif x > 0 && y < 0
        theta = 2 * pi + theta;
    end
    stem(theta, r)
end
hold off
features = [];
end