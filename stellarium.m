close all
clear
clc

img = imread('sampleStellarium.png');
img = rgb2gray(img);
figure
imshow(img)
figure
imhist(img)
imc = img > 50;
figure
imshow(imc)

prewitt = edge(imc, 'prewitt');
figure
imshow(prewitt)
