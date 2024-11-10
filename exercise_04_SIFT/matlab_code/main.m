clear all
close all
clc

rotation_inv = 1;
rotation_img2_deg = 60;

num_scales = 3; % Scales per octave.
num_octaves = 5; % Number of octaves.
sigma = 1.0;
contrast_threshold = 0.04;
image_file_1 = '../data/img_1.jpg';
image_file_2 = '../data/img_2.jpg';
rescale_factor = 0.3; % Rescaling of the original image for speed.

left_img = getImage(image_file_1, rescale_factor);
right_img = getImage(image_file_2, rescale_factor);

% to test rotation invariance of SIFT
if rotation_img2_deg ~= 0
% TODO: Your code here
end

images = {left_img, right_img};

% Actually compute the SIFT features. For both images do:
% - construct the image pyramid
% - compute the blurred images
% - compute difference of gaussians
% - extract the keypoints
% - compute the descriptors
kpt_locations = cell(1, 2);
descriptors = cell(1, 2);

for img_idx = 1:2
% TODO: Your code here
end

%{ 
Remove this comment if you have completed the code until here
indexPairs = matchFeatures(descriptors{1}, descriptors{2},...
    'MatchThreshold', 100, 'MaxRatio', 0.7, 'Unique', true);
%}

% TODO: Your code here

%{ 
Remove this comment if you have completed the code until here
figure; ax = axes;
showMatchedFeatures(images{1}, images{2}, kpt_matched_1, kpt_matched_2, ...
    'montage','Parent',ax);
title(ax, 'Candidate point matches');
legend(ax, 'Matched points 1','Matched points 2');
%}
