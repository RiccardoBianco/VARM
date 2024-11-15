close all;
clear all;

%% Load an (undistorted) image and the detected corners
img_index = 1;

undimg_path = sprintf('../data/images_undistorted/img_%04d.jpg', img_index);
undimg = imread(undimg_path);

K = load('../data/K.txt');

p_W_corners = 0.01 * load('../data/p_W_corners.txt');
num_corners = length(p_W_corners);

% Load the 2D projected points (detected on the undistorted image)
% TODO: Your code here


%% Now that we have the 2D <-> 3D correspondences (pts2d+normalized <-> p_W_corners),
%% let's find the camera pose with respect to the world using DLT

% TODO: Your code here

%% Plot the original 2D points and the reprojected ones on the image

% TODO: Your code here

%{ 
Remove this comment if you have completed the code until here
figure(1);
imshow(undimg); hold on;
plot(pts2d(:,1), pts2d(:,2), 'o'); hold on;
plot(p_reproj(:,1), p_reproj(:,2), '+');
legend('Original points','Reprojected points');
%}

%% Produce a 3d plot containing the corner positions and a visualization of the camera axis

%{ 
Remove this comment if you have completed the code until here
figure(2);
scatter3(p_W_corners(:,1), p_W_corners(:,2), p_W_corners(:,3)); hold on;
axis equal;

camup([0 1 0]);
view([0 0 -1]);
%}

% Position of the camera given in the world frame
% TODO: Your code here

%{ 
Remove this comment if you have completed the code until here
scaleFactorArrow = .05;

axisX = quiver3(pos(1),pos(2),pos(3), rotMat(1,1),rotMat(2,1),rotMat(3,1), 'r', 'ShowArrowHead', 'on', 'AutoScale', 'on', 'AutoScaleFactor', scaleFactorArrow);
axisY = quiver3(pos(1),pos(2),pos(3), rotMat(1,2),rotMat(2,2),rotMat(3,2), 'g', 'ShowArrowHead', 'on', 'AutoScale', 'on', 'AutoScaleFactor', scaleFactorArrow);
axisZ = quiver3(pos(1),pos(2),pos(3), rotMat(1,3),rotMat(2,3),rotMat(3,3), 'b', 'ShowArrowHead', 'on', 'AutoScale', 'on', 'AutoScaleFactor', scaleFactorArrow);
%}
