%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                           %
% cv-RNN FOR IMAGE SEGMENTATION             %
% MULLER LAB                                %
% AUGUST 2024                               %
%                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% setup

clearvars; clc;

% add directories to path
addpath( './helper_functions' );
addpath( './image_segmentation' ); 
addpath( './graphs' );

%#ok<*FNDSB> 
%#ok<*ASGLU>
%#ok<*NASGU>

%% cv-RNN hyperparameters - n.b. same for all examples

% amplitude (alpha) and spatial spread (sigma) of Gaussian connection weights
alpha = [0.5 0.5]; sigma = [0.9 0.0313]; % parameters for layer 1 and 2 in each array

% time points for recurrent dynamics in layer 1 and layer 2
layer_1_time_range = 1:60; layer_2_time_range = 61:200;
layer_time_points = [ layer_1_time_range(end) layer_2_time_range(end) ];

% parameters for spatiotemporal clustering
dim = 1:3;                                                                 % select dimensions of projection to plot 
window_size = 40; window_step = 40;                                        % select time windows to analyse 

%% EXAMPLE 1: 2-shapes dataset
%
% this example demonstrates segmenting an image with two objects in the 
% cv-RNN (as in Figure 3 of the manuscript). three example images from the
% 2Shapes dataset in Löwe et al. [https://arxiv.org/abs/2204.02075]
% are included in ./dataset/2shapes.mat.
%

close all; clearvars -except alpha sigma layer_1_time_range layer_2_time_range ...
    layer_time_points dim window_size window_step

% load 2Shapes dataset
load( 'dataset/2shapes.mat' );

% choose example image - can select from image 1, 2, or 3
image_number = 1; 
im = images(:,:,image_number); lb = labels(:,:,image_number);

% cv-RNN dynamics
random_seed = 1;
[X, mask] = run_2layer( im, alpha, sigma, layer_time_points, random_seed ); 
x = reshape( angle(X), size(im,1), size(im,2), [] );
mask = reshape( mask, size(im,1), size(im,2) );

% plotting 1 - plot image input to the system
fg1 = figure; imagesc( im ); colormap( flipud(gray) ); 
set( gca, 'xtick', [], 'ytick', [], 'fontname', 'arial', 'fontsize', 15 ); 
set( gcf, 'position', [223   684   281   205]); title( 'input' );

% plotting 2 - visualizing cv-RNN dynamics
fg2 = figure; layer_1_time_final = layer_time_points(1);
h = plot_dynamics( x, layer_1_time_final );

% spectral clustering
x = reshape( x, size(im,1)*size(im,2), [] ); x = exp( 1i*x ); 
mask = reshape( mask, [], 1 ); 
x = x( ~mask, : );                                                         % consider nodes that have not been labeled as background 

[rho, V, D, prj] = spatiotemporal_segmentation( ...                        %%% outputs from this calculation:
     x, dim, layer_time_points, window_size, window_step );                % rho: similarity matrix (nodes x nodes x time window) 
                                                                           % V: eigenvectors of rho
                                                                           % D: eigenvalues of rho
                                                                           % prj: rho*V

% plotting 3 - visualize result of spectral clustering
w = size( prj, 3 );                                                        % select window to plot (here selecting the last index)
fg3 = figure; set( gcf, 'position', [560   151   552   330]);
colors = angle( x(:,120) );                                                % colour nodes by their phase during this window
scatter3( prj(:,1,w), prj(:,2,w), prj(:,3,w), 50, colors, 'filled' ); 
colormap( 'hsv' ); clim( [-pi,pi] );
xlabel( 'dimension 1' ); ylabel( 'dimension 2' ); zlabel( 'dimension 3' );
set( gca, 'fontname', 'arial', 'fontsize', 15 );
title( 'similarity projection' );
c = colorbar; c.Label.String = 'phase (rad)';
predict = kmeans( prj(:,1:3,end), 2 );                                     % here, we are using k-means for the last step of obtaining
                                                                           % object labels. note that the objects are fully separated by the
                                                                           % 3-dimensional projection (cf. Fig. 3C), and that this step simply
                                                                           % assigns a label to the cluster

% plotting 4 - visualize segmentation result
objects = find( mask == 0 );                                               % nodes that are part of the objects
segmented_image = double( ~(mask) ); segmented_image( objects ) = predict; % set object nodes to their predicted label 
segmented_image = reshape( segmented_image, size(im,1), size(im,2) );      % reshape to image size 
fg4 = figure; set( gcf, 'position', [230   223   281   205])
p1 = imagesc( segmented_image );
set( gca, 'xtick', [], 'xtick', [], 'fontname', 'arial', 'fontsize', 15 );
set( p1, 'alphadata', ~reshape( mask, size(im) ) )
title( 'segmented image' );

%% EXAMPLE 2: 3-shapes dataset
%
% this example demonstrates segmenting images with three shapes in the
% cv-RNN (as in Figure 4A of the manuscript). three example images from the
% 3Shapes dataset in Löwe et al. [https://arxiv.org/abs/2204.02075] are
% included in ./dataset/3shapes.mat. 
%

close all; clearvars -except alpha sigma layer_1_time_range layer_2_time_range ...
    layer_time_points dim window_size window_step

% load 3shapes dataset 
load( 'dataset/3shapes.mat' );

% choose example image - can select from image 1, 2, or 3
image_number = 1; 
im = images(:,:,image_number); lb = labels(:,:,image_number);

% plotting 1 - plot image input to the system
fg5 = figure; imagesc( im ); colormap( flipud(gray) ); 
set( gca, 'xtick', [], 'ytick', [], 'fontname', 'arial', 'fontsize', 15 ); 
set( gcf, 'position', [223   684   281   205]); title( 'input' );

% cv-RNN dynamics
random_seed = 9;
[X, mask] = run_2layer( im, alpha, sigma, layer_time_points, random_seed ); 
x = reshape( angle(X), size(im,1), size(im,2), [] );
mask = reshape( mask, size(im,1), size(im,2) );

% plotting 2 - visualizing cv-RNN dynamics
fg6 = figure;
layer_1_time_final = layer_time_points(1);
h = plot_dynamics( x, layer_1_time_final );

% spectral clustering
x = reshape( x, size(im,1)*size(im,2), [] ); x = exp( 1i*x ); 
mask = reshape( mask, [], 1 ); 
x = x( ~mask, : );                                                         % consider nodes that have not been labeled as background 

[rho, V, D, prj] = spatiotemporal_segmentation( ...                        %%% outputs from this calculation:
     x, dim, layer_time_points, window_size, window_step );                % rho: similarity matrix (nodes x nodes x time window) 
                                                                           % V: eigenvectors of rho
                                                                           % D: eigenvalues of rho
                                                                           % prj: defined to be rho*V

% plotting 3 - visualize result of spectral clustering
w = size( prj, 3 );                                                        % select window to plot (here selecting the last index)
fg7 = figure; set( gcf, 'position', [560   151   552   330]);
colors = angle( x(:,120) );                                                % colour nodes by their phase during this window
scatter3( prj(:,1,w), prj(:,2,w), prj(:,3,w), 50, colors, 'filled' ); 
colormap( 'hsv' ); clim( [-pi,pi] );
xlabel( 'dimension 1' ); ylabel( 'dimension 2' ); zlabel( 'dimension 3' );
set( gca, 'fontname', 'arial', 'fontsize', 15 );
title( 'similarity projection' );
c = colorbar; c.Label.String = 'phase (rad)';
predict = kmeans( prj(:,1:3,end), 3 );                                     % here, we are using k-means for the last step, as on line 89.
                                                                           % as above, note that the objects are fully separated by the 
                                                                           % 3-dimensional projection in Figure 4A.

% plotting 4 - visualize segmentation result
objects = find( mask == 0 );                                               % nodes that are part of the objects
segmented_image = double( ~(mask) ); segmented_image( objects ) = predict; % set object nodes to their predicted label 
segmented_image = reshape( segmented_image, size(im,1), size(im,2) );      % reshape to image size 
fg8 = figure; set( gcf, 'position', [230   223   281   205])
p1 = imagesc( segmented_image );
set( gca, 'xtick', [], 'xtick', [], 'fontname', 'arial', 'fontsize', 15 );
set( p1, 'alphadata', ~reshape( mask, size(im) ) )
title( 'segmented image' );

%% EXAMPLE 3: natural image
%
% this example demonstrates segmenting a natural image in the cv-RNN (as in
% Figure 4C of the manuscript).
% 

close all; clearvars -except alpha sigma layer_1_time_range layer_2_time_range ...
    layer_time_points dim window_size window_step

% load example natural image
load( 'dataset/natural_image.mat' );

% plotting 1 - plot original image 
fg9 = figure; imagesc( in ); colormap( gray ); 
set( gca, 'xtick', [], 'ytick', [], 'fontname', 'arial', 'fontsize', 15 ); 
set( gcf, 'position', [19   670   281   205]); title( 'original image' );

% plotting 2 - plot image input to system 
fg10 = figure; imagesc( im ); colormap( gray ); 
set( gca, 'xtick', [], 'ytick', [], 'fontname', 'arial', 'fontsize', 15 ); 
set( gcf, 'position', [301   670   281   205]); title( 'input' ); 

% cv-RNN dynamics
random_seed = 1;
[X, mask] = run_2layer( im, alpha, sigma, layer_time_points, random_seed ); 
x = reshape( angle(X), size(im,1), size(im,2), [] );
mask = reshape( mask, size(im,1), size(im,2) );

% plotting 3 - visualizing cv-RNN dynamics
fg11 = figure;
layer_1_time_final = layer_time_points(1);
h = plot_dynamics( x, layer_1_time_final );

% spectral clustering
x = reshape( x, size(im,1)*size(im,2), [] ); x = exp( 1i*x ); 
mask = reshape( mask, [], 1 ); 
x = x( ~mask, : );                                                         % consider nodes that have not been labeled as background 

[rho, V, D, prj] = spatiotemporal_segmentation( ...                        %%% outputs from this calculation:
     x, dim, layer_time_points, window_size, window_step );                % rho: similarity matrix (nodes x nodes x time window) 
                                                                           % V: eigenvectors of rho
                                                                           % D: eigenvalues of rho
                                                                           % prj: defined to be rho*V

% plotting 4 - visualize result of spectral clustering
w = size( prj, 3 );                                                        % select window to plot (here selecting the last index)
fg12 = figure; set( gcf, 'position', [560   151   552   330]);
colors = angle( x(:,120) );                                                % colour nodes by their phase during this window
scatter3( prj(:,1,w), prj(:,2,w), prj(:,3,w), 50, colors, 'filled' ); 
colormap( 'hsv' ); clim( [-pi,pi] );
xlabel( 'dimension 1' ); ylabel( 'dimension 2' ); zlabel( 'dimension 3' );
set( gca, 'fontname', 'arial', 'fontsize', 15 );
title( 'similarity projection' );
c = colorbar; c.Label.String = 'phase (rad)';
predict = kmeans( prj(:,1:3,end), 2 );                                     % here, we are using k-means for the last step, as on line 89.
                                                                           % as above, note that the objects are fully separated by the 
                                                                           % 3-dimensional projection in Figure 4C.

% plotting 5 - visualize segmentation result
objects = find( mask == 0 );                                               % nodes that are part of the objects
segmented_image = double( ~(mask) ); segmented_image( objects ) = predict; % set object nodes to their predicted label 
segmented_image = reshape( segmented_image, size(im,1), size(im,2) );      % reshape to image size 
fg13 = figure; set( gcf, 'position', [230   223   281   205] )
p1 = imagesc( segmented_image );
set( gca, 'xtick', [], 'xtick', [], 'fontname', 'arial', 'fontsize', 15 );
set( p1, 'alphadata', ~reshape( mask, size(im) ) )
title( 'segmented image' );