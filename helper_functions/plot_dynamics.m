function h = plot_dynamics( x, layer_1_final_time )
% *CV-RNN*
%
% PLOT DYNAMICS    plot the dynamics for the cv-RNN in an image plot
%
% INPUTS
% x - datacube containing the phase dynamics of the network
%
% OUTPUTS
% h - axis handle for the plot
%

% set up plot
h = imagesc( x(:,:,1) ); colormap hsv
axis off; set( gca, 'fontname', 'arial', 'fontsize', 15 );
t1 = text( 'string', '1', 'fontname', 'arial', 'fontsize', 15, 'position', [13, -1 ,0] );

% visualize dynamics over a given time range
times = 2:180;
for ii = times
    tmp = x(:,:,ii);
    set( h,'cdata', tmp ); 
    set( h, 'AlphaData', ~isnan(tmp) );
    if( ii > layer_1_final_time ), clim( [-pi,pi] ); end
    set( t1, 'string', sprintf( '%d timesteps', ii ) );
    pause( 0.1 );
end