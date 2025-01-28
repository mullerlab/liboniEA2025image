function W = gaussian_sheet(Nrow, Ncol, a, s, phi)

% *CV-RNN*
%
% GAUSSIAN SHEET   wire network with distance-dependent connections
%
% INPUTS
% Nrow : # of rows in image
% Ncol : # of cols in image
% s : standard dev of gaussian
% a : amplitude of gaussian
%
% OUTPUT
% W : weighted adjacency matrix 

drow = 1 / Nrow;
row = drow : drow : 1;
dcol = 1 / Ncol;
col = dcol : dcol : 1;
[ROW,COL] = meshgrid(row,col);
pos = [ROW(:) COL(:)];
D = pdist2(pos, pos, 'euclidean');
W = a * exp( -D.^2 / 2 / s^2 );
W2 =  exp( -D.^2 );

if nargin > 4

        W( W < 0.04 ) = W( W < 0.04 ) * exp( 1i * phi );
       phi = ( ( ( D - min( D(:) ) ) ./ (max( D(:) ) - min( D(:) ) ) ) * phi_range ) + phi_min;

end

end