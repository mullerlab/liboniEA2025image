function [rho, V, D, prj] = spatiotemporal_segmentation( x, dim, win, ws, dw) 

% *CV-RNN*
%
% SPATIOTEMPORAL SEGMENTATION    use cv-RNN dynamics to cluster nodes 
%
% INPUTS
% X = pixels x time dynamics (complex) 
% dim = dimensions to consider
% win = [start timestep, end timestep]
% ws = window size 
% dw = window step 
%
% OUTPUTS
% rho = similarity matrix 
% V = eigenvectors
% D = eigenvalues
% prj = projection

window_start = win(1):dw:(win(2)-ws); window_end = window_start + ws;  
Nn = size(x,1); Nw = length(window_start); 
rho = zeros( Nn, Nn, Nw ); V = zeros( Nn, Nn, Nw ); D = zeros( Nn, Nn, Nw ); prj = zeros(Nn,length(dim), Nw);

for w = 1:Nw
    
    for ii = 1:Nn
        for jj = 1:Nn
            p = x(ii,window_start(w):window_end(w)); q = conj( x(jj,window_start(w):window_end(w)) );  
            rho(ii,jj,w) = sum(p.*q)/length(p); 
        end
    end
    
    [v,d] = eig(rho(:,:,w)); [~,ind] = sort( abs(diag(d)), 'descend'); 
    prj(:,:,w) = real(rho(:,:,w))*real(v(:,ind(dim))); 
    
    V(:,:,w) = v(:,ind); D(:,ind,w) = d; 
end

end