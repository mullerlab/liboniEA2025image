function [save_x, mask] = run_2layer( im, a, s, nt, seed )
% *CV-RNN*
%
% RUN 2LAYER    run dynamics for cv-RNN layer 1 and layer 2
%
% INPUTS
% im = input image (size Nr x Nc )
% s = standard deviation of gaussian [layer 1, layer 2]
% a = amplitude of gaussian [layer 1, layer 2]
% kappa = update rate 
% nt = [ number of time steps when mask is applied, number of timesteps
% when trial ends ] 
% seed = initialization of random initial condition
%
% OUTPUTS 
% save_x = Nr*Nc x nt matrix of network dynamics (complex) 
% mask = Nr*Nc mask applied at timestep nt(1)

[Nr,Nc] = size(im);
K = gaussian_sheet(Nr, Nc,a(1),s(1)); 
rng(seed);
x0 = exp( 1i * ( rand(Nr*Nc,1) - 0.5 )*2*pi ); % initial condition
save_x = zeros( Nr*Nc, nt(2)); save_x(:,1) = x0;

omega = im(:); % inputs as intrisic frequencies

% layer 1
x = x0;
for ii = 2: nt(1)
    x = (diag(1i*omega) + K ) * x; 
    save_x(:,ii) = x;
end

% remove the background, find the mask
thr = mean(angle(save_x(:,nt(1))));
if sum(angle(save_x(:,nt(1)))>thr)>sum(angle(save_x(:,nt(1)))<thr)
    mask = angle(save_x(:,nt(1)))>thr;  % mask to drive second layer
else
    mask = angle(save_x(:,nt(1)))<thr; % mask to drive second layer
end

K = gaussian_sheet(Nr, Nc, a(2),s(2)); 
K(mask,:) = 0; K(:,mask) = 0;
omega(mask) = 0;

x02 = x0; % same initial condition
x02(mask) = 0;    
x = x02;
for ii = nt(1)+1:nt(2)
     x =  (diag(1i*omega) + K ) * x;
     save_x(:,ii) = x;
 end

save_x(mask,nt(1)+1:nt(2)) = nan;

end