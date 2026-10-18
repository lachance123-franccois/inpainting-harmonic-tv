function [ux, uy] = grad_fwd(u)
%Gradient discret par differences avant (conditions de Neumann).
%   ux(i,j) = u(i,j+1) - u(i,j), nul sur la derniere colonne
%   uy(i,j) = u(i+1,j) - u(i,j), nul sur la derniere ligne
[H, W] = size(u);
ux = [diff(u, 1, 2), zeros(H, 1)];
uy = [diff(u, 1, 1); zeros(1, W)];
end
