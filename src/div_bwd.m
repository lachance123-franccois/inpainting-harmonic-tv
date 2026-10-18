function d = div_bwd(px, py)
%Divergence discrete par differences arriere, adjointe de -grad_fwd.
%   Pour tous u, p :  sum(grad_fwd(u) .* p) = -sum(u .* div_bwd(p))
%   (propriete verifiee numeriquement dans main.m)

dx = [px(:, 1), diff(px(:, 1:end-1), 1, 2), -px(:, end-1)];
dy = [py(1, :); diff(py(1:end-1, :), 1, 1); -py(end-1, :)];
d = dx + dy;
end
