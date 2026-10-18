function mask = make_mask(type, sz, param)
%Masques de pixels manquants (true = manquant).
%   mask = MAKE_MASK('rayures', sz, nRayures)   traits epais aleatoires
%   mask = MAKE_MASK('carre', sz, cote)         trou carre au centre
%   mask = MAKE_MASK('aleatoire', sz, p)        perte aleatoire d'une proportion p des pixels
%   mask = MAKE_MASK('texte', sz, chaine)       texte incruste (rendu via une figure)

switch type
    case 'rayures'
        if nargin < 3
            param = 12; % param : le nombre de rayures voulu sur l'image
        end 
        mask = false(sz, sz);
        [X, Y] = meshgrid(1:sz);
        for k = 1:param
            p0 = rand(1, 2) * sz;
            ang = rand * pi;
            len = (0.2 + 0.4*rand) * sz;
            p1 = p0 + len * [cos(ang) sin(ang)];
            w = 1 + 2*rand;                    % demi-epaisseur 1 a 3 px
            d=dist_to_segment(X, Y, p0, p1);  % renvoi une matrice de meme taille que le mask et qui donne la distance de chaque pixel par rapport au segment (rayureS)
            new= d< w;
            mask = mask | new ;
        end
    case 'carre'
        mask = false(sz, sz);
        c = round(sz/2);
        h = floor(param/2);
        mask(c-h : c-h+param-1, c-h : c-h+param-1) = true;
    case 'aleatoire'
        mask = rand(sz, sz) < param;
    case 'texte'
        if nargin < 3
            param = 'INPAINTING';
        end
        mask = render_text(param, sz);
    otherwise
        error('make_mask:type', 'Type inconnu : %s', type);
end
end

function d = dist_to_segment(X, Y, p0, p1)
v = p1 - p0;
t = ((X - p0(1)) * v(1) + (Y - p0(2)) * v(2)) / (v * v');
t = min(max(t, 0), 1);
d = hypot(X - (p0(1) + t*v(1)), Y - (p0(2) + t*v(2)));
end

function mask = render_text(str, sz)
% Rendu du texte dans une figure invisible puis capture
fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100 100 sz sz], 'Color', 'w');
ax = axes(fig, 'Position', [0 0 1 1], 'XLim', [0 1], 'YLim', [0 1], 'Visible', 'off');
text(ax, 0.5, 0.62, str, 'FontSize', round(sz/9), 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Units', 'normalized');
text(ax, 0.5, 0.38, str, 'FontSize', round(sz/9), 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'Units', 'normalized');
frame = getframe(fig);
close(fig);
g = double(frame.cdata(:, :, 1)) / 255;
[Xq, Yq] = meshgrid(linspace(1, size(g, 2), sz), linspace(1, size(g, 1), sz));
mask = interp2(g, Xq, Yq) < 0.5;
end
