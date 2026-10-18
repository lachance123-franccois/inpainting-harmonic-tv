function [u, energy] = inpaint_harmonic(f, mask, method, nIter)
%INPAINT_HARMONIC  Inpainting par diffusion (equation de Laplace dans le trou).
%   [u, energy] = INPAINT_HARMONIC(f, mask, method, nIter)
%     f      : image (double), valeurs dans le masque ignorees
%     mask   : logique, true = pixel manquant
%     method : 'direct'   resolution exacte du systeme lineaire creux  (defaut)
%              'explicit' equation de la chaleur, schema explicite (nIter iterations)
%   u      : image restauree
%   energy : energie  sum |grad u|^2 au fil des iterations ('explicit')

if nargin < 3 || isempty(method), method = 'direct'; end
if nargin < 4 || isempty(nIter), nIter = 2000; end
[H, W] = size(f);
u = f;
u(mask) = mean(f(~mask));
energy = [];

switch method
    case 'direct'
        unk = find(mask);
        n = numel(unk);
        id = zeros(H, W);
        id(unk) = 1:n;
        [ii, jj] = ind2sub([H W], unk);
        rows = []; cols = []; vals = [];
        deg = zeros(n, 1);
        rhs = zeros(n, 1);
        shifts = [0 1; 0 -1; 1 0; -1 0];
        for s = 1:4
            ni = ii + shifts(s, 1);
            nj = jj + shifts(s, 2);
            inside = ni >= 1 & ni <= H & nj >= 1 & nj <= W;
            deg = deg + inside;
            p = find(inside);
            q = sub2ind([H W], ni(inside), nj(inside));
            qUnk = mask(q);
            % voisin inconnu  coefficient -1 dans la matrice
            rows = [rows; p(qUnk)]; 
            cols = [cols; id(q(qUnk))]; 
            vals = [vals; -ones(nnz(qUnk), 1)]; 
            % voisin connu  passe au second membre
            rhs = rhs + accumarray(p(~qUnk), f(q(~qUnk)), [n 1]);
        end
        A = sparse([rows; (1:n)'], [cols; (1:n)'], [vals; deg], n, n);
        u(unk) = A \ rhs;
        [ux, uy] = grad_fwd(u);
        energy = sum(ux(:).^2 + uy(:).^2);

    case 'explicit'
            dt = 0.24; tol = 1e-4; res = inf; k = 0;
            energy = zeros(1, nIter);
        while res > tol && k < nIter
            k = k + 1;
            up = u([1 1:end-1], :); dn = u([2:end end], :);
            lf = u(:, [1 1:end-1]); rt = u(:, [2:end end]);
            lap = up + dn + lf + rt - 4*u;
            res = max(abs(lap(mask)));          % = max |b - A x|
            u(mask) = u(mask) + dt * lap(mask);
            [ux, uy] = grad_fwd(u);
            energy(k) = sum(ux(:).^2 + uy(:).^2);
        end
        energy = energy(1:k);
        if res > tol
            warning('Non convergé : %d itérations, résidu %.2e', k, res);
        end
end
end
