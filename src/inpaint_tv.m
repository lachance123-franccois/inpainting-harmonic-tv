function [u, tvHist] = inpaint_tv(f, mask, method, nIter, opts)


if nargin < 3 || isempty(method), method = 'cp'; end
if nargin < 4 || isempty(nIter), nIter = 500; end
if nargin < 5, opts = struct(); end
if isfield(opts, 'u0')
    u = opts.u0;
else
    % Initialisation par la solution harmonique : beaucoup plus proche de
    % l'optimum qu'une valeur constante, d'ou une convergence plus rapide.
    u = inpaint_harmonic(f, mask, 'direct');
end
u(~mask) = f(~mask);
tvHist = zeros(1, nIter);
tol = 1e-4;
switch method
    case 'cp'
        tau = 0.99 / sqrt(8);
        sigma = 0.99 / sqrt(8);
        px = zeros(size(u)); py = zeros(size(u));
        ubar = u;
        r1=inf;
        r2=inf;
        k=0;

        while (r1 > tol || r2 > tol) && k <nIter
            k=k+1 ;
            [gx, gy] = grad_fwd(ubar);
            px = px + sigma * gx;
            py = py + sigma * gy;
            nrm = max(1, sqrt(px.^2 + py.^2));
            px = px ./ nrm;
            py = py ./ nrm;
            uOld = u;
            d = div_bwd(px, py);
            u = u + tau * d;
            u(~mask) = f(~mask);
            ubar = 2*u - uOld;
            TV = sum(sqrt(gx(:).^2 + gy(:).^2));
            tvHist(k) = TV;
            r1 = max(abs(d(mask)));
            r2 = (TV - sum(px(:).*gx(:) + py(:).*gy(:))) / TV;

        end
    case 'gd'
        epsTV = 0.05; % Ce paramettre de lissage nous perme d''eviter de diviser par zero dans les zone uniforme (plate)
        if isfield(opts, 'eps'), epsTV = opts.eps; end
        dt = epsTV / 5;                     % pas stable pour la TV lissee
        for k = 1:nIter
            [gx, gy] = grad_fwd(u);
            nrm = sqrt(gx.^2 + gy.^2 + epsTV^2);
            d = div_bwd(gx ./ nrm, gy ./ nrm);
            u(mask) = u(mask) + dt * d(mask);
            TV = sum(sqrt(gx(:).^2 + gy(:).^2));
            tvHist(k) = TV;
        end
    otherwise
        error('inpaint_tv:method', 'Methode inconnue : %s', method);
end
end

%Inpainting par minimisation de la variation totale

%   [u, tvHist] = INPAINT_TV(f, mask, method, nIter, opts)
%     f      : image (double)
%     mask   : logique, true = pixel manquant
%     method : 'cp'  algorithme primal-dual de Chambolle-Pock (defaut)
%              'gd'  descente de gradient sur la TV lissee sqrt(|grad u|^2 + eps^2)
%     nIter  : nombre d'iterations                     [500]
%     opts   : .eps (lissage pour 'gd', defaut 0.05)
%              .u0  (initialisation, defaut : solution harmonique)
%   u      : image restauree
%   tvHist : variation totale TV(u) = sum |grad u| a chaque iteration
