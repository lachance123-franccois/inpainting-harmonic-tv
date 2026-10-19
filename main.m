%% Inpainting : diffusion harmonique vs variation totale
% Comparaison des 4 methodes, en qualite (PSNR, SSIM) et en vitesse :
%   1) diffusion harmonique, solveur direct (systeme lineaire creux)
%   2) diffusion harmonique, schema explicite (equation de la chaleur)
%   3) variation totale, descente de gradient sur la TV lissee
%   4) variation totale, Chambolle-Pock (primal-dual)
% Aucune toolbox requise.
%
% Auteur : Lachance - github.com/lachance123-franccois

clear; close all; clc;
root = fileparts(mfilename('fullpath'));
if ~exist(fullfile(root, 'src'), 'dir'), root = pwd; end   % lance depuis l'editeur
addpath(fullfile(root, 'src'));
resDir = fullfile(root, 'results');
if ~exist(resDir, 'dir'), mkdir(resDir); end
rng(1);
sz = 256;

%% 0. Les 4 methodes
% Chaque methode renvoie [u, historique]. Le nombre d'iterations effectuees
% est la longueur de l'historique (0 pour la methode directe) :
%   - diffusion explicite et Chambolle-Pock s'arretent sur leur critere de
%     convergence, le nombre maximal d'iterations n'est qu'un garde-fou ;
%   - la descente de gradient fait un nombre fixe d'iterations.
epsTV = 0.05;                                   % lissage de la TV ('gd')
nGD = 3000;                                     % iterations de la descente de gradient
meth = { ...
    'Diffusion directe',   @(f, m) inpaint_harmonic(f, m, 'direct');
    'Diffusion explicite', @(f, m) inpaint_harmonic(f, m, 'explicit', 20000);
    'TV gradient',         @(f, m) inpaint_tv(f, m, 'gd', nGD, struct('eps', epsTV));
    'TV Chambolle-Pock',   @(f, m) inpaint_tv(f, m, 'cp', 5000)};
nM = size(meth, 1);

% Appel de chauffe : le premier appel d'une fonction inclut sa compilation
% (JIT) et fausserait le chronometrage de la premiere methode testee.
w = make_mask('carre', 32, 6); W0 = rand(32);
ws = warning('off', 'all');
for j = 1:nM, [~, ~] = meth{j, 2}(W0, w); end
warning(ws);

%% 1. Rayures et texte : qualite et temps des 4 methodes
I = load_test_image('naturelle', sz);
try
    maskText = make_mask('texte', sz, 'MATLAB');
catch
    maskText = make_mask('rayures', sz, 30);    % si le rendu du texte echoue
end
masks = {make_mask('rayures', sz, 14), maskText};
names = {'rayures', 'texte'};
for m = 1:2
    mask = masks{m};
    f = I; f(mask) = 1;                         % zone masquee affichee en blanc
    U = cell(1, nM);
    P = zeros(nM, 1); S = zeros(nM, 1); T = zeros(nM, 1); K = zeros(nM, 1);
    for j = 1:nM
        [U{j}, K(j), T(j)] = run_timed(meth{j, 2}, I, mask);
        [P(j), S(j)] = quality(U{j}, I, mask);
    end
    msIt = 1000 * T ./ max(K, 1);
    msIt(K == 0) = NaN;                         % methode directe : pas d'iterations
    fprintf('\n[%s] %.1f %% de pixels manquants\n', names{m}, 100*mean(mask(:)));
    disp(table(P, S, T, K, msIt, 'RowNames', meth(:, 1), ...
        'VariableNames', {'PSNR_dB', 'SSIM', 'Temps_s', 'Iterations', 'ms_par_iteration'}));

    fig = figure('Position', [50 50 1200 720]);
    subplot(2, 3, 1); imshow(I); title('Original');
    subplot(2, 3, 2); imshow(f); title(sprintf('Degrade (%s)', names{m}));
    for j = 1:nM
        subplot(2, 3, 2 + j); imshow(U{j});
        title(sprintf('%s : %.1f dB, %.2f s', meth{j, 1}, P(j), T(j)));
    end
    save_fig(fig, fullfile(resDir, sprintf('fig%d_%s.png', m, names{m})));
end

%% 2. Qualite et temps en fonction de la taille du trou
sides = [4 8 12 16 24 32 48];
PS = zeros(numel(sides), nM); TS = zeros(numel(sides), nM); KS = zeros(numel(sides), nM);
for k = 1:numel(sides)
    mask = make_mask('carre', sz, sides(k));
    for j = 1:nM
        [u, KS(k, j), TS(k, j)] = run_timed(meth{j, 2}, I, mask);
        PS(k, j) = quality(u, I, mask);
    end
end
fig = figure('Position', [50 50 1200 450]);
subplot(1, 2, 1);
plot(sides, PS, 'o-', 'LineWidth', 1.5); grid on;
xlabel('Cote du trou carre (pixels)'); ylabel('PSNR dans le trou (dB)');
legend(meth(:, 1), 'Location', 'northeast');
title('Qualite vs taille du trou');
subplot(1, 2, 2);
semilogy(sides, TS, 'o-', 'LineWidth', 1.5); grid on;
xlabel('Cote du trou carre (pixels)'); ylabel('Temps de calcul (s)');
legend(meth(:, 1), 'Location', 'northwest');
title('Temps vs taille du trou');
save_fig(fig, fullfile(resDir, 'fig3_taille_trou.png'));
fprintf('\nPSNR (dB) en fonction du cote du trou :\n');
disp(array2table([sides' PS], 'VariableNames', [{'cote_px'}, matlab.lang.makeValidName(meth(:, 1)')]));
fprintf('Temps (s) en fonction du cote du trou :\n');
disp(array2table([sides' TS], 'VariableNames', [{'cote_px'}, matlab.lang.makeValidName(meth(:, 1)')]));
fprintf('Iterations en fonction du cote du trou :\n');
disp(array2table([sides' KS], 'VariableNames', [{'cote_px'}, matlab.lang.makeValidName(meth(:, 1)')]));

%% 3. Convergence des methodes iteratives (masque rayures)
mask = masks{1};
[~, eH] = meth{2, 2}(I, mask);                  % energie de Dirichlet, schema explicite
uDirect = meth{1, 2}(I, mask);
[gx, gy] = grad_fwd(uDirect); eStar = sum(gx(:).^2 + gy(:).^2);
[~, tvGD] = meth{3, 2}(I, mask);
[~, tvCP] = meth{4, 2}(I, mask);
fig = figure('Position', [50 50 1200 420]);
subplot(1, 2, 1);
semilogy(1:numel(eH), max(eH - eStar, eps), 'LineWidth', 1.5); grid on;
xlabel('Iteration'); ylabel('E(u_k) - E(u^*)');
title('Diffusion explicite : ecart a la solution directe');
subplot(1, 2, 2);
plot(1:numel(tvGD), tvGD, 'LineWidth', 1.5); hold on;
plot(1:numel(tvCP), tvCP, 'LineWidth', 1.5); grid on;
xlabel('Iteration'); ylabel('TV(u_k)');
legend(sprintf('Descente de gradient (eps = %.2f)', epsTV), 'Chambolle-Pock');
title('Variation totale au fil des iterations');
save_fig(fig, fullfile(resDir, 'fig4_convergence.png'));

%% 4. Suppression d'objet sur votre photo (optionnel)
% Placer data/photo.png (ou .jpg). Un masque peut etre fourni (data/masque.png,
% blanc = a supprimer), sinon dessinez un polygone autour de l'objet.
photo = dir(fullfile(root, 'data', 'photo.*'));
if ~isempty(photo)
    Ph = imread(fullfile(photo(1).folder, photo(1).name));
    if isinteger(Ph), Ph = double(Ph) / double(intmax(class(Ph))); end   % 8 ou 16 bits -> [0,1]
    s = 800 / max(size(Ph, 1), size(Ph, 2));     % limite la taille (temps de calcul)
    if s < 1
        [Xq, Yq] = meshgrid(linspace(1, size(Ph, 2), round(s*size(Ph, 2))), ...
                            linspace(1, size(Ph, 1), round(s*size(Ph, 1))));
        P2 = zeros([size(Xq), size(Ph, 3)]);
        for c = 1:size(Ph, 3), P2(:, :, c) = interp2(Ph(:, :, c), Xq, Yq); end
        Ph = P2;
    end
    mFile = fullfile(root, 'data', 'masque.png');
    if exist(mFile, 'file')
        M = double(imread(mFile));
        M = M(:, :, 1);
        [Xq, Yq] = meshgrid(linspace(1, size(M, 2), size(Ph, 2)), linspace(1, size(M, 1), size(Ph, 1)));
        M = interp2(M, Xq, Yq) > 0.5 * max(M(:));  % masque 0/1 ou 0/255
    else
        figure; imshow(Ph); title('Cliquez les sommets autour de l''objet, Entree pour finir');
        [xp, yp] = ginput;
        [Xg, Yg] = meshgrid(1:size(Ph, 2), 1:size(Ph, 1));
        M = inpolygon(Xg, Yg, xp, yp);
    end
    R = Ph;
    for c = 1:size(Ph, 3)
        R(:, :, c) = inpaint_tv(Ph(:, :, c), M, 'cp', 1500);
    end
    fig = figure('Position', [50 50 1100 450]);
    subplot(1, 2, 1); imshow(Ph); title('Photo');
    subplot(1, 2, 2); imshow(R); title('Objet supprime (TV)');
    save_fig(fig, fullfile(resDir, 'fig5_photo.png'));
else
    fprintf('\n(Ajoutez data/photo.png pour tester la suppression d''objet)\n');
end
fprintf('\nFigures enregistrees dans %s\n', resDir);

%% ------------------------------------------------------------------------
function [u, nIt, t] = run_timed(fh, f, mask)
% Execute fh(f, mask) et mesure son temps. Pour les methodes rapides
% (< 0.5 s), le calcul est repete et on garde la mediane, car une mesure
% unique de quelques millisecondes est trop bruitee.
tic; [u, h] = fh(f, mask); t = toc;
if t < 0.5
    nRep = 5; tt = zeros(1, nRep);
    for r = 1:nRep
        tic; [u, h] = fh(f, mask); tt(r) = toc;
    end
    t = median(tt);
end
nIt = numel(h);                                 % longueur de l'historique
end