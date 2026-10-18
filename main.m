%% Inpainting : diffusion harmonique vs variation totale
% Auteur : Lachance - github.com/lachance123-franccois
clear; close all; clc;
root = fileparts(mfilename('fullpath'));
if ~exist(fullfile(root, 'src'), 'dir'), root = pwd; end      % <- si lancé depuis l'éditeur
addpath(fullfile(root, 'src'));
resDir = fullfile(root, 'results');
if ~exist(resDir, 'dir'), mkdir(resDir); end
rng(1);
sz = 256;

%% 1. Rayures et texte sur une image naturelle
I = load_test_image('naturelle', sz);
try
    maskText = make_mask('texte', sz, 'MATLAB');
catch
    maskText = make_mask('rayures', sz, 30);
end
masks = {make_mask('rayures', sz, 14), maskText};
names = {'rayures', 'texte'};
for m = 1:2
    mask = masks{m};
    f = I; f(mask) = 1;                                       % zone masquée affichée en blanc
    tic; uH = inpaint_harmonic(I, mask, 'direct'); tH = toc;  % <- tic/toc corrects
    tic; uT = inpaint_tv(I, mask, 'cp', 800);      tT = toc;  % <- uT = image, pas tvHist
    [pH, sH] = quality(uH, I, mask);
    [pT, sT] = quality(uT, I, mask);
    fprintf('\n[%s] %.1f %% de pixels manquants\n', names{m}, 100*mean(mask(:)));
    fprintf('  Diffusion (directe) : PSNR(trou) %.2f dB, SSIM %.4f, %.2f s\n', pH, sH, tH);
    fprintf('  TV (Chambolle-Pock) : PSNR(trou) %.2f dB, SSIM %.4f, %.2f s\n', pT, sT, tT);
    fig = figure('Position', [50 50 1300 380]);
    subplot(1, 4, 1); imshow(I);  title('Original');
    subplot(1, 4, 2); imshow(f);  title(sprintf('Degrade (%s)', names{m}));
    subplot(1, 4, 3); imshow(uH); title(sprintf('Diffusion : %.1f dB', pH));
    subplot(1, 4, 4); imshow(uT); title(sprintf('TV : %.1f dB', pT));
    save_fig(fig, fullfile(resDir, sprintf('fig%d_%s.png', m, names{m})));
end

%% 2. PSNR en fonction de la taille du trou
sides = [4 8 12 16 24 32 48];
pH = zeros(size(sides)); 
pT = zeros(size(sides));
for k = 1:numel(sides)
    mask = make_mask('carre', sz, sides(k));
    pH(k) = quality(inpaint_harmonic(I, mask, 'direct'), I, mask);
    pT(k) = quality(inpaint_tv(I, mask, 'cp', 1500), I, mask);
end
fig = figure('Position', [50 50 700 420]);
plot(sides, pH, 'o-', 'LineWidth', 1.5); 
hold on;
plot(sides, pT, 's-', 'LineWidth', 1.5); 
grid on;
xlabel('Cote du trou carre (pixels)'); 
ylabel('PSNR dans le trou (dB)');
legend('Diffusion harmonique', 'Variation totale');
title('Qualite de reconstruction vs taille du trou');
save_fig(fig, fullfile(resDir, 'fig4_psnr_vs_taille.png'));
disp(table(sides', pH', pT', 'VariableNames', {'cote_px', 'PSNR_diffusion', 'PSNR_TV'}));

%% 3. Convergence des algorithmes
mask = masks{1};
nIt = 1500;
[~, eH] = inpaint_harmonic(I, mask, 'explicit', nIt);
uDirect = inpaint_harmonic(I, mask, 'direct');
[gx, gy] = grad_fwd(uDirect); eStar = sum(gx(:).^2 + gy(:).^2);
[~, tvGD] = inpaint_tv(I, mask, 'gd', nIt, struct('eps', 0.02));
[~, tvCP] = inpaint_tv(I, mask, 'cp', nIt);
fig = figure('Position', [50 50 1100 400]);
subplot(1, 2, 1);
semilogy(1:numel(eH), max(eH - eStar, eps), 'LineWidth', 1.5); 
grid on;   % <- longueur réelle
xlabel('Iteration'); ylabel('E(u_k) - E(u^*)');
title('Diffusion explicite : ecart a la solution directe');
subplot(1, 2, 2);
plot(1:numel(tvGD), tvGD, 'LineWidth', 1.5); 
hold on;                    % <- longueur réelle
plot(1:numel(tvCP), tvCP, 'LineWidth', 1.5); 
grid on;                    
xlabel('Iteration'); 
ylabel('TV(u_k)');
legend('Descente de gradient (TV lissee)', 'Chambolle-Pock');
title('Variation totale au fil des iterations');
save_fig(fig, fullfile(resDir, 'fig5_convergence.png'));