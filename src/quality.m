function [psnrMask, ssimVal] = quality(u, ref, mask)
%PSNR mesure uniquement dans la zone reconstruite, et SSIM global.
err = u(mask) - ref(mask);
psnrMask = 10 * log10(1 / mean(err.^2));
ssimVal = ssim_simple(u, ref);
end

function s = ssim_simple(x, y)

C1 = 0.01^2; 
C2 = 0.03^2;
t = -5:5;
g = exp(-t.^2/(2 * 1.5^2));
g = g / sum(g);
w = g' * g;
mx = conv2(x, w, 'valid'); 
my = conv2(y, w, 'valid');
sxx = conv2(x.^2, w, 'valid') - mx.^2;
syy = conv2(y.^2, w, 'valid') - my.^2;
sxy = conv2(x.*y, w, 'valid') - mx.*my;
m = ((2*mx.*my + C1) .* (2*sxy + C2)) ./((mx.^2 + my.^2 + C1) .* (sxx + syy + C2));
s = mean(m(:));
end
