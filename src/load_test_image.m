function I = load_test_image(name, sz)
%LOAD_TEST_IMAGE  Image de test en niveaux de gris dans [0,1], taille sz x sz.
%   name : 'naturelle' (cameraman.tif ou peppers.png livres avec MATLAB,
%          sinon image synthetique) ou 'formes' (image constante par morceaux)
if nargin < 2
    sz = 256; 
end
switch name
    case 'naturelle'
        I = [];
        for cand = {'cameraman.tif', 'peppers.png', 'ngc6543a.jpg'}
            try
                I = imread(cand{1});
                break;
            catch
            end
        end
        I = double(I) / 255;
        if size(I, 3) == 3
            I = rgb2gray(I);
        end
        I = resize_square(I, sz);
    otherwise
        error('load_test_image:name', 'Image inconnue : %s', name);
end
end


 % cette fonction n'est utile que dans le cas de la mise a l'epreuve de notre algorithme. d'ou la neccessite d'avoir une image carree  car 
 % fonction make_mask nous fourni un masque carre
 
function I = resize_square(I, sz)
[H, W] = size(I);
s = min(H, W);
I = I(floor((H-s)/2)+1 : floor((H-s)/2)+s, floor((W-s)/2)+1 : floor((W-s)/2)+s);
[Xq, Yq] = meshgrid(linspace(1, s, sz));
I = interp2(I, Xq, Yq, 'linear');
end




