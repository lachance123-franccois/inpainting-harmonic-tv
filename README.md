# Inpainting : diffusion harmonique vs variation totale

Reconstruction des pixels manquants d'une image (rayures, texte incrusté, objet à supprimer) par deux modèles variationnels : la **diffusion harmonique** et la **variation totale (TV)**. Chaque modèle est résolu par deux algorithmes, codés à la main en MATLAB **sans toolbox** :

| Modèle | Algorithmes |
|---|---|
| Diffusion harmonique | Système linéaire creux (solveur direct) ; schéma explicite |
| Variation totale | Descente de gradient sur une TV lissée ; Chambolle-Pock |

![Rayures](results/fig1_rayures.png)

## 1. Problème

**Notations.**
- $`f`$ : l'image observée, de taille $`H \times W`$, à valeurs dans $`[0,1]`$.
- $`\Omega`$ : l'ensemble des pixels manquants, donné par le masque `mask`.
- $`u`$ : l'image reconstruite.

**Contrainte.** Hors du trou, on garde les pixels observés : $`u = f`$ sur $`\Omega^c`$.

**Problème.** On cherche l'image la plus régulière qui respecte cette contrainte :

```math
\min_{u}\; R(u) \quad \text{sous} \quad u = f \text{ sur } \Omega^c
```

Le masque est une **donnée** du problème. Dans les expériences, on part d'une image intacte et on fabrique le masque, ce qui permet de mesurer l'erreur par rapport à la vérité terrain.

**Opérateurs discrets.**

- **Gradient** (`grad_fwd`), par différences avant, avec une condition de Neumann (différence nulle sur la dernière colonne et la dernière ligne) :

  
  $`(\nabla u)_{ij} = \big(u_{i,j+1}-u_{ij},\; u_{i+1,j}-u_{ij}\big)`$
  

- **Divergence** (`div_bwd`) : elle est construite comme l'adjoint exact de $`-\nabla`$, c'est-à-dire $`\langle \nabla u, p\rangle = -\langle u, \mathrm{div}\,p\rangle`$ pour tous $`u`$ et $`p`$.

- **Norme du gradient** : $`\|\nabla\|^2 \le 8`$. En effet, $`(a-b)^2 \le 2a^2+2b^2`$, et chaque pixel intervient dans au plus 2 différences par direction. La borne est presque atteinte par un damier $`(-1)^{i+j}`$.

## 2. Diffusion harmonique

### Modèle

```math
R(u) = \sum_{ij} |\nabla u_{ij}|^2
```

La condition d'optimalité est $`\Delta u = 0`$ dans $`\Omega`$ : **chaque pixel inconnu vaut la moyenne de ses voisins**. Au bord de l'image, on fait la moyenne sur les voisins existants seulement (Neumann).

### Méthode directe

Les $`n`$ inconnues sont numérotées par `id(mask) = 1:n`. L'équation du pixel $`p_k`$ est :

```math
\deg(p_k)\,x_k - \sum_{\substack{q \in N(p_k) \\ q \in \Omega}} x_{\text{id}(q)} = \sum_{\substack{q \in N(p_k) \\ q \notin \Omega}} f_q
```

où $`N(p_k)`$ est l'ensemble des voisins de $`p_k`$ dans l'image, et $`\deg(p_k) = |N(p_k)| \in \{2,3,4\}`$.

- **Matrice $`A`$** : $`A_{kk} = \deg(p_k)`$, et $`A_{kl} = -1`$ si $`p_l`$ est un voisin inconnu de $`p_k`$.
- **Second membre $`b`$** : $`b_k`$ est la somme des voisins connus de $`p_k`$.

$`A`$ est symétrique définie positive dès que chaque composante du trou touche un pixel connu. Le code la construit sans boucle sur les pixels : 4 décalages, `sub2ind`, `accumarray`, puis `sparse`. Le système est résolu par `A \ b`.

### Méthode explicite (équation de la chaleur)

```math
u^{k+1} = u^k + dt\,\Delta u^k \quad \text{dans } \Omega
```

- **Laplacien** : il est calculé avec 4 images décalées dont les bords sont répliqués (Neumann).
- **Lien avec la méthode directe** : sur le trou, `lap` vaut exactement $`b - Ax`$. L'itération s'écrit donc $`x^{k+1} = x^k + dt\,(b - Ax^k)`$.
- **Convergence** : l'erreur vérifie $`e^{k+1} = (I - dt\,A)\,e^k`$. Comme $`\lambda(A) \in \,]0, 8]`$, il y a convergence pour $`dt < 1/4`$. Le code prend $`dt = 0{,}24`$.
- **Arrêt** : la boucle s'arrête quand $`\max_\Omega |b - Ax| < 10^{-4}`$, avec `nIter` comme garde-fou.
- **Coût** : pour un trou de côté $`L`$, $`\lambda_{\min} \approx 2\pi^2/L^2`$. Il faut donc environ $`1{,}5\,L^2`$ itérations pour diviser l'erreur par 1000. La méthode explicite devient inutilisable pour les grands trous.
- **Suivi de l'énergie** : à chaque itération, on enregistre $`E(u^k) = \sum|\nabla u^k|^2`$. Comme $`E(u) - E(u^*) = (x - x^*)^\top A\,(x - x^*)`$, cette quantité mesure l'erreur par rapport à la solution directe (figure 5).

## 3. Variation totale

### Modèle

```math
TV(u) = \sum_{ij} |\nabla u_{ij}| = \sum_{ij}\sqrt{(u_x)_{ij}^2 + (u_y)_{ij}^2}
```

**Pourquoi la TV préserve les contours.** Prenons un saut de hauteur $`h`$ dans un trou, étalé sur $`n`$ pixels :

| Critère | Coût du saut | Conséquence |
|---|---|---|
| Harmonique | $`h^2/n`$ | Diminue quand on étale : la solution floute le contour. |
| TV | $`h`$ | Indépendant de $`n`$ : un bord franc ne coûte pas plus qu'une rampe. |

**Interprétation géométrique.** La TV est égale à la somme des longueurs des lignes de niveau (formule de la coaire). À l'optimum, $`\mathrm{div}(\nabla u/|\nabla u|) = 0`$ dans $`\Omega`$, ce qui signifie que la courbure des lignes de niveau est nulle. Les contours sont donc **prolongés en ligne droite**.

### Descente de gradient sur la TV lissée (`'gd'`)

$`|\nabla u|`$ n'est pas dérivable en 0. On le remplace par une version lissée :

```math
J_\varepsilon(u) = \sum \sqrt{|\nabla u|^2 + \varepsilon^2}
```

**Gradient.** Par la règle de dérivation en chaîne, et avec $`\nabla^\top = -\mathrm{div}`$ :

```math
\frac{\partial J_\varepsilon}{\partial u} = -\mathrm{div}\!\left(\frac{\nabla u}{\sqrt{|\nabla u|^2+\varepsilon^2}}\right)
```

**Itération**, appliquée dans $`\Omega`$ seulement :

```math
u \leftarrow u + dt\;\mathrm{div}\!\left(\frac{\nabla u}{\sqrt{|\nabla u|^2+\varepsilon^2}}\right)
```

**Pas de temps.** La courbure de $`\sqrt{x^2+\varepsilon^2}`$ est au plus $`1/\varepsilon`$. La constante de Lipschitz du gradient vérifie donc $`L \le 8/\varepsilon`$, et la stabilité impose $`dt < 2/L = \varepsilon/4`$. Le code prend $`dt = \varepsilon/5`$.

**Rôle de $`\varepsilon`$** : c'est un seuil de contraste.
- Si $`|\nabla u| \ll \varepsilon`$, l'itération est approximativement $`u \leftarrow u + \tfrac15\Delta u`$ : on retrouve la diffusion harmonique.
- Si $`|\nabla u| \gg \varepsilon`$, on retrouve la vraie TV.

Diminuer $`\varepsilon`$ rapproche le résultat de la vraie TV, mais réduit le pas d'autant, donc demande plus d'itérations.

### Chambolle-Pock (`'cp'`)

**Reformulation duale.** Comme $`|g| = \max_{|p|\le1} p\cdot g`$, le problème devient un problème min-max :

```math
\min_{u \in C}\;\max_{|p_{ij}|\le 1}\;\langle \nabla u,\, p\rangle, \qquad C = \{u : u = f \text{ sur } \Omega^c\}
```

**Itération :**

```
p    ← proj_{|p|≤1}( p + σ ∇ū )          montée duale, puis projection p / max(1,|p|)
u⁺   ← proj_C( u + τ div p )              descente primale, puis remise de f hors du trou
ū    ← 2u⁺ − u                            extrapolation
```

- **Convergence** : elle est garantie si $`\sigma\tau\|\nabla\|^2 < 1`$. Le code prend $`\sigma = \tau = 0{,}99/\sqrt8`$.
- **Initialisation** : par la solution harmonique.
- **Conditions d'optimalité et critères d'arrêt.** L'arrêt intervient quand ces deux conditions sont satisfaites à $`10^{-4}`$ près :

| Condition | Résidu calculé | Rôle |
|---|---|---|
| $`\mathrm{div}\,p = 0`$ dans $`\Omega`$ | $`r_1 = \max_\Omega \|\mathrm{div}\,p\|`$, égal à $`\max\|u^+ - u\|/\tau`$ | $`u`$ est stationnaire. |
| $`p = \nabla u/\|\nabla u\|`$ là où $`\nabla u \ne 0`$ | $`r_2 = \big(TV(u) - \langle p, \nabla u\rangle\big)/TV(u) \ge 0`$ | $`p`$ est cohérent avec $`u`$. |

$`r_1`$ seul ne suffit pas, car $`p = 0`$ le vérifie pour n'importe quelle image.

## 4. Mesures de qualité

**PSNR**, mesuré **uniquement sur $`\Omega`$** :

```math
\text{PSNR} = 10\log_{10}\frac{1}{\frac{1}{|\Omega|}\sum_{\Omega}(u - \text{ref})^2}
```

Hors du trou, l'erreur est nulle par construction : l'inclure gonflerait le score.

**SSIM** (Wang et al., 2004). On utilise une fenêtre gaussienne de 11 × 11 et d'écart-type 1,5, avec $`C_1 = 0{,}01^2`$ et $`C_2 = 0{,}03^2`$ :

```math
\text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
```

Elle est moyennée sur toute l'image. **Limite** : la SSIM globale est optimiste pour les petits trous, car les fenêtres entièrement hors du masque valent 1.

## 5. Expériences et résultats

| Figure | Contenu |
|---|---|
| `fig1_rayures.png` | 14 rayures aléatoires (demi-épaisseur 1 à 3 px) : original, dégradé, diffusion, TV |
| `fig2_texte.png` | Texte « MATLAB » incrusté |
| `fig4_psnr_vs_taille.png` | PSNR dans le trou en fonction du côté d'un trou carré (4 à 48 px) |
| `fig5_convergence.png` | À gauche, $`E(u_k) - E(u^*)`$ pour la diffusion explicite (échelle log). À droite, $`TV(u_k)`$ pour la descente de gradient ($`\varepsilon = 0{,}02`$) et pour Chambolle-Pock. |
| `fig6_photo.png` | Suppression d'un objet sur une photo personnelle (optionnel) |

**Résultats** (image 256 × 256, `rng(1)`), à compléter avec la sortie de `main` :

| Masque | Pixels manquants | PSNR diffusion | PSNR TV | SSIM diffusion | SSIM TV | Temps diffusion | Temps TV |
|---|---|---|---|---|---|---|---|
| Rayures | … % | … dB | … dB | … | … | … s | … s |
| Texte | … % | … dB | … dB | … | … | … s | … s |

| Côté du trou (px) | 4 | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| PSNR diffusion (dB) | … | … | … | … | … | … | … |
| PSNR TV (dB) | … | … | … | … | … | … | … |

**Comportements attendus :**
- **Trous fins** (rayures, texte) : la TV restitue des contours nets là où la diffusion floute.
- **Trous larges** : les deux méthodes se dégradent. La TV produit des aplats, et peut couper un contour plutôt que de le prolonger sur une grande distance.
- **Textures** : aucune des deux méthodes ne les reconstruit.

## 6. Lancer le projet

```matlab
>> cd chemin/vers/inpainting-harmonic-tv
>> main
```

- MATLAB R2021a ou plus récent, sans aucune toolbox.
- **Image de test** : `cameraman.tif` ou `peppers.png` si elles sont disponibles, sinon une image synthétique. L'image est recadrée au centre puis rééchantillonnée en 256 × 256.
- **Masque texte** : il est obtenu en dessinant le texte dans une figure invisible, puis en la capturant avec `getframe`. Sans affichage graphique, le script utilise des rayures à la place.
- **Suppression d'objet** : placez `data/photo.png` ou `.jpg`. Ajoutez éventuellement `data/masque.png` (blanc = à supprimer, accepté en 0/1 ou 0/255) ; sinon, cliquez un polygone autour de l'objet. La photo est réduite à 800 px au maximum, puis chaque canal couleur est reconstruit par TV.

Utilisation directe des fonctions :

```matlab
u = inpaint_harmonic(f, mask, 'direct');
[u, E]  = inpaint_harmonic(f, mask, 'explicit', 5000);    % E : énergie à chaque itération
[u, tv] = inpaint_tv(f, mask, 'cp', 2000);                % arrêt sur r1, r2 < 1e-4
[u, tv] = inpaint_tv(f, mask, 'gd', 2000, struct('eps', 0.02));
```

## 7. Structure

```
main.m                    toutes les expériences et figures
src/grad_fwd.m            gradient discret (différences avant, Neumann)
src/div_bwd.m             divergence discrète, adjointe de -grad_fwd
src/inpaint_harmonic.m    diffusion : solveur direct creux, schéma explicite avec arrêt sur le résidu
src/inpaint_tv.m          TV : descente de gradient lissée, Chambolle-Pock avec arrêt sur r1 et r2
src/make_mask.m           masques : rayures, texte, trou carré, pertes aléatoires
src/quality.m             PSNR dans le trou, SSIM
src/load_test_image.m     image de test (recadrage carré, rééchantillonnage)
src/save_fig.m            export des figures
```

## 8. Limites et pistes

- **Textures** : utiliser l'inpainting par patchs (Criminisi, PatchMatch).
- **Effet d'aplats de la TV** : utiliser la TV généralisée (TGV).
- **Trous larges** : utiliser des modèles d'ordre supérieur, qui prolongent aussi la courbure des contours (Euler elastica, CDD).
- **Couleur** : remplacer le traitement canal par canal par une TV vectorielle couplant les canaux.

## Lien avec mes autres projets

Même cadre que la [déconvolution variationnelle](https://github.com/lachance123-franccois/deconvolution-variational-to-unrolled) : un terme d'attache aux données (ici une contrainte dure), un terme de régularisation (Tikhonov ou TV) et un algorithme d'optimisation proximale.

## Licence

MIT