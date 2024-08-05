import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

''' 
Crée un masque Gaussien 2D centré dans la fenêtre de la taille spécifiée.
- size : taille du masque.
- sigma : écart-type de la distribution gaussienne (contrôle la largeur de la cloche).
'''


def normalized_gaussian_2d_mask(size, sigma):
    x, y = np.meshgrid(np.arange(-(size // 2), size // 2 + 1),
                       np.arange(-(size // 2), size // 2 + 1))
    # Calcule la valeur de la distribution gaussienne pour chaque point de la grille.
    g = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    # On normalise pour que la somme de toutes les valeurs du masque = 1.
    return g / np.sum(g)


''' 
Calcule la somme des différences au carré pondérées entre deux patches d'images en utilisant un masque de poids.
- patch1 : patch de l'image 1.
- patch2 : patch de l'image 2.
- weight_mask : masque de poids.
'''


def weighted_ssd(patch1, patch2, weight_mask):
    # Mesure la différence entre les pixels des deux patches.
    squared_differences = (patch1 - patch2) ** 2
    # En divisant par 3, on obtient la moyenne des carrés des différences pour chaque pixel dans les canaux de couleur?
    weighted_ssd_result = np.sum(
        (np.sum(squared_differences, axis=2) / 3) * weight_mask)  # Mesure pondérée de la différence entre les patches.
    # On normalise.
    return weighted_ssd_result / np.sum(weight_mask)


'''
Extrait un patch rectangulaire d'une image à partir d'un centre donné. 
- image : image à partir de laquelle extraire le patch.
- center : coordonnées du centre du patch.
- default_value : valeur par défaut à utiliser pour les pixels en dehors de l'image.
- window_size : taille du patch.
'''


def get_patch_from_image(image, center, default_value, window_size):

    # Coordonnée x du coin supérieur gauche.
    x_start = center[0] - window_size//2
    # Coordonnée x du coin inférieur droit.
    x_end = min(window_size, image.shape[0] - x_start)

    # Coordonnée y du coin supérieur gauche.
    y_start = center[1] - window_size//2
    # Coordonnée y du coin inférieur droit.
    y_end = min(window_size, image.shape[1] - y_start)

    # Pour s'assurer que le patch ne dépasse pas de l'image.
    x_start_offset = max(0, -x_start)
    y_start_offset = max(0, -y_start)

    if image.ndim == 3:  # Si l'image est en couleur.
        # image.shape[2] = nombre de canaux de couleur.
        patch = np.full((window_size, window_size, image.shape[2]), default_value,
                        dtype=image.dtype)
    else:  # Si l'image est en noir et blanc.
        patch = np.full((window_size, window_size), default_value,
                        dtype=image.dtype)

    # On copie les pixels de l'image dans le patch.
    patch[x_start_offset:x_end, y_start_offset:y_end] = image[x_start + x_start_offset:x_start + x_end,
                                                              y_start + y_start_offset:y_start + y_end]
    return patch


'''
TextureGenerator permet de générer une texture synthétique.
'''


class TextureGenerator:

    '''
    Initialise les attributs, les paramètres de fenêtre et les seuils d'erreur spécifiés. Elle prépare l'état initial de l'objet pour le processus itératif de synthèse de texture.
    - example_path : chemin vers l'image d'exemple.
    - window_size : taille du patch.
    - output_size : taille de la texture synthétique à générer.
    - sigma : écart-type de la distribution gaussienne (contrôle la largeur de la cloche).
    - error_threshold : seuil d'erreur.
    '''

    def __init__(self, example_path, window_size, output_size, sigma, error_threshold):
        # Tailles.
        self._window_size = window_size
        self.patch_offset = window_size // 2
        self._output_size = output_size

        # Chargement de l'image.
        self.example_image = np.array(Image.open(example_path))
        self.example_height, self.example_width, _ = self.example_image.shape
        # Calcule la couleur moyenne de l'image pour chaque canal.
        self.mean_color = np.mean(self.example_image, axis=(0, 1))
        # Crée un tableau pour stocker les patches de l'image.
        self.example_image_patches = (
            np.empty((self.example_height - 2 * self.patch_offset,
                      self.example_width - 2 * self.patch_offset,
                      window_size, window_size, 3))
        )
        # On extrait les patches de l'image.
        for i in range(self.example_image_patches.shape[0]):
            for j in range(self.example_image_patches.shape[1]):
                # On extrait le patch centré en (i,j).
                self.example_image_patches[i][j] = get_patch_from_image(image=self.example_image,
                                                                        center=(
                                                                            i, j),
                                                                        default_value=self.mean_color,
                                                                        window_size=self._window_size)

        # Initialisation de l'image synthétique, création d'un tableau 3D initialisé à zéro.
        self.synthesized_texture = np.zeros(
            (output_size[0], output_size[1], 3))
        # On initialise le pixel central de l'image synthétique à la couleur moyenne de l'image d'exemple.
        self.synthesized_texture[output_size[0] // 2,
                                 output_size[1] // 2, :] = self.mean_color
        # Pour suivre les pixels déjà remplis dans l'image synthétisée.
        self.synthesized_texture_filled_mask = np.zeros(
            (output_size[0], output_size[1]), dtype=bool)
        # On marque le pixel central comme rempli.
        self.synthesized_texture_filled_mask[output_size[0] //
                                             2][output_size[1] // 2] = True
        # Stocke temporairement l'état de l'image synthétisée lors de l'itération précédente.
        self.synthesized_texture_filled_mask_cache = np.zeros(
            (output_size[0], output_size[1]), dtype=bool)

        # Crée un masque gaussien 2D centré dans la fenêtre de la taille spécifiée.
        self.gaussian_mask = normalized_gaussian_2d_mask(window_size, sigma)
        self._error_threshold = error_threshold

    '''
    Obtient les coordonnées des pixels non remplis dans l'image synthétisée qui ont au moins un voisin généré.
    Précondition : au moins un pixel généré.
    '''

    def get_unfilled_neighbors(self):
        # Les pixels voisins des pixels déjà remplis sont marqués comme pouvant être remplis dans l'itération suivante.
        dilated_filled_mask = binary_dilation(
            self.synthesized_texture_filled_mask)

        # Identifie les pixels non remplis qui ont au moins un voisin déjà rempli.
        unfilled_with_generated_neighbors_mask = dilated_filled_mask ^ self.synthesized_texture_filled_mask

        # On obtient les coordonnées des pixels non remplis dans l'image synthétisée qui ont au moins un voisin généré.
        coordinates_to_fill = np.argwhere(
            unfilled_with_generated_neighbors_mask)

        # Tri par ordre décroissant du nombre de voisins générés.
        coordinates_to_fill = \
            sorted(coordinates_to_fill,
                   key=lambda coord: np.sum(
                       self.synthesized_texture_filled_mask[coord[0]-1:coord[0]+2, coord[1]-1:coord[1]+2]),
                   reverse=True)

        # Conserve l'état des pixels remplis dans l'itération actuelle, permettant de comparer les voisins remplis dans les itérations successives.
        self.synthesized_texture_filled_mask_cache = dilated_filled_mask
        return coordinates_to_fill

    '''
    Trouve le meilleur candidat de pixel dans l'image d'exemple pour un patch donné de la texture synthétisée.
    - pixel_coords : coordonnées du pixel à synthétiser.
    '''

    def synthesize_pixel(self, pixel_coords):
        # Extrait un patch du masque de pixels remplis centré sur les coordonnées pixel_coords.
        patch_filled_mask = get_patch_from_image(image=self.synthesized_texture_filled_mask,
                                                 center=pixel_coords,
                                                 default_value=False,
                                                 window_size=self._window_size)
        # Extrait un patch de l'image synthétisée centré sur les coordonnées pixel_coords.
        synthetic_texture_patch = get_patch_from_image(image=self.synthesized_texture,
                                                       center=pixel_coords,
                                                       default_value=0,
                                                       window_size=self._window_size)
        # On pondère le masque de pixels remplis avec le masque gaussien.
        weight_mask = patch_filled_mask * self.gaussian_mask

        # On calcule la somme des différences au carré pondérées entre le patch synthétisé et les patches de l'image d'exemple.
        matches = []
        for i in range(self.example_image_patches.shape[0]):
            for j in range(self.example_image_patches.shape[1]):
                # On extrait le patch de l'image d'exemple centré en (i,j).
                example_patch = self.example_image_patches[i][j]
                # On calcule la somme des différences au carré pondérées entre le patch synthétisé et le patch de l'image d'exemple.
                current_ssd = weighted_ssd(patch1=synthetic_texture_patch,
                                           patch2=example_patch,
                                           weight_mask=weight_mask)
                # On ajoute le résultat à la liste des candidats.
                matches.append((current_ssd, self.example_image[i][j]))
        # On trie les candidats par ordre croissant de différence.
        matches = sorted(matches, key=lambda x: x[0])

        # Gestion de l'erreur
        # accepted_error = matches[0][0] * (1 + self._error_threshold)
        # best_matches_color = [match[1] for match in matches if match[0] <= accepted_error]
        self.synthesized_texture[pixel_coords[0]][pixel_coords[1]] = matches[0][1]  # on prend le meilleur candidat
        # np.mean(best_matches_color, axis=0) / len(best_matches_color)

    '''
    Génère une texture synthétique à partir de l'image d'exemple.
    '''

    def synthesize_texture(self):
        # On obtient les coordonnées des pixels non remplis dans l'image synthétisée qui ont au moins un voisin généré.
        pixels_to_synthesize = self.get_unfilled_neighbors()
        # Tant qu'il reste des pixels à synthétiser.
        while np.any(pixels_to_synthesize):
            for pix in pixels_to_synthesize:
                # On synthétise le pixel.
                self.synthesize_pixel(pix)
            # On met à jour l'état de l'image synthétisée.
            self.synthesized_texture_filled_mask = self.synthesized_texture_filled_mask_cache
            # On obtient les coordonnées des pixels non remplis dans l'image synthétisée qui ont au moins un voisin généré.
            pixels_to_synthesize = self.get_unfilled_neighbors()
        # On retourne la texture synthétique normalisée.
        return self.synthesized_texture / 255


if __name__ == "__main__":
    # Génère une texture synthétique à partir d'une image d'exemple.
    print("generating a 32x32 texture from textures_data/screen03.jpg")
    text0generator = TextureGenerator(example_path="textures_data/screen03.jpg",
                                      window_size=15,
                                      output_size=(32, 32),
                                      sigma=15/6,
                                      error_threshold=100)
    synthetic_texture = text0generator.synthesize_texture()

    # Convertis les valeurs des pixels de la texture synthétique de [0,1] à [0,255].
    synthetic_texture = (synthetic_texture * 255).astype(np.uint8)

    # Spécifier le chemin du fichier que l'on veut sauvegarder en TIF.
    output_path = "./synthetic_patch.tif"

    # Convertis un tableau contenant des données d'image en un objet d'image.
    synthetic_texture_pil = Image.fromarray(synthetic_texture)

    # Sauvegarde la texture synthétique comme une image TIF.
    synthetic_texture_pil.save(output_path)
    print("generated texture as "+output_path)
