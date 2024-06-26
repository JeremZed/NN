import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import math
import cv2

class ImageAugmentation():

    """ Class regroupant différente fonction de traitement d'images dans le but d'une augmentation de données pour les datasets images """

    def __init__(self, **kwargs) -> None:

        self.shuffle__saturation_range = (0.1,5.0)
        self.shuffle__contrast_range = (0.1,5.0)
        self.shuffle__gamma_range = (0.1,5.0)
        self.shuffle__brightness_range = (0.1,0.5)
        self.shuffle__hue_range = (-0.5,0.5)
        self.shuffle__quality_range = (10, 90)
        self.shuffle__zoom_range = (0.01,0.25)
        self.shuffle__rotation_range = (45,125)
        self.shuffle__black_pixel_count_range = (1,5)
        self.shuffle__black_pixel_size_range = (1,5)
        self.shuffle__translate_ratio_range = (3,8)
        self.shuffle__translate_direction_list_range = ['top', 'left', 'right', 'bottom']
        self.shuffle__symmetry_list_range = ['both', 'horizontal', 'vertical']

        __default_options = {
            "count_new_image" : 1,
            "count_augmentation_per_image" : 1,
            "actions" : None,
            "shuffle" : False
        }

        self.default_actions = [
            "symmetry", "saturation", "contrast", "gamma", "brightness", "hue", "quality", "zoom", "rotation", "black_pixel", "translate"
        ]

        # On modifie les options par défaut par ceux indiqués en paramètres
        self.options = __default_options | kwargs


    def visualize(self, original, augmented):
        """ Permet de faire apparâitre un visuel de comparaison entre l'image d'origine et l'image augmentée """

        original = np.array(original)
        augmented = np.array(augmented)

        fig, ax = plt.subplots(figsize=(10, 15), nrows=1, ncols=2)

        ax[0].imshow(original)
        ax[0].set_title(f'Original image ({original.shape})')
        ax[0].set_yticks([])
        ax[0].set_xticks([])

        ax[1].imshow(augmented)
        ax[1].set_title(f'Augmented image ({augmented.shape})')
        ax[1].set_yticks([])
        ax[1].set_xticks([])

        fig.tight_layout()
        plt.show()

    def get_random_range_value(self, range):
        """ Permet de retourner une valeur aléatoire depuis un range passé en paramètre """
        start, end = range
        return round(random.uniform(start, end), 2)

    def get_random_value_list(self, range):
        """ Permet de retourner une valeur aléatoire depuis une liste de valeur """
        return random.choice(range)

    def get_value(self, name, default_value):
        """ Permet de retourner une valeur en fonction des options indiquées """
        attr = None
        is_list = False
        name_attr = f'shuffle__{name}_range'
        try:
            attr = getattr(self, name_attr)
            if 'list' in name_attr:
                is_list = True
        except:
            pass

        name_range = f'{name}_range'

        if name in self.options and self.options[f'{name}'] is not None:
            _factor = self.options[f'{name}']

        elif name_range in self.options and self.options[name_range] is not None:
            if "list" in name_range:
                _factor = self.get_random_value_list(self.options[name_range])
            else:
                _factor = self.get_random_range_value(self.options[name_range])

        elif self.options['shuffle'] is not None and self.options['shuffle'] == True and attr is not None:
            if is_list:
                _factor = self.get_random_value_list(attr)
            else:
                _factor = self.get_random_range_value(attr)
        else:
            _factor = default_value

        return _factor

    def generate(self, image, type=None):
        """ Permet de lancer l'augmentation en fonction des options """

        images = []

        for c in range(self.options['count_new_image']):

            if self.options['actions'] is not None :
                actions = self.options['actions']
            else:
                actions = []
                for i in range(self.options['count_augmentation_per_image']):
                    actions.append(random.choice(self.default_actions))

            img = image
            for action in actions:
                try:
                    func = getattr(self,action)
                    img = func(img)
                except Exception as e:
                    print("Erreur avec l'action : ", action)
                    print(e)
                    pass

            if type == "array":
                img = np.array(img)

            images.append(img)

        return images

    def symmetry(self, image, direction="horizontal"):
        """ Permet d'effectuer une symétrie horizontale / verticale"""

        # _direction = self.options['symmetry'] if 'symmetry' in self.options and self.options['symmetry'] is not None else direction
        _direction = self.get_value("symmetry_list", direction)

        if _direction == "horizontal":
            return tf.image.flip_left_right(image)
        elif _direction == "both":
            img = tf.image.flip_left_right(image)
            return tf.image.flip_up_down(img)
        else:
            return tf.image.flip_up_down(image)

    def grayscale(self, image):
        """ Permet de mettre une image en noir et blanc """
        return tf.image.rgb_to_grayscale(image)

    def saturation(self, image, factor=0.5):
        """ Permet de modifier la saturation de l'image """
        f = self.get_value("saturation", factor)
        return tf.image.adjust_saturation(image, f)

    def contrast(self, image, factor=1.5):
        """ Permet de modifier le contraste de l'image """
        f = self.get_value("contrast", factor)
        return tf.image.adjust_contrast(image, f)

    def gamma(self, image, factor=1.2):
        """ Permet de modifier les gammas de l'image """
        f = self.get_value("gamma", factor)
        return tf.image.adjust_gamma(image, f)

    def brightness(self, image, factor=0.5):
        """ Permet de modifier la luminosité de l'image """
        f = self.get_value("brightness", factor)
        return tf.image.adjust_brightness(image, f)

    def hue(self, image, delta=0.5):
        """ Permet de modifier la teinte de l'image """
        f = self.get_value("hue", delta)
        return tf.image.adjust_hue(image, delta=f)

    def quality(self, image, quality=50):
        """ Permet de modifier la qualité de l'image """
        f = math.ceil(self.get_value("quality", quality))
        return tf.image.adjust_jpeg_quality(image, jpeg_quality=f)

    def zoom(self, image, scale=0.1):
        """ Permet de faire un crop de l'image depuis le centre """

        shape = np.array(image).shape

        f = self.get_value("zoom", scale)
        f = 1 - f
        img  = tf.image.central_crop(image, central_fraction=f)
        return tf.image.resize(img, size=[shape[0], shape[1]], method='nearest')

    def rotation(self, image, angle=25):
        """ Permet d'appliquer une rotation à l'image """
        f = self.get_value("rotation", angle)

        image = np.array(image)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, f, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
        return result

    def get_random_coordinate_rectangle(self, im, size):
        """ Permet de retourner une zone de pixel de façon aléatoire """

        p1_x = random.randint(0, im.shape[0])
        p1_y = random.randint(0, im.shape[1])

        p2_x = p1_x + size
        p2_y = p1_y + size

        if p2_x > im.shape[0]:
            p2_x = im.shape[0]

        if p2_y > im.shape[1]:
            p2_y = im.shape[1]

        return p1_x, p1_y, p2_x, p2_y

    def create_rectangle(self, image, size=3):
        """ Permet de créer un rectangle noir sur une image """
        im = np.array(image)
        p1_x, p1_y, p2_x, p2_y = self.get_random_coordinate_rectangle(im, size)

        start_point = (p1_x, p1_y)
        end_point = (p2_x, p2_y)
        color = (0, 0, 0)
        thickness = -1

        return cv2.rectangle(im, start_point, end_point, color, thickness)

    def black_pixel(self, image, count=1, size=3):
        """ Permet d'ajouter des pixels noirs aléatoirement dans l'image en guise de masquage arbitraire """

        c = self.get_value("black_pixel_count", count)
        s = self.get_value("black_pixel_size", size)

        for i in range( math.ceil(c) ):
            image = self.create_rectangle(image, size= math.ceil(s) )

        return image

    def translate(self, image, ratio=8, direction="right", coord=(0,0)):
        """ Permet de faire une translation de l'image par défaut un huitième de deplacement sur la droite """
        img = np.array(image)
        h, w = img.shape[:2]

        r = math.ceil(self.get_value("translate_ratio", ratio))
        d = self.get_value("translate_direction_list", direction)

        ratio_height, ratio_width = h / r, w / r

        # Déplacement haut gauche
        if d == "top left":
            T = np.float32([[1, 0, -ratio_width], [0, 1, -ratio_height]])

        # Déplacement bas droite
        elif d == "bottom right":
            T = np.float32([[1, 0, ratio_width], [0, 1, ratio_height]])

        # Déplacement bas gauche
        elif d == "bottom left":
            T = np.float32([[1, 0, -ratio_width], [0, 1, ratio_height]])

        # Déplacement haut droite
        elif d == "top right":
            T = np.float32([[1, 0, ratio_width], [0, 1, -ratio_height]])

        # Déplacement haut
        elif d == "top":
            T = np.float32([[1, 0, 0], [0, 1, -ratio_height]])

        # Déplacement Bas
        elif d == "bottom":
            T = np.float32([[1, 0, 0], [0, 1, ratio_height]])

        # Déplacement Droite
        elif d == "right":
            T = np.float32([[1, 0, ratio_width], [0, 1, 0]])

        # Déplacement Gauche
        elif d == "left":
            T = np.float32([[1, 0, -ratio_width], [0, 1, 0]])

        # Déplacement personnalisé
        elif d == "custom":

            if self.options["translate_custom_coord"] is not None:
                coord = self.options["translate_custom_coord"]
            elif self.options["translate_custom_coord_range"] is not None:
                x = self.get_random_range_value(self.options["translate_custom_coord_range"][0])
                y = self.get_random_range_value(self.options["translate_custom_coord_range"][1])
                coord = (math.ceil(x), math.ceil(y))

            T = np.float32([[1, 0, coord[0]], [0, 1, coord[1]]])

        # Déplacement par défaut vers la droite
        else:
            T = np.float32([[1, 0, ratio_width], [0, 1, 0]])

        return cv2.warpAffine(img, T, (w, h))
