import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import math
import cv2

class ImageAugmentation():

    """ Class regroupant différente fonction de traitement d'images dans le but d'une augmentation de données pour les datasets images """

    def __init__(self, **kwargs) -> None:

        __default_options = {
            "symmetry" : None, # "horizontal, vertical, both"
            "saturation" : None,
            "saturation_range" : None,
            "contrast" : None,
            "contrast_range" : None,
            "gamma" : None,
            "gamma_range" : None,
            "brightness" : None,
            "brightness_range" : None,
            "hue" : None,
            "hue_range" : None,
            "quality" : None,
            "quality_range" : None,
            "zoom" : None,
            "zoom_range" : None,
            "rotation" : None,
            "rotation_range" : None,
            "black_pixel_count" : None,
            "black_pixel_count_range" : None,
            "black_pixel_size" : None,
            "black_pixel_size_range" : None,
            "count_augmentation" : 1,
            "actions" : None
        }

        self.default_actions = [
            "symmetry", "saturation", "contrast", "gamma", "brightness", "hue", "quality", "zoom", "rotation", "black_pixel"
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

    def get_value(self, name, default_value):
        """ Permet de retourner une valeur en fonction des options indiquées """

        if self.options[f'{name}'] is not None:
            _factor = self.options[f'{name}']
        elif self.options[f'{name}_range'] is not None:
            _factor = self.get_random_range_value(self.options[f'{name}_range'])
        else:
            _factor = default_value

        return _factor

    def generate(self, image):
        """ Permet de lancer l'augmentation en fonction des options """

        if self.options['actions'] is not None :
            actions = self.options['actions']
        else:
            actions = []
            for i in range(self.options['count_augmentation']):
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

        return img

    def symmetry(self, image, direction="horizontal"):
        """ Permet d'effectuer une symétrie horizontale / verticale"""

        _direction = self.options['symmetry'] if self.options['symmetry'] is not None else direction

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
        f = self.get_value("quality", quality)
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