import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import cv2

class ImageAugmentation():

    """ Class regroupant différente fonction de traitement d'images dans le but d'une augmentation de données pour les datasets images """

    def __init__(self) -> None:
        pass

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

    def symmetry(self, image, direction="horizontal"):
        """ Permet d'effectuer une symétrie horizontale / verticale"""
        if direction == "horizontal":
            return tf.image.flip_left_right(image)
        else:
            return tf.image.flip_up_down(image)

    def grayscale(self, image):
        """ Permet de mettre une image en noir et blanc """
        return tf.image.rgb_to_grayscale(image)

    def saturation(self, image, factor=0.5):
        """ Permet de modifier la saturation de l'image """
        return tf.image.adjust_saturation(image, factor)

    def contrast(self, image, factor=1.5):
        """ Permet de modifier le contraste de l'image """
        return tf.image.adjust_contrast(image, factor)

    def gamma(self, image, factor=1.2):
        """ Permet de modifier les gammas de l'image """
        return tf.image.adjust_gamma(image, factor)

    def brightness(self, image, factor=0.5):
        """ Permet de modifier la luminosité de l'image """
        return tf.image.adjust_brightness(image, factor)

    def hue(self, image, delta=0.5):
        """ Permet de modifier la teinte de l'image """
        return tf.image.adjust_hue(image, delta=delta)

    def quality(self, image, quality=50):
        """ Permet de modifier la qualité de l'image """
        return tf.image.adjust_jpeg_quality(image, jpeg_quality=quality)

    def zoom(self, image, scale=0.1):
        """ Permet de faire un crop de l'image depuis le centre """
        scale = 1 - scale
        return tf.image.central_crop(image, central_fraction=scale)

    def rotate_image(self, image, angle=25):
        """ Permet d'appliquer une rotation à l'image """
        image = np.array(image)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
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

        for i in range(count):
            image = self.create_rectangle(image, size=size)

        return image