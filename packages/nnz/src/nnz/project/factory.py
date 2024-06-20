import nnz

from nnz.project.data import ProjectData
from nnz.project.ml import ProjectMLClassifier, ProjectMLPrediction
from nnz.project.nn import ProjectNNClassifier
from nnz.project.yolo import ProjectYoloClassifier, ProjectYoloDetection

class ProjectFactory:

    def __init__(self, type):
        self.instance = None

        if type == nnz.PROJECT_TYPE_DATA:
            self.instance = ProjectData()

        elif type == nnz.PROJECT_TYPE_ML_PREDICTION:
            self.instance = ProjectMLPrediction()

        elif type == nnz.PROJECT_TYPE_ML_CLASSIFIER:
            self.instance = ProjectMLClassifier()

        elif type == nnz.PROJECT_TYPE_NN_CLASSIFIER:
            self.instance = ProjectNNClassifier()

        elif type == nnz.PROJECT_TYPE_YOLO_CLASSIFIER:
            self.instance = ProjectYoloClassifier()

        elif type == nnz.PROJECT_TYPE_YOLO_DETECTION:
            self.instance = ProjectYoloDetection()
