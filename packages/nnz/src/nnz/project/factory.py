import nnz

from nnz.project.data import ProjectData
from nnz.project.ml import ProjectMLClassifier, ProjectMLPrediction
from nnz.project.nn import ProjectNNClassifier
from nnz.project.yolo import ProjectYoloClassifier, ProjectYoloDetection

class ProjectFactory:

    def __init__(self, name=None, type=None, **kwargs):
        self.instance = None

        if type == nnz.PROJECT_TYPE_DATA:
            self.instance = ProjectData(name=name)

        elif type == nnz.PROJECT_TYPE_ML_PREDICTION:
            self.instance = ProjectMLPrediction(name=name, **kwargs)

        elif type == nnz.PROJECT_TYPE_ML_CLASSIFIER:
            self.instance = ProjectMLClassifier(name=name)

        elif type == nnz.PROJECT_TYPE_NN_CLASSIFIER:
            self.instance = ProjectNNClassifier(name=name)

        elif type == nnz.PROJECT_TYPE_YOLO_CLASSIFIER:
            self.instance = ProjectYoloClassifier(name=name)

        elif type == nnz.PROJECT_TYPE_YOLO_DETECTION:
            self.instance = ProjectYoloDetection(name=name)
