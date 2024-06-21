from nnz.project.project import Project

class ProjectNN(Project):
    def __init__(self, name=None):
        super().__init__(name=name)

class ProjectNNClassifier(ProjectNN):

    def __init__(self, name=None):
        super().__init__(name=name)