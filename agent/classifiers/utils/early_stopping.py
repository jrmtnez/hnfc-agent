import copy

class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.best_model = None

    def stop_training(self, measure, model):
        if self.best is None:
            self.best = measure
            self.best_model = copy.deepcopy(model)
            return False

        if self.best < measure:
            self.num_bad_epochs = 0
            self.best = measure
            self.best_model = copy.deepcopy(model)
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def get_best_model(self):
        return self.best_model
