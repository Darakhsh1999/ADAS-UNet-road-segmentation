import copy

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_model = None
        self.best_score = None
        self.counter = 0

    def __call__(self, model, score):
        if self.best_score is None: # First call
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_score - score >= self.min_delta: # New best
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_score = score
            self.counter = 0
        else: # Increment counter
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_model)
                return True
        return False
    
    def load_best_model(self, model):
        model.load_state_dict(self.best_model)
        return