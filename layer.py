from helper import Helper

class Layer:
    def __init__(self, previous):
        self.prev = previous

    def activate(self):
        if self.prev:
            X = self.prev.activate()
            self.Y = Helper.sigmoid(X.dot(self.W))
        return self.Y