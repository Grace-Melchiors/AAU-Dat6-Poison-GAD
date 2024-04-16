class Experiment():
    def __init__(self, data, pred, score, prob, conf):
        self.data = data
        self.anomaly_labels = data.y.bool()
        self.pred = pred
        self.score = score
        self.prob = prob
        self.conf = conf
