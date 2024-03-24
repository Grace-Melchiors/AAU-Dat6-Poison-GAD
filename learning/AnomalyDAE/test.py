from pygod.utils import load_data
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
import torch


# Load data and boolean truth values
data = Planetoid('./data/Cora', 'Cora', transform=T.NormalizeFeatures())[0]
data.y = data.y.bool()

detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
# Train the anomaly detector on the dataset
detector.fit(data)



detector_inj = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
data_inj, ya = gen_contextual_outlier(data, n=100, k=50)
data_inj, ys = gen_structural_outlier(data, m=10, n=10)
data_inj.y = torch.logical_or(ys, ya).long()
detector_inj.fit(data_inj)

def evaluate_anomaly_detector(detector):

    pred, score, prob, conf = detector.predict(data,
                                                return_pred=True,
                                                return_score=True,
                                                return_prob=True,
                                                return_conf=True)
    
    return (pred, score, prob, conf)

detector.predict
pred, score, prob, conf = evaluate_anomaly_detector(detector)
pred_inj, score_inj, prob_inj, conf_inj = evaluate_anomaly_detector(detector_inj)