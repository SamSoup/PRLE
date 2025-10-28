from .base import BasePrototypicalRegressor
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class SklearnExpertPRLE(BasePrototypicalRegressor):
    def __init__(self, *args, expert_type="knn", **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_type = expert_type
        self.expert_models = [
            self._init_sklearn_expert(expert_type)
            for _ in range(self.num_prototypes)
        ]
        self.trained = False

    def _init_sklearn_expert(self, type_):
        if type_ == "knn":
            return KNeighborsRegressor(n_neighbors=3)
        elif type_ == "svm":
            return SVR()
        elif type_ == "random_forest":
            return RandomForestRegressor()

    def forward(self, input_ids, attention_mask):
        assert self.trained, "Call fit_sklearn_experts before forward."
        embeds = self.shared_forward(input_ids, attention_mask)
        x_np = embeds.detach().cpu().numpy()
        dists = torch.cdist(embeds.unsqueeze(1), self.prototypes.unsqueeze(0))
        sim_weights = torch.softmax(-dists.squeeze(1), dim=1).cpu().numpy()

        preds = []
        for i in range(x_np.shape[0]):
            pred = 0.0
            for j, model in enumerate(self.expert_models):
                pred += sim_weights[i, j] * model.predict(x_np[i : i + 1])[0]
            preds.append(pred)
        return torch.tensor(preds, device=embeds.device), embeds

    def fit_sklearn_experts(self, x_embed, y):
        dists = torch.cdist(x_embed.unsqueeze(1), self.prototypes.unsqueeze(0))
        soft_assignments = torch.softmax(-dists.squeeze(1), dim=1)

        x_np = x_embed.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        for i in range(self.num_prototypes):
            weights = soft_assignments[:, i].detach().cpu().numpy()
            indices = weights.argsort()[
                -max(10, len(weights) // self.num_prototypes) :
            ]
            self.expert_models[i].fit(x_np[indices], y_np[indices])

        self.trained = True
