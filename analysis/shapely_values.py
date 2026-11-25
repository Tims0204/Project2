import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from model import FullModel
from dataset import load_data, CIFAR10Dataset

# Here I am trying to be as close as possible to our lecture:
r"""
### Procedure (of Shapely Value) - Explainable AI / Lecture 5 (Shapely Values)

1. Consider an attribute $j$ of an instance $x$
2. Select an arbitrary other instance $z$ from the dataset
3. Select an arbitrary permutation of the set $\{1, \dots, n\}$
4. Create new instances $x_{i,+j}$ und $x_{i,-j}$ from $x$ and $z$, and calculate the difference in the model prediction:

$$f(x_{i,+j}) - f(x_{i,-j})$$

5. Repeat steps 2–4 $M$ times and estimate the Shapley value of attribute $j$ for instance $x$ as the average of the results:

$$\phi_j(x) = \frac{1}{M} \sum_{i=1}^{M} (f(x_{i,+j}) - f(x_{i,-j}))$$
"""

class ShapelyValues():
    def __init__(self, path:str):
        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.2470, 0.2435, 0.2616], device=self.device).view(1, 3, 1, 1)

    def load_ckpt(self):
        self.ckpt = torch.load(self.path, map_location=self.device, weights_only=False)
        self.model = FullModel(**self.ckpt["model_kwargs"]).to(self.device)
        self.model.load_state_dict(self.ckpt["state_dict"])
        self.model.eval()       # model now in inference mode -> for analysis

        mk = self.ckpt["model_kwargs"]
        self.P, self.H = mk["P"], mk["H"]
        self.G = self.H // self.P

    def _gpu_prep(self, x):
        x = torch.as_tensor(x)
        if x.shape[-1] == 3:                 # (N,32,32,3) -> (N,3,32,32)
            x = x.permute(0, 3, 1, 2)
        x = x.to(self.device, non_blocking=True).float().div_(255)
        return (x - self.mean) / self.std
    
    def _load_data(self):
        train_images, _, test_images, test_labels = load_data()
        self.pool = self._gpu_prep(train_images[:1000])          # Pool for z
        self.test_images = test_images
        self.test_labels = torch.as_tensor(test_labels).long()   # numpy -> Tensor  

    def test_accuracy(self, batch_size=512)->float:
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(self.test_images), batch_size):
                images = self.test_images[i:i+batch_size]
                labels = self.test_labels[i:i+batch_size]

                images = self._gpu_prep(images)
                labels = labels.to(self.device)

                outputs = self.model(images)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total

        return accuracy

    def patch_slice(self, j):
        r = j // self.G      # row
        c = j % self.G       # column

        return slice(self.P*r, self.P*r+self.P), slice(self.P*c, self.P*c+self.P)

    @torch.no_grad()
    def one_sample(self, x, z, j, y):
        G = self.G                         # aus ckpt
        perm = torch.randperm(G*G)
        pos = pos = (perm == j).nonzero().item()
        before = perm[:pos]
        base = z.clone()
        for k in before:
            ys, xs = self.patch_slice(k.item())
            base[:, ys, xs] = x[:, ys, xs]

        x_minus = base
        x_plus = base.clone()
        ys, xs = self.patch_slice(j)
        x_plus[:, ys, xs] = x[:, ys, xs]

        f = lambda img: self.model(img[None]).softmax(-1)[0, y]
        return (f(x_plus) - f(x_minus)).item()  

    def phi_j(self, x, j, y, M):
        values = []
        for m in range(M):
            idx = torch.randint(len(self.pool), (1,)).item()
            z = self.pool[idx]
            values.append(self.one_sample(x, z, j, y))
        vals = torch.tensor(values)
        return vals.mean().item(), (vals.std() / M**0.5).item()

    def phi_all(self, x, y, M=20):
        result = {}
        for patch in range(self.G * self.G):    # 64
            result[patch] = self.phi_j(x=x, y=y, j=patch, M=M)

        return result

    

    import numpy as np
    import matplotlib.pyplot as plt
    import os   

    def plot(self, x, result, y=None, pred=None, fname="plot.png",
             out_dir="analysis/plots", alpha=0.7):
        vals = result.values() if isinstance(result, dict) else result
        phi = torch.tensor([r[0] for r in vals]).view(self.G, self.G)
        se  = torch.tensor([r[1] for r in vals]).view(self.G, self.G)

        img = (x * self.std[0] + self.mean[0]).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        H = img.shape[0]
        a = phi.abs().max().item()
        a = a if a > 0 else 1e-8

        fig, ax = plt.subplots(
            1, 3, figsize=(10, 4.6),
            gridspec_kw={"width_ratios": [1, 1, 0.05], "wspace": 0.08},
        )

        # 1) Input
        ax[0].imshow(img, interpolation="nearest")
        t = "Input"
        if y is not None:
            t += f" (true: {y}" + (f", pred: {pred})" if pred is not None else ")")
        ax[0].set_title(t, fontsize=11)

        # 2) Input + Heatmap, Patches mit Kontur
        ax[1].imshow(img, interpolation="nearest")
        im = ax[1].imshow(
            phi.numpy(), cmap="coolwarm", vmin=-a, vmax=a, alpha=alpha,
            extent=(-0.5, H - 0.5, H - 0.5, -0.5), interpolation="nearest",
        )
        # Patch-Gitterlinien (Kontur um jeden Patch)
        for k in range(self.G + 1):
            pos = k * self.P - 0.5
            ax[1].axhline(pos, color="white", lw=0.6, alpha=0.8)
            ax[1].axvline(pos, color="white", lw=0.6, alpha=0.8)
        ax[1].set_xlim(-0.5, H - 0.5)
        ax[1].set_ylim(H - 0.5, -0.5)
        ax[1].set_title("Shapley-Value (Overlay)", fontsize=11)
        ax[1].axis("off")

        # Colorbar
        cb = fig.colorbar(im, cax=ax[2])
        cb.set_label("φ (Contribution to p(class))", fontsize=9)
        cb.ax.tick_params(labelsize=8)

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return phi, se

import numpy as np

def rank_corr(a, b):
    """Spearman: Pearson-Korrelation der Ränge (a, b: 1D-Tensoren)."""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def run_config(cfg, seeds=(42, 43, 44), n_images=20, M=200,
               runs_dir="runs/runs", out_dir="analysis/plots"):
    per_seed_phi, per_seed_pred = [], []

    for s in seeds:
        shap = ShapelyValues(f"{runs_dir}/{cfg}_s{s}.pt")
        shap.load_ckpt(); shap._load_data()
        phis, preds = [], []

        for i in range(n_images):
            # gleicher RNG-Zustand pro Bild -> gleiche z und Permutationen für alle Seeds
            torch.manual_seed(1000 + i)
            x = shap._gpu_prep(shap.test_images[i:i+1])[0]
            y = int(shap.test_labels[i])
            with torch.no_grad():
                preds.append(shap.model(x[None]).argmax(-1).item())
            result = shap.phi_all(x, y, M=M)
            phis.append(torch.tensor([result[j][0] for j in range(shap.G**2)]).view(shap.G, shap.G))
            print(f"{cfg} s{s} [{i+1}/{n_images}] pred={preds[-1]} wahr={y} sum={phis[-1].sum():.3f}", flush=True)

        per_seed_phi.append(torch.stack(phis))     # (N,G,G)
        per_seed_pred.append(preds)

    phi_all = torch.stack(per_seed_phi)            # (S,N,G,G)
    phi_mean = phi_all.mean(0)                     # (N,G,G)
    phi_std  = phi_all.std(0)                      # Streuung zwischen Seeds

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"phi": phi_all, "preds": per_seed_pred}, f"{out_dir}/{cfg}_phi_seeds.pt")

    # Plots mit dem Seed-Mittel (shap = letztes Modell, nur für Daten/Plot-Methode)
    for i in range(n_images):
        x = shap._gpu_prep(shap.test_images[i:i+1])[0]
        y = int(shap.test_labels[i])
        n_ok = sum(per_seed_pred[k][i] == y for k in range(len(seeds)))
        res = [(phi_mean[i].flatten()[j].item(), phi_std[i].flatten()[j].item())
               for j in range(shap.G**2)]
        shap.plot(x, res, y, pred=None, fname=f"{cfg}_img{i:02d}_class{y}_{n_ok}of{len(seeds)}correct.png",
                  out_dir=out_dir)

    # Stabilität: Rangkorrelation der Karten zwischen je zwei Seeds, pro Bild
    pairs = [(a, b) for a in range(len(seeds)) for b in range(a + 1, len(seeds))]
    rho = torch.tensor([[rank_corr(phi_all[a, i].flatten(), phi_all[b, i].flatten())
                         for (a, b) in pairs] for i in range(n_images)])
    print(f"\n{cfg}: mittlere Spearman-Korrelation zwischen Seeds = {rho.mean():.3f}")
    return phi_mean, phi_all, rho


if __name__ == "__main__":
    run_config("hd384_nh6_d6", seeds=(42, 43, 44), n_images=20, M=200)