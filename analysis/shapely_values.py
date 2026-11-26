from pathlib import Path

def run_config(cfg=None, ckpts=None, seeds=(42, 43, 44), n_images=20, M=200,
               runs_dir="runs", out_dir="analysis/plots"):
    # --- Checkpoint-Pfade bestimmen ---
    if ckpts is None:                                   # alter Modus: cfg + seeds
        paths = [f"{runs_dir}/{cfg}_s{s}.pt" for s in seeds]
    elif isinstance(ckpts, (str, os.PathLike)):         # ein einzelnes .pt
        paths = [str(ckpts)]
    else:                                               # Liste von .pt
        paths = [str(p) for p in ckpts]

    if cfg is None:
        cfg = Path(paths[0]).stem                       # Name aus Dateiname
    n_models = len(paths)

    per_seed_phi, per_seed_pred = [], []

    for path in paths:
        shap = ShapelyValues(path)
        shap.load_ckpt(); shap._load_data()
        phis, preds = [], []

        for i in range(n_images):
            torch.manual_seed(1000 + i)   # gleiche z und Permutationen pro Bild
            x = shap._gpu_prep(shap.test_images[i:i+1])[0]
            y = int(shap.test_labels[i])
            with torch.no_grad():
                preds.append(shap.model(x[None]).argmax(-1).item())
            result = shap.phi_all(x, y, M=M)
            phis.append(torch.tensor([result[j][0] for j in range(shap.G**2)]).view(shap.G, shap.G))
            print(f"{cfg} {Path(path).stem} [{i+1}/{n_images}] pred={preds[-1]} wahr={y} sum={phis[-1].sum():.3f}", flush=True)

        per_seed_phi.append(torch.stack(phis))     # (N,G,G)
        per_seed_pred.append(preds)

    phi_all = torch.stack(per_seed_phi)            # (S,N,G,G)
    phi_mean = phi_all.mean(0)                     # (N,G,G)
    # bei nur einem Modell ist std undefiniert (NaN) -> 0
    phi_std = phi_all.std(0) if n_models > 1 else torch.zeros_like(phi_mean)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"phi": phi_all, "preds": per_seed_pred}, f"{out_dir}/{cfg}_phi_seeds.pt")

    for i in range(n_images):
        x = shap._gpu_prep(shap.test_images[i:i+1])[0]
        y = int(shap.test_labels[i])
        n_ok = sum(per_seed_pred[k][i] == y for k in range(n_models))
        res = [(phi_mean[i].flatten()[j].item(), phi_std[i].flatten()[j].item())
               for j in range(shap.G**2)]
        shap.plot(x, res, y, pred=None,
                  fname=f"{cfg}_img{i:02d}_class{y}_{n_ok}of{n_models}correct.png",
                  out_dir=out_dir)

    # Stabilität nur sinnvoll bei >= 2 Modellen
    rho = None
    if n_models > 1:
        pairs = [(a, b) for a in range(n_models) for b in range(a + 1, n_models)]
        rho = torch.tensor([[rank_corr(phi_all[a, i].flatten(), phi_all[b, i].flatten())
                             for (a, b) in pairs] for i in range(n_images)])
        print(f"\n{cfg}: mittlere Spearman-Korrelation zwischen Seeds = {rho.mean():.3f}")
    else:
        print(f"\n{cfg}: nur ein Modell -> keine Seed-Korrelation berechnet")
    return phi_mean, phi_all, rho


if __name__ == "__main__":
    # a) einzelnes .pt
    run_config(ckpts="runs/hd96_nh6_d2_s42.pt", n_images=20, M=200)

    # b) wie bisher mehrere Seeds
    # run_config("hd96_nh6_d2", seeds=(42, 43, 44), n_images=20, M=200)

    # c) beliebige Liste
    # run_config("mein_name", ckpts=["a.pt", "b.pt"])