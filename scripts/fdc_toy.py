from pathlib import Path
from omegaconf import OmegaConf
import wandb
import copy
from tqdm import tqdm
from vendi_score.vendi import score_K

import torch
from torch import nn

from genexp.sampling import EulerMaruyamaSampler
from genexp.trainers.genexp import FDCTrainerFlow
from genexp.models import DiffusionModel, VPSDE, FlowModel
from genexp.utils import parse_args, update_config_with_args, seed_everything


def estimate_vendi(model: FlowModel, device, batch_size=512, num_samples=50000, n_metrics=1000, T=1000):
    sampler = EulerMaruyamaSampler(model.to(device), data_shape=(2,), device=device)

    samples = []
    
    for i in tqdm(range(num_samples // batch_size + 1)):
        trajs, ts = sampler.sample_trajectories(N=batch_size, T=T, device=device)
        samples.append(trajs[-1].full.detach().cpu())

    samples = torch.vstack(samples)[:num_samples]

    vendi_scores = []

    for i in tqdm(range(num_samples // n_metrics)):
        sub_samples = samples[i * n_metrics: (i + 1) * n_metrics]
        sims = torch.exp(-torch.cdist(sub_samples, sub_samples, p=2))

        vendi = score_K(sims)
        vendi_scores.append(vendi)

    return samples, torch.mean(torch.tensor(vendi_scores).float())


def main():
    args = parse_args()
    config = OmegaConf.load('configs/example_fdc.yaml')
    config = update_config_with_args(config, args)

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    am_config = config.adjoint_matching

    print(f"--- Running FDC ---", flush=True)
    # Setup - WandB
    if args.use_wandb:
        wandb.init(project='repo', config=OmegaConf.to_container(config))


    network = nn.Sequential(
        nn.Linear(3, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )

    sde = VPSDE(0.1, 12)

    model = DiffusionModel(network, sde)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.model.load_state_dict(torch.load('models/gauss_model.pth'))
    sampler = EulerMaruyamaSampler(model.to(device), data_shape=(2,), device=device)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sampler = EulerMaruyamaSampler(model, data_shape=(2,), device=device)
    model = model.to(device)
    fdc_trainer = FDCTrainerFlow(config, copy.deepcopy(model), copy.deepcopy(model), device=device, sampler=sampler)

    for k in tqdm(range(config.num_md_iterations)):
        for i in range(am_config.num_iterations):
            am_dataset = fdc_trainer.generate_dataset()
            fdc_trainer.finetune(am_dataset, steps=am_config.finetune_steps)

        fdc_trainer.update_base_model()

    print(f"--- Calculing Metrics ---", flush=True)
    samples_pre, vendi_pre = estimate_vendi(fdc_trainer.base_base_model, device)
    samples_fdc, vendi_fdc = estimate_vendi(fdc_trainer.fine_model, device)

    if args.use_wandb:
        wandb.log({'vendi_pre': vendi_pre, 'vendi_fdc': vendi_fdc})
    else:
        print({'vendi_pre': vendi_pre, 'vendi_fdc': vendi_fdc})
    

    if args.use_wandb and wandb.run is not None and args.save_model:
        run_name = wandb.run.name
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "No_sweep"
        save_path = Path('output') / Path(sweep_id) / Path(run_name)
        save_path.mkdir(parents=True, exist_ok=True)
        model_path = save_path / Path('final_model.pth')
        torch.save(fdc_trainer.fine_model.cpu().state_dict(), model_path)
        torch.save(samples_pre.detach().cpu(), save_path / Path('samples_pre.pth'))
        torch.save(samples_fdc.detach().cpu(), save_path / Path('samples_fdc.pth'))
    

if __name__ == '__main__':
    main()