# #------------------- libraries
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import copy
import matplotlib
import imageio
matplotlib.use("Agg")
import time
from PIL import Image
import torchvision.transforms as T

from deepinv.utils import load_example
from forward.lin_rad import LinearRadon
from methods.Tikhonov import TikhonovLandweber
import aux_function as af
from methods.mlp import MLP, ImplicitTikhonovSolver
from deepinv.models import DnCNN
import deepinv as dinv
from methods.PnP import PnPPGD
import scipy.io


#------------------- device
device = "cuda" if torch.cuda.is_available() else "cpu"

for snr in [np.linspace(18, 30, 10)[0]]:
    for realizzazioni in range(25):
        # ------------------- settings
        img_size = 128
        n_angles = 180
        snr = 30.0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"results_nat/results_delta_{snr:.1f}/RadonLinear/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        x = load_example("SheppLogan.png", img_size=img_size, grayscale=True, resize_mode="resize", device=device)
        # x = dinv.utils.load_example("demo_mini_subset_fastmri_brain_0.pt", device=device)[:, 0, :, :].unsqueeze(0)
        # x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        # x = x / x.max()
        mat = scipy.io.loadmat("GroundTruthReconstruction.mat")
        x = torch.from_numpy(mat['FBP1200'].astype(np.float32))  # converti in float32
        x = x.unsqueeze(0).unsqueeze(0)
        x = x.to(device)
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))


# Define physics

        # img = imageio.imread("data/2022-07-01_Walnuss_1200.tif")
        # img = img.astype("float32")
        # x = torch.from_numpy(img).float().to(device)
        # x = x / x.max()  
        # x = x.unsqueeze(0).unsqueeze(0)
        # x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        angles = torch.linspace(0, 180, n_angles, device=device)
        A = LinearRadon(angles)
        y = A.forward(x)
        noise_level = 1 / 10 ** (snr / 20) / np.sqrt(y.shape[1] * y.shape[2]) * torch.norm(y)
        y_noise = y + noise_level * torch.randn_like(y)

        # ------------------- save data
        meta_data_dic = {
            "metadata": {
                "img_size": img_size,
                "n_angles": n_angles,
                "snr": snr,
                "timestamp": timestamp
            },
            "data": {
                "x_true": x.cpu().squeeze().tolist(),
                "y_noisy": y_noise.cpu().squeeze().tolist(),
                "angles": angles.cpu().squeeze().tolist()
            },
        }
        af.save_json(meta_data_dic, os.path.join(save_dir, "data.json"))
        af.plot_data([x, y, y_noise], save_path=save_dir, titles=['Ground Truth', 'Sinogram', 'Noisy Sinogram'])

        #-------------------------------- Tikhonov
        tik_dir = os.path.join(save_dir, "Tikhonov")
        os.makedirs(tik_dir, exist_ok=True)

        #big grid
        alpha_array_big = 10.0 ** np.arange(-10, 6)
        step_array = (np.array([1.0, 5.0])[:, None] * 10.0 ** np.arange(-5, -3)).reshape(-1)
        step_array = np.sort(step_array)
        n_iter_big = 500
        tol_rel = 1e-3

        recons_big = torch.zeros((len(alpha_array_big), len(step_array), img_size, img_size))
        errors_big = torch.zeros((len(alpha_array_big), len(step_array)))
        results_big = []
        best_err_big = np.inf
        best_alpha_big = None
        best_x_big = None
        start_time = time.time()
        for i_alpha, alpha in enumerate(alpha_array_big):
            for i_step, step in enumerate(step_array):
                solver = TikhonovLandweber(operator=A, alpha=alpha, step_size=step, n_iters=n_iter_big, tol_rel = tol_rel, verbose=True)
                x_rec = solver.run(y=y_noise)
                err = torch.norm(x_rec - x).item()
                recons_big[i_alpha, i_step] = x_rec.cpu()
                errors_big[i_alpha, i_step] = err

                entry = {
                    "alpha": float(alpha),
                    "step": float(step),
                    "error": err,
                    "x_rec": x_rec.detach().cpu()
                }
                results_big.append(entry)
                if err < best_err_big:
                    best_err_big = err
                    best_alpha_big = alpha
                    best_step_big = step
                    best_x_big = x_rec.detach()
        af.save_reconstructions_grid(recons_big, errors_big, alpha_array_big, step_array, tik_dir, method_name="Tikhonov", filename="reconstruction_big", param1_name='alpha', param2_name='step size')
        af.plot_semi_convergence(alpha_array_big, errors_big, tik_dir, method_name="Tikhonov", filename="semi_convergence_big")

        # fine grid
        n_iter_fine = 500
        alpha_array_fine = np.linspace(best_alpha_big*0.5, best_alpha_big*1.5, 10)
        recons_fine = torch.zeros((len(alpha_array_fine), 1, img_size, img_size))
        errors_fine = torch.zeros(len(alpha_array_fine))
        results_fine = []
        best_err_fine = np.inf
        best_alpha_fine = None
        best_x_fine = None

        for i_alpha, alpha in enumerate(alpha_array_fine):
            solver = TikhonovLandweber(operator=A, alpha=alpha, step_size=best_step_big, n_iters=n_iter_fine, tol_rel = tol_rel, verbose=True)
            x_rec = solver.run(y=y_noise)
            err = torch.norm(x_rec - x).item()
            recons_fine[i_alpha, 0] = x_rec.cpu()
            errors_fine[i_alpha,] = err
            entry = {
                "alpha": float(alpha),
                "error": err,
                "x_rec": x_rec.detach().cpu()
            }
            results_fine.append(entry)
            if err < best_err_fine:
                best_err_fine = err
                best_alpha_fine = alpha
                best_x_fine = x_rec.detach()
        end_time = time.time()
        af.save_reconstructions_grid(recons_fine, errors_fine, alpha_array_fine, [best_step_big], tik_dir, method_name="Tikhonov", filename="reconstruction_fine", param1_name='alpha', param2_name='step size')
        af.plot_semi_convergence(alpha_array_fine, errors_fine, tik_dir, method_name="Tikhonov", filename="semi_convergence_fine")

        params_tikhonov_gen = {
        "method": "Tikhonov",
        "n_iters_big": n_iter_big,
        "tol_rel": tol_rel,
        "alpha_array_big": alpha_array_big.tolist(),
        "step_array": step_array.tolist(),
        "best_alpha_big": float(best_alpha_big), 
        "best_step_big": float(best_step_big), 
        "best_error_big": float(best_err_big),
        "n_iters_fine": n_iter_fine,
        "alpha_array_fine": alpha_array_fine.tolist(),
        "best_alpha_fine": float(best_alpha_fine), 
        "best_error_fine": float(best_err_fine),
        "time": end_time - start_time
        }
        af.save_json(params_tikhonov_gen, os.path.join(tik_dir, "params_tikhonov.json"))

        torch.save(
            {
                "results_big": results_big,
                "results_fine": results_fine,
            },
            os.path.join(tik_dir, "reconstructions.pt")
        )
        torch.save(
            {
                "x_rec": best_x_fine.cpu(),
                "alpha": float(best_alpha_fine),
                "step": float(best_step_big),
                "error": float(best_err_fine),
                "method": "Tikhonov",
                "grid": "fine"
            },
            os.path.join(tik_dir, "best_reconstruction.pt")
        )

        # ----------------------- PnP
        pnp_dir = os.path.join(save_dir, "PnP")
        os.makedirs(pnp_dir, exist_ok=True)

        x0 = A.adjoint(y_noise, img_shape=(img_size, img_size))
        x0 = x0.clamp(min=0.0)
        x0 = x0 / torch.max(x0)

        denoiser = DnCNN(in_channels=1, out_channels=1, pretrained="download", device=device,)
        denoiser.eval()

        n_it_init = 10
        tol_rel_init = 1e-4
        step_init = 1e-4 
        sigma_init = 1000 / 255.0 


        warmup_solver = PnPPGD(operator=A, denoiser=denoiser, sigma_denoiser=sigma_init, step_size=step_init, n_iters=n_it_init, tol_rel=tol_rel_init, verbose=True)
        x0_warm = warmup_solver.run(y=y_noise, x0=x0, x_gt=x)

        sigma_array_pnp = torch.arange(00, 500, 50, device=device) / 255.0
        step_array_pnp = torch.tensor([1e-5, 5e-5, 1e-4, 5e-4, 1e-3], device=device)
        n_iter_pnp = 1000
        tol_rel = 1e-3

        recons_pnp = torch.zeros((len(sigma_array_pnp), len(step_array_pnp), img_size, img_size))
        errors_pnp = torch.zeros((len(sigma_array_pnp), len(step_array_pnp)))
        best_err = np.inf
        best_sigma = None
        best_step = None
        best_x = None
        results_pnp = []
        start_time = time.time()
        for i, sigma in enumerate(sigma_array_pnp):
            for j, step in enumerate(step_array_pnp):

                solver = PnPPGD(operator=A, denoiser=denoiser, sigma_denoiser=sigma, step_size=step, n_iters=n_iter_pnp, tol_rel=tol_rel, verbose=True)
                x_rec = solver.run(y=y_noise, x0=x0_warm, x_gt=x)
                err = torch.norm(x_rec - x).item()

                recons_pnp[i, j] = x_rec.cpu()
                errors_pnp[i, j] = err

                entry = {
                    "sigma": float(sigma),
                    "step": float(step),
                    "error": err,
                    "x_rec": x_rec.detach().cpu()
                }

                results_pnp.append(entry)

                if err < best_err:
                    best_err = err
                    best_sigma = sigma
                    best_step = step 
                    best_x = x_rec


        af.save_reconstructions_grid(recons_pnp, errors_pnp, sigma_array_pnp, step_array_pnp, pnp_dir, method_name="PnP", filename="reconstruction_grid", param1_name='des', param2_name='step size')
        end_time = time.time()

        params_pnp_gen = {
            "method": "PnP",
            "denoiser": {
                "name": "DnCNN",
                "in_channels": 1,
                "out_channels": 1,
                "pretrained": "download",
                "eval_mode": True
            },
            "n_iters_init": n_it_init,
            "tol_rel_init": tol_rel_init,
            "sigma_init": sigma_init,
            "step_init": step_init,
            "n_iters_pnp": n_iter_pnp,
            "tol_rel_pnp": tol_rel,
            "sigma_array": sigma_array_pnp.tolist(),
            "step_array": step_array_pnp.tolist(),
            "best_sigma": float(best_sigma),
            "best_step": float(best_step),
            "best_err": float(best_err),
            "time": end_time - start_time
        }
        af.save_json(params_pnp_gen, os.path.join(pnp_dir, "params_pnp.json"))

        torch.save(
            {
                "results_pnp": results_pnp,
            },
            os.path.join(pnp_dir, "reconstructions.pt")
        )
        torch.save(
            {
                "x_rec": best_x.cpu(),
                "sigma": float(best_sigma),
                "step": float(best_step),
                "error": float(best_err),
                "method": "PnP",
            },
            os.path.join(pnp_dir, "best_reconstruction.pt")
        )


        #------------------------------------ mlp
        mlp_dir = os.path.join(save_dir, "MLP")
        os.makedirs(mlp_dir, exist_ok=True)

        height, width = x.shape[-2:]
        x_points = torch.linspace(-1, 1, width, device=device, requires_grad=True) 
        y_points = torch.linspace(-1, 1, height, device=device, requires_grad=True) 
        xv, yv = torch.meshgrid(x_points, y_points, indexing='ij')
        points = torch.stack((xv, yv), dim=-1).reshape(-1, 2)
        points = points.to(device).detach()

        n_layers = 6
        n_nodes = 256
        lr = 1e-2

        #big grid
        alpha_array_big = 10.0 ** np.arange(-10, 6)
        num_init = 10
        n_inter_init = 200
        n_iter_long = 10_000 
        scheduler_gamma = 0.9997

        recons_big = torch.zeros((len(alpha_array_big), 1, img_size, img_size))
        errors_big = torch.zeros((len(alpha_array_big)))
        init_big = torch.zeros((len(alpha_array_big)))
        results_big = []
        init_states = []
        best_err_big = np.inf
        best_alpha_big = None
        best_x_big = None

        for k in range(num_init):
            model = MLP(
                num_layers=n_layers,
                num_nodes_per_layer=n_nodes,
                activation="leakyrelu",
                input_dim=2
            ).to(device)
            init_states.append(copy.deepcopy(model.state_dict()))

        start_time = time.time()
        for i, alpha in enumerate(alpha_array_big):

            best_short_loss = np.inf
            best_init = None
            best_init_state = None

            print(f"\n[alpha {alpha:.2e}]")

            # ----------------------------
            # short runs (same inits)
            # ----------------------------
            for init_num in range(num_init):
                print(f"  [init {init_num+1}/{num_init}] short run")

                model = MLP(
                    num_layers=n_layers,
                    num_nodes_per_layer=n_nodes,
                    activation="leakyrelu",
                    input_dim=2
                ).to(device)

                model.load_state_dict(init_states[init_num])

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                solver = ImplicitTikhonovSolver(
                    model=model,
                    operator=A,
                    optimizer=optimizer,
                    alpha=alpha,
                    n_iters=n_inter_init,
                    verbose=True,
                    device=device,
                    scheduler_gamma=scheduler_gamma
                )

                _, best_state, _, best_loss = solver.run(
                    points=points,
                    y=y_noise,
                    image_gt=x
                )

                if best_loss < best_short_loss:
                    best_short_loss = best_loss
                    best_init = init_num
                    best_init_state = best_state

            # ----------------------------
            # long run (best init only)
            # ----------------------------
            model = MLP(
                num_layers=n_layers,
                num_nodes_per_layer=n_nodes,
                activation="leakyrelu",
                input_dim=2
            ).to(device)

            model.load_state_dict(best_init_state)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            solver = ImplicitTikhonovSolver(
                model=model,
                operator=A,
                optimizer=optimizer,
                alpha=alpha,
                n_iters=n_iter_long,
                verbose=True,
                device=device,
                scheduler_gamma=scheduler_gamma
            )

            model_opt, best_state_long, loss_hist_long, best_loss = solver.run(
                points=points,
                y=y_noise,
                image_gt=x
            )

            with torch.no_grad():
                H, W = x.shape[-2:]
                x_rec = model_opt(points).view(1, 1, H, W)

            err = torch.norm(x_rec - x).item()

            recons_big[i] = x_rec.cpu()
            errors_big[i] = err
            init_big[i] = best_init

            results_big.append({
                "alpha": float(alpha),
                "error": err,
                "init": best_init,
                "x_rec": x_rec.cpu()
            })

            if err < best_err_big:
                best_err_big = err
                best_alpha_big = alpha
                best_x_big = x_rec.detach()

        af.save_reconstructions_grid(recons_big, errors_big, alpha_array_big, init_big, mlp_dir, method_name="MLP", filename="reconstruction_big", param1_name='alpha', param2_name='init')
        af.plot_semi_convergence(alpha_array_big, errors_big, mlp_dir, method_name="MLP", filename="semi_convergence_big")

        # fine grid
        alpha_array_fine = np.linspace(best_alpha_big*0.5, best_alpha_big*1.5, 10)
        recons_fine = torch.zeros((len(alpha_array_fine), 1, img_size, img_size))
        errors_fine = torch.zeros(len(alpha_array_fine))
        init_fine = torch.zeros((len(alpha_array_fine)))
        results_fine = []
        best_err_fine = np.inf
        best_alpha_fine = None
        best_x_fine = None




        for i, alpha in enumerate(alpha_array_fine):

            best_short_loss = np.inf
            best_init = None
            best_init_state = None

            print(f"\n[FINE alpha {alpha:.2e}]")

            for init_num in range(num_init):
                print(f"  [init {init_num+1}/{num_init}] short run")

                model = MLP(
                    num_layers=n_layers,
                    num_nodes_per_layer=n_nodes,
                    activation="leakyrelu",
                    input_dim=2
                ).to(device)

                model.load_state_dict(init_states[init_num])

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                solver = ImplicitTikhonovSolver(
                    model=model,
                    operator=A,
                    optimizer=optimizer,
                    alpha=alpha,
                    n_iters=n_inter_init,
                    verbose=True,
                    device=device,
                    scheduler_gamma=scheduler_gamma
                )

                _, best_state, _, best_loss = solver.run(
                    points=points,
                    y=y_noise,
                    image_gt=x
                )

                if best_loss < best_short_loss:
                    best_short_loss = best_loss
                    best_init = init_num
                    best_init_state = best_state

            init_fine[i] = best_init

            model = MLP(
                num_layers=n_layers,
                num_nodes_per_layer=n_nodes,
                activation="leakyrelu",
                input_dim=2
            ).to(device)

            model.load_state_dict(best_init_state)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            solver = ImplicitTikhonovSolver(
                model=model,
                operator=A,
                optimizer=optimizer,
                alpha=alpha,
                n_iters=n_iter_long,
                verbose=True,
                device=device,
                scheduler_gamma=scheduler_gamma
            )

            model_opt, best_state_long, loss_hist_long, best_loss = solver.run(
                points=points,
                y=y_noise,
                image_gt=x
            )

            with torch.no_grad():
                H, W = x.shape[-2:]
                x_rec = model_opt(points).view(1, 1, H, W)

            err = torch.norm(x_rec - x).item()

            recons_fine[i] = x_rec.cpu()
            errors_fine[i] = err

            results_fine.append({
                "alpha": float(alpha),
                "error": err,
                "init": best_init,
                "x_rec": x_rec.cpu()
            })

            if err < best_err_fine:
                best_err_fine = err
                best_alpha_fine = alpha
                best_x_fine = x_rec.detach()

        af.save_reconstructions_grid(recons_fine, errors_fine, alpha_array_fine, init_big, mlp_dir, method_name="MLP", filename="reconstruction_fine", param1_name='alpha', param2_name='init')
        af.plot_semi_convergence(alpha_array_fine, errors_fine, mlp_dir, method_name="MLP", filename="semi_convergence_fine")
        end_time = time.time()
        params_mlp_gen = {
        "method": "MLP",
        "n_layers": n_layers,
        "n_nodes": n_nodes, 
        "lr_init": lr, 
        "scheduler_gamma": scheduler_gamma,
        "num_init": num_init,
        "n_inter_init": n_inter_init, 
        "alpha_array_big": alpha_array_big.tolist(),
        "init_big": init_big.tolist(),
        "best_alpha_big": float(best_alpha_big), 
        "init_fine": init_fine.tolist(),
        "best_err_big": float(best_err_big),
        "n_iter_long": n_iter_long,
        "alpha_array_fine": alpha_array_fine.tolist(), 
        "best_alpha_fine": float(best_alpha_fine),
        "best_err_fine": float(best_err_fine),
        "time": end_time - start_time
        }

        af.save_json(params_mlp_gen, os.path.join(mlp_dir, "params_mlp.json"))

        torch.save(model, os.path.join(mlp_dir, "model_full.pth"))
        torch.save(
            {
                "results_big": results_big,
                "results_fine": results_fine,
            },
            os.path.join(mlp_dir, "reconstructions.pt")
        )
        torch.save(
            {
                "x_rec": best_x_fine.cpu(),
                "alpha": float(best_alpha_fine),
                "error": float(best_err_fine),
                "method": "MLP",
                "grid": "fine"
            },
            os.path.join(mlp_dir, "best_reconstruction.pt")
        )







