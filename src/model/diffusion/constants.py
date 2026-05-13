import torch


def add_constants(model, betas, timesteps):
    assert (betas >= 0).all() and (betas <= 1).all()

    # buffers -> move in the GPU with model.to() auto
    # persistent=False -> not saved in state_dict

    model.register_buffer("betas", betas, persistent=False)

    model.register_buffer("alphas", 1.0 - model.betas, persistent=False)
    model.register_buffer(
        "alphas_cumprod",
        torch.cumprod(model.alphas, axis=0),
        persistent=False
    )
    model.register_buffer(
        "one_minus_alphas_cumprod",
        1.0 - model.alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "inv_one_minus_alphas_cumprod",
        1.0 / model.one_minus_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "alphas_cumprod_prev",
        torch.cat([torch.ones(1), model.alphas_cumprod[:-1]]),
        persistent=False
    )
    model.register_buffer(
        "alphas_cumprod_next",
        torch.cat([model.alphas_cumprod[:-1], torch.ones(0)]),
        persistent=False
    )
    # calculations for diffusion q(x_t | x_{t-1}) and others
    model.register_buffer(
        "sqrt_alphas_cumprod",
        torch.sqrt(model.alphas_cumprod),
        persistent=False
    )
    model.register_buffer(
        "sqrt_one_minus_alphas_cumprod",
        torch.sqrt(1.0 - model.alphas_cumprod),
        persistent=False
    )
    model.register_buffer(
        "sqrt_one_minus_alphas_cumprod_prev",
        torch.sqrt(1 - model.alphas_cumprod_prev),
        persistent=False
    )
    model.register_buffer(
        "inv_sqrt_alphas_cumprod",
        1.0 / model.sqrt_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "sqrt_inv_alphas_cumprod_minus_one",
        torch.sqrt(1.0 / model.alphas_cumprod - 1),
        persistent=False
    )
    model.register_buffer(
        "sqrt_alphas_cumprod_over_one_minus_alphas_cumprod",
        torch.sqrt(model.alphas_cumprod) / (1.0 - model.alphas_cumprod),
        persistent=False
    )
    model.register_buffer(
        "sqrt_alphas_cumprod_over_sqrt_one_minus_alphas_cumprod",
        torch.sqrt(model.alphas_cumprod / (1.0 - model.alphas_cumprod)),
        persistent=False
    )
    model.register_buffer(
        "one_minus_alphas_cumprod_over_sqrt_alphas_cumprod",
        (1.0 - model.alphas_cumprod) / torch.sqrt(model.alphas_cumprod),
        persistent=False
    )
    model.register_buffer(
        "sqrt_inv_one_minus_alphas_cumprod",
        torch.sqrt(1.0 / (1.0 - model.alphas_cumprod)),
        persistent=False
    )
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    model.register_buffer(
        "posterior_variance",
        model.betas * (1.0 - model.alphas_cumprod_prev) / model.one_minus_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_coef1",
        model.betas * torch.sqrt(model.alphas_cumprod_prev) / model.one_minus_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_coef2",
        (1.0 - model.alphas_cumprod_prev) * torch.sqrt(model.alphas) / model.one_minus_alphas_cumprod,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_eps_coef1",
        1.0 / torch.sqrt(model.alphas),
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_eps_coef2",
        betas / (model.sqrt_one_minus_alphas_cumprod * torch.sqrt(model.alphas)),
        persistent=False
    )
    # same coef for xt
    model.register_buffer(
        "posterior_mean_score_coef1",
        model.posterior_mean_eps_coef1,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_score_coef2",
        betas / torch.sqrt(model.alphas),
        persistent=False
    )


def add_flow_constants(model, sigmas, timesteps):
    """
    Add all constants needed for flow matching calculations.
    Similar to add_constants but with flow matching specific values.
    
    Args:
        model: The flow matching model to register buffers on
        sigmas: The sigma values for the timesteps
        timesteps: The number of timesteps
    """
    assert (sigmas >= 0).all() and (sigmas <= 1).all()

    # Register basic sigma schedule
    model.register_buffer("sigmas", sigmas, persistent=False)
    model.register_buffer("timesteps", torch.tensor(timesteps), persistent=False)
    
    # Core flow matching coefficients
    model.register_buffer("one_minus_sigmas", 1.0 - model.sigmas, persistent=False)
    
    # Previous and next sigmas (for transitions)
    model.register_buffer(
        "sigmas_prev",
        torch.cat([torch.zeros(1), model.sigmas[:-1]]),
        persistent=False
    )
    model.register_buffer(
        "sigmas_next",
        torch.cat([model.sigmas[1:], torch.zeros(1)]),
        persistent=False
    )
    
    # Square roots and inverses
    model.register_buffer("sqrt_sigmas", torch.sqrt(model.sigmas), persistent=False)
    model.register_buffer("sqrt_one_minus_sigmas", torch.sqrt(model.one_minus_sigmas), persistent=False)
    model.register_buffer("inv_sqrt_one_minus_sigmas", 1.0 / model.sqrt_one_minus_sigmas, persistent=False)
    model.register_buffer("inv_sigmas", 1.0 / model.sigmas, persistent=False)
    model.register_buffer("inv_sqrt_sigmas", 1.0 / model.sqrt_sigmas, persistent=False)
    
    # Flow transition coefficients
    model.register_buffer("delta_sigmas", model.sigmas_next - model.sigmas, persistent=False)
    
    # Velocity field coefficients
    model.register_buffer(
        "velocity_scaling",
        model.sigmas / model.one_minus_sigmas,
        persistent=False
    )
    
    # For converting between different representations
    # v to x0 coefficient: x0 = xt - (sigmas/one_minus_sigmas) * v
    model.register_buffer(
        "v_to_x0_coef",
        model.sigmas / model.one_minus_sigmas,
        persistent=False
    )
    
    # noise to x0 coefficient: x0 = (xt - sigmas * noise) / one_minus_sigmas
    model.register_buffer(
        "noise_to_x0_coef1",
        model.sigmas / model.one_minus_sigmas,
        persistent=False
    )
    model.register_buffer(
        "noise_to_x0_coef2",
        1.0 / model.one_minus_sigmas,
        persistent=False
    )
    
    # x0 to noise coefficient: noise = (xt - one_minus_sigmas * x0) / sigmas
    model.register_buffer(
        "x0_to_noise_coef",
        model.one_minus_sigmas / model.sigmas,
        persistent=False
    )
    
    # For backward ODE solver - Euler step
    # x_{t-dt} = xt - v * dt where dt = sigma_t - sigma_{t-dt}
    model.register_buffer(
        "euler_step_coef",
        model.sigmas - model.sigmas_prev,
        persistent=False
    )
    
    # For numerical stability in probability flow
    model.register_buffer(
        "flow_norm_const",
        0.5 * torch.log(2 * torch.tensor(3.14159265358979323846) * model.sigmas),
        persistent=False
    )
    
    # ODE-based flow coefficients
    model.register_buffer(
        "flow_mean_coef1",
        model.one_minus_sigmas,
        persistent=False
    )
    model.register_buffer(
        "flow_mean_coef2",
        model.sigmas,
        persistent=False
    )
    
    # For compatibility with traditional diffusion code
    # Map sigmas to alpha-like parameters
    model.register_buffer("alphas", model.one_minus_sigmas, persistent=False)
    model.register_buffer("alphas_cumprod", model.one_minus_sigmas, persistent=False)
    model.register_buffer("sqrt_alphas_cumprod", model.sqrt_one_minus_sigmas, persistent=False)
    model.register_buffer("sqrt_one_minus_alphas_cumprod", model.sqrt_sigmas, persistent=False)
    
    # For calculating probability flow
    model.register_buffer(
        "flow_score_coef",
        -0.5 / model.sigmas,
        persistent=False
    )
    
    # For denoising process
    model.register_buffer(
        "posterior_mean_coef1",
        model.one_minus_sigmas,
        persistent=False
    )
    model.register_buffer(
        "posterior_mean_coef2",
        model.sigmas,
        persistent=False
    )
    
    # Used in loss weighting schemes
    model.register_buffer(
        "loss_weight_sigmas",
        1.0 / (model.sigmas ** 2),
        persistent=False
    )
    model.register_buffer(
        "loss_weight_logit_normal",
        torch.ones_like(model.sigmas),  # Will be computed dynamically during training
        persistent=False
    )