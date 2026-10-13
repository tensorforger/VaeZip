import torch
from torch import nn
from diffusers.models import AutoencoderKLFlux2


class Flux2VAE(nn.Module):
    NAME = "flux_2"

    def __init__(self, path_to_model: str, device: torch.device = "cuda"):
        super().__init__()

        self.latent_channels = 32
        self.spatial_compression = 8

        self.model = AutoencoderKLFlux2.from_pretrained(
            path_to_model, device=device
        ).to(device)
        self.model.eval()

        self.mean = self.model.bn.running_mean.view(1, -1, 1, 1)
        self.std = torch.sqrt(
            self.model.bn.running_var.view(1, -1, 1, 1)
            + self.model.config.batch_norm_eps
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def patchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(
            batch_size, num_channels_latents * 4, height // 2, width // 2
        )
        return latents

    def unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(
            batch_size, num_channels_latents // (2 * 2), 2, 2, height, width
        )
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(
            batch_size, num_channels_latents // (2 * 2), height * 2, width * 2
        )
        return latents

    def norm_unpatchified_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.patchify_latents(latents)
        latents = latents - self.mean
        latents = latents / self.std
        latents = self.unpatchify_latents(latents)

        return latents

    def denorm_unpatchified_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.patchify_latents(latents)
        latents = latents * self.std
        latents = latents + self.mean
        latents = self.unpatchify_latents(latents)

        return latents

    @torch.no_grad()
    def encode(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        latents = self.model.encode(image.to(self.dtype)).latent_dist.sample()
        latents = self.norm_unpatchified_latents(latents)

        return latents

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        latents = self.denorm_unpatchified_latents(latents)
        image = self.model.decode(latents.to(self.dtype), return_dict=False)[0]

        return image

    def get_latent_channels(self) -> int:
        """
        returns:
            latent_channels of the VAE
        """
        return self.latent_channels

    def get_spatial_compression(self) -> int:
        """
        returns:
            spatial_compression of the VAE
        """
        return self.spatial_compression

    def get_vae_name(self) -> str:
        return self.NAME
