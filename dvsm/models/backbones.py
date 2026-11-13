import torch
from transformers import AutoModel

from .modules import checkpoint_wrapper


class DINO(torch.nn.Module):
    def __init__(self, version, latent_dim, grad_checkpoint=True, use_compile=True):
        super().__init__()
        if version == 'dino' or version == 'dino-large':
            version = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        if version == 'dino-big':
            version = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        if version == 'dino-huge':
            version = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

        # Prepare the model
        try:
            self.prior = AutoModel.from_pretrained(version, local_files_only=True)
        except:
            print(f"Fail to load pretrained foundation model. May be due to missing local file.")
            print(f"Open an python terminal and run the following to download to local first.")
            print(f"  from transformers import AutoModel; AutoModel.from_pretrained({version})")
            raise Exception

        self.patch_size = self.prior.config.patch_size
        self.prior_prefix = self.prior.embeddings.cls_token.shape[1] + self.prior.embeddings.register_tokens.shape[1]
        del self.prior.embeddings.mask_token
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(3,1,1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(3,1,1))

        # Zero-init channel projection layer
        self.out_proj = torch.nn.Linear(self.prior.config.hidden_size, latent_dim, bias=False)
        torch.nn.init.constant_(self.out_proj.weight, 0.)

        # Apply grad checkpointing & compile
        for i in range(len(self.prior.layer)):
            if use_compile:
                self.prior.layer[i].forward = torch.compile(self.prior.layer[i].forward)
            if grad_checkpoint:
                self.prior.layer[i].forward = checkpoint_wrapper(self.prior.layer[i])

    def forward(self, image):
        # image: [B V 3 H W] under intensity range [0, 1]
        x = self.prior((image.flatten(0,1) - self.img_mean) / self.img_std)
        x = x.last_hidden_state[:, self.prior_prefix:]
        x = self.out_proj(x)
        x = x.unflatten(0, image.shape[:2])
        return x


class RADIO(torch.nn.Module):
    def __init__(self, version, latent_dim, grad_checkpoint=True, use_compile=True):
        super().__init__()
        if version == 'radio' or version == 'radio-large' or version == 'radio-so400m':
            version = "c-radio_v4-so400m"
        if version == 'radio-huge':
            version = "c-radio_v4-h"

        # Prepare the model
        # TODO: may cause bug if weight is not downloaded with multi-gpu training
        self.prior = torch.hub.load('NVlabs/RADIO', 'radio_model', version=version, skip_validation=True)
        self.patch_size = self.prior._patch_size

        # Zero-init channel projection layer
        self.out_proj = torch.nn.Linear(self.prior.model.embed_dim, latent_dim, bias=False)
        torch.nn.init.constant_(self.out_proj.weight, 0.)

        # Apply grad checkpointing & compile
        for i in range(len(self.prior.model.blocks)):
            if use_compile:
                self.prior.model.blocks[i].forward = torch.compile(self.prior.model.blocks[i].forward)
            if grad_checkpoint:
                self.prior.model.blocks[i].forward = checkpoint_wrapper(self.prior.model.blocks[i])

    def forward(self, image):
        # image: [B V 3 H W] under intensity range [0, 1]
        x = self.prior(image.flatten(0,1))
        x = x.features
        x = self.out_proj(x)
        x = x.unflatten(0, image.shape[:2])
        return x
