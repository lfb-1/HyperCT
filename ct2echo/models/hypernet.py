import torch.nn as nn
import torch
from typing import Dict, Iterable, List, Optional, Set, Tuple
from ct2echo.models.archs import DINOv3_Encoder


def get_dinov3_target_modules_with_features(
    model: DINOv3_Encoder,
    allowed_modules: Optional[Iterable[str]] = None,
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """
    Get target modules with their input/output features for DINOv3_Encoder

    Based on dinov3_hypernet_config.json, DINOv3 has these target modules:
    - attention_query: layer.*.attention.q_proj (768→768)
    - attention_key: layer.*.attention.k_proj (768→768)
    - attention_value: layer.*.attention.v_proj (768→768)
    - attention_output: layer.*.attention.o_proj (768→768)
    - mlp_up: layer.*.mlp.up_proj (768→3072)
    - mlp_down: layer.*.mlp.down_proj (3072→768)

    Returns:
        Tuple of (target_modules, in_features, out_features)
    """

    all_pattern_specs = {
        "attention_query": ("q_proj", 768, 768),
        "attention_key": ("k_proj", 768, 768),
        "attention_value": ("v_proj", 768, 768),
        "attention_output": ("o_proj", 768, 768),
        "mlp_up": ("up_proj", 768, 3072),
        "mlp_down": ("down_proj", 3072, 768),
    }

    requested_modules: List[str] = []
    include_mlp_head = True
    if allowed_modules is None:
        requested_modules = ["mlp_up", "mlp_down"]
    else:
        include_mlp_head = False
        seen: Set[str] = set()
        for module_name in allowed_modules:
            if module_name == "mlp_head":
                include_mlp_head = True
                continue
            if module_name not in all_pattern_specs:
                raise ValueError(
                    "Unknown DINOv3 target module requested via configuration: "
                    f"{module_name}. Supported modules: {sorted(all_pattern_specs)} + ['mlp_head']"
                )
            if module_name in seen:
                continue
            requested_modules.append(module_name)
            seen.add(module_name)
        if not requested_modules:
            requested_modules = list(all_pattern_specs.keys())

    dinov3_model = model.dinov3
    target_modules: List[str] = []
    in_features: Dict[str, int] = {}
    out_features: Dict[str, int] = {}

    for name, module in dinov3_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for module_name in requested_modules:
            suffix, expected_in, expected_out = all_pattern_specs[module_name]
            if suffix in name and "layer" in name:
                if module.in_features == expected_in and module.out_features == expected_out:
                    if module_name not in target_modules:
                        target_modules.append(module_name)
                    in_features[module_name] = module.in_features
                    out_features[module_name] = module.out_features
                else:
                    print(
                        f"⚠️ Module {name} has unexpected dimensions: in={module.in_features} "
                        f"(expected {expected_in}), out={module.out_features} (expected {expected_out})"
                    )

    missing_patterns = [module for module in requested_modules if module not in target_modules]
    if missing_patterns:
        raise ValueError(
            "Specified DINOv3 target modules were not found in the encoder: "
            f"{missing_patterns}. Available modules: {sorted(target_modules)}"
        )

    if include_mlp_head:
        mlp_head_found = False
        if hasattr(model, "mlp_head"):
            for name, module in model.mlp_head.named_modules():
                if isinstance(module, nn.Linear):
                    target_modules.append("mlp_head")
                    in_features["mlp_head"] = module.in_features
                    out_features["mlp_head"] = module.out_features
                    mlp_head_found = True
                    break
        if not mlp_head_found:
            raise ValueError("Requested mlp_head adaptation, but no linear layer was found in model.mlp_head")

    print(f"✅ DINOv3 target modules selected: {target_modules}")

    return target_modules, in_features, out_features


def find_all_dinov3_linear_names(model: DINOv3_Encoder) -> List[str]:
    """
    Find all linear layer names in DINOv3_Encoder

    Args:
        model: DINOv3_Encoder model

    Returns:
        List of linear layer names
    """
    lora_module_names = set()

    # Access the DINOv3 model from the wrapper
    dinov3_model = model.dinov3

    for name, module in dinov3_model.named_modules():
        if isinstance(module, nn.Linear):
            # Include all linear layers in the transformer
            if "layer" in name and any(
                suffix in name for suffix in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
            ):
                lora_module_names.add(name)

    return list(lora_module_names)


def calculate_dinov3_total_layers(model: DINOv3_Encoder) -> int:
    """
    Calculate total number of transformer layers in DINOv3

    Returns:
        Number of transformer layers (typically 12 for DINOv3-B)
    """
    dinov3_model = model.dinov3

    # Count the number of transformer layers
    layer_count = 0
    for name, _ in dinov3_model.named_modules():
        if name.startswith("encoder.layer.") and "." not in name[len("encoder.layer.") :]:
            layer_count += 1

    return layer_count


class TaskEncoder(nn.Module):
    def __init__(self, task_emb_size: int, encoded_task_emb_size: int):
        super().__init__()
        self.encoded_task_emb_size = encoded_task_emb_size
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, encoded_task_emb_size),
            nn.LayerNorm(encoded_task_emb_size),
        )

    def get_one_hot_task_emb(self, num_tasks: int, task_idx: torch.Tensor) -> torch.Tensor:
        return torch.eye(num_tasks, device=task_idx.device)[task_idx]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"encoded_task_emb": self.mlp(x)}


class MLPResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pre_layer_norm=True, post_dropout=True):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class LoRA_Hypernet(nn.Module):
    def __init__(
        self,
        target_modules: List[str],
        task_embedding_dim: int = 128,
        num_layers: int = 12,
        lora_rank: int = 32,
        latent_size: int = 128,
        head_in_size: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        in_features: Optional[Dict[str, int]] = None,
        out_features: Optional[Dict[str, int]] = None,
        metadata_dim: int = 0,
    ):
        super().__init__()
        self.target_modules = target_modules
        self.num_layers = num_layers
        self.lora_rank = lora_rank
        self.latent_size = latent_size
        self.head_in_size = head_in_size
        self.device = device
        self.dtype = dtype
        self.metadata_dim = metadata_dim

        # Default feature sizes if not provided
        if in_features is None:
            self.in_features = {module: 512 for module in target_modules}
        else:
            self.in_features = in_features

        if out_features is None:
            self.out_features = {module: 512 for module in target_modules}
        else:
            self.out_features = out_features

        # Task encoder for embedding task descriptions
        encoded_task_emb_size = latent_size // 2
        self.task_embedding_input_dim = task_embedding_dim
        self.task_encoder = TaskEncoder(
            task_emb_size=self.task_embedding_input_dim, encoded_task_emb_size=encoded_task_emb_size
        )

        if metadata_dim > 0:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, encoded_task_emb_size, device=device),
                nn.LayerNorm(encoded_task_emb_size),
                nn.GELU(),
            )
        else:
            self.metadata_encoder = None

        # Layer depth and type encoders
        depth_emb_size = latent_size // 4
        type_emb_size = latent_size // 4

        self.layer_depth_encoder = nn.Sequential(
            nn.Embedding(num_layers, depth_emb_size),
            nn.LayerNorm(depth_emb_size),
        )
        self.layer_type_encoder = nn.Sequential(
            nn.Embedding(len(target_modules), type_emb_size),
            nn.LayerNorm(type_emb_size),
        )

        # Module name to index mapping
        self.module_to_int = {m: i for i, m in enumerate(target_modules)}

        # MLP input size
        mlp_inp_size = depth_emb_size + type_emb_size + encoded_task_emb_size

        # Main processing network
        self.mixer = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, mlp_inp_size),
            nn.SiLU(),
            nn.Dropout(0.05),
        )

        self.mlp1 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=True,
        )

        self.mlp2 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=True,
        )

        self.mlp3 = nn.Sequential(
            nn.LayerNorm(mlp_inp_size),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, head_in_size),
            nn.SiLU(),
        )

        self.split_shapes = {}
        for module in target_modules:
            in_feat = self.in_features[module]
            out_feat = self.out_features[module]
            self.split_shapes[module] = [lora_rank * in_feat, lora_rank * out_feat]
        # Output heads for each target module
        heads = {}
        for module in target_modules:
            in_feat = self.in_features[module]
            out_feat = self.out_features[module]
            # Output size for LoRA A and B matrices
            # output_size = lora_rank * (in_feat + out_feat)
            output_size = self.split_shapes[module][0] + self.split_shapes[module][1]

            layer = nn.Linear(head_in_size, output_size, bias=True, device=device)
            # Initialize weights with small random values to allow learning
            # Use small initialization so LoRA starts with minimal effect but can learn
            nn.init.normal_(layer.weight, std=0.01)

            with torch.no_grad():
                # The first part of the bias vector corresponds to `lora_A`.
                split_size_A = self.split_shapes[module][0]

                # Initialize LoRA_A bias with small random values to enable learning
                nn.init.normal_(layer.bias[:split_size_A], std=0.01)

                # The second part of the bias corresponds to `lora_B`, which should
                # start at zero to ensure no initial effect (delta_W = B*A = 0).
                layer.bias[split_size_A:].zero_()

            heads[module] = layer

        self.heads = nn.ModuleDict(heads)

        # Note: mlp_head is now part of the base model and will be adapted via LoRA

        self.to(device).to(dtype)

    def _embed_layer_depth(self, depth_indices: torch.Tensor) -> torch.Tensor:
        """Embed layer depth indices"""
        return self.layer_depth_encoder(depth_indices)

    def _embed_layer_type(self, layer_type: str) -> torch.Tensor:
        """Embed layer type"""
        module_idx = self.module_to_int[layer_type]
        module_idx = torch.tensor([module_idx], dtype=torch.long, device=self.device)
        return self.layer_type_encoder(module_idx)

    def _hypernet_forward(self, layer_indices: torch.Tensor, layer_type: str, encoded_task_emb: torch.Tensor):
        """Forward pass through the hypernetwork"""
        bs = len(layer_indices)

        # Ensure layer_indices has batch dimension and is on correct device
        if layer_indices.dim() == 1:
            layer_indices = layer_indices.to(device=self.device, dtype=torch.long)

        # Get embeddings
        depth_emb = self._embed_layer_depth(layer_indices)  # [bs, depth_emb_size]
        layer_type_emb = self._embed_layer_type(layer_type)  # [1, type_emb_size]
        layer_type_emb = layer_type_emb.expand(bs, -1)  # [bs, type_emb_size]

        # Concatenate all embeddings
        cat_emb = torch.cat([encoded_task_emb, depth_emb, layer_type_emb], dim=-1)

        # Process through MLPs
        mlp_inp = self.mixer(cat_emb)
        mlp_out = self.mlp1(mlp_inp)
        mlp_out = self.mlp2(mlp_out)
        head_input = self.mlp3(mlp_out)

        # Get output from the appropriate head
        head = self.heads[layer_type]
        head_out = head(head_input)

        # Split into A and B matrices
        splitted_out = torch.split(head_out, self.split_shapes[layer_type], dim=-1)

        return splitted_out

    def get_lora_weights(
        self,
        layer_indices: torch.Tensor,
        layer_type: str,
        task_embedding: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate LoRA A and B matrices for given layers and task

        Args:
            layer_indices: Tensor of layer indices [bs]
            layer_type: Target module type (e.g., 'q_proj', 'v_proj')
            task_embedding: Task description embedding [bs, task_emb_size]

        Returns:
            Tuple of (A_matrices, B_matrices)
        """
        # Encode task embedding
        encoded_task_output = self.task_encoder(task_embedding)
        encoded_task_emb = encoded_task_output["encoded_task_emb"]

        if self.metadata_encoder is not None:
            if metadata is None:
                metadata = torch.zeros(
                    task_embedding.shape[0],
                    self.metadata_dim,
                    device=task_embedding.device,
                    dtype=task_embedding.dtype,
                )
            metadata = metadata.to(self.device, dtype=self.dtype)
            # metadata_features = self.metadata_encoder(metadata)
            encoded_task_emb = encoded_task_emb
            # encoded_task_emb = encoded_task_emb + metadata_features

        # Forward through hypernetwork
        A_flat, B_flat = self._hypernet_forward(layer_indices, layer_type, encoded_task_emb)

        # Reshape to proper matrix dimensions
        bs = len(layer_indices)
        in_feat = self.in_features[layer_type]
        out_feat = self.out_features[layer_type]

        A_matrices = A_flat.view(bs, self.lora_rank, in_feat)
        B_matrices = B_flat.view(bs, out_feat, self.lora_rank)

        return A_matrices, B_matrices

    def forward(
        self,
        layer_indices: torch.Tensor,
        layer_type: str,
        task_embedding: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning LoRA weights

        Args:
            layer_indices: Tensor of layer indices
            layer_type: Target module type
            task_embedding: Task description embedding

        Returns:
            Dictionary with 'lora_A' and 'lora_B' tensors
        """
        A_matrices, B_matrices = self.get_lora_weights(layer_indices, layer_type, task_embedding, metadata)

        return {"lora_A": A_matrices, "lora_B": B_matrices}

    def generate_lora_for_layer(
        self,
        layer_idx: int,
        task_embedding: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate LoRA weights for all target modules in a specific layer

        Args:
            layer_idx: Layer index
            task_embedding: Task description embedding [batch_size, embed_dim]

        Returns:
            Dictionary mapping module names to their LoRA A and B matrices
            Format: {module_name: {"lora_A": [batch_size, rank, in_features],
                                   "lora_B": [batch_size, out_features, rank]}}
        """
        batch_size = task_embedding.shape[0]
        layer_indices = torch.full((batch_size,), layer_idx, device=self.device)
        layer_lora_weights = {}

        for module_name in self.target_modules:
            A_matrices, B_matrices = self.get_lora_weights(layer_indices, module_name, task_embedding, metadata)
            layer_lora_weights[module_name] = {"lora_A": A_matrices, "lora_B": B_matrices}

        return layer_lora_weights

    def generate_full_model_lora(
        self,
        task_embedding: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate LoRA weights for all layers and all target modules

        Args:
            task_embedding: Task description embedding [batch_size, embed_dim]

        Returns:
            Dictionary mapping BASE module names to their LoRA A and B matrices for all layers
            Format: {base_module_name: {"lora_A": [batch_size, num_layers, rank, in_features],
                                       "lora_B": [batch_size, num_layers, out_features, rank]}}
        """
        batch_size = task_embedding.shape[0]
        device = task_embedding.device

        # Create layer indices for all layers, repeated for each batch item
        # Ensure proper batch dimension for layer_indices
        layer_indices = torch.arange(self.num_layers, device=device, dtype=torch.long)
        layer_indices = layer_indices.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_layers]
        layer_indices = layer_indices.reshape(-1)  # [batch_size * num_layers]

        # Repeat task embeddings for each layer
        task_embedding_expanded = task_embedding.unsqueeze(1).expand(-1, self.num_layers, -1)
        task_embedding_expanded = task_embedding_expanded.reshape(-1, task_embedding.shape[-1])
        metadata_expanded = None
        if metadata is not None and self.metadata_encoder is not None:
            metadata_expanded = metadata.unsqueeze(1).expand(-1, self.num_layers, -1)
            metadata_expanded = metadata_expanded.reshape(-1, metadata.shape[-1])

        full_lora_weights = {}

        for module_name in self.target_modules:
            A_matrices, B_matrices = self.get_lora_weights(
                layer_indices,
                module_name,
                task_embedding_expanded,
                metadata_expanded,
            )

            # Reshape back to [batch_size, num_layers, ...]
            A_matrices = A_matrices.view(batch_size, self.num_layers, self.lora_rank, self.in_features[module_name])
            B_matrices = B_matrices.view(batch_size, self.num_layers, self.out_features[module_name], self.lora_rank)

            # Use base module name directly
            full_lora_weights[module_name] = {"lora_A": A_matrices, "lora_B": B_matrices}

        return full_lora_weights

    @torch.no_grad()
    def evaluate_mode(self):
        """Set the model to evaluation mode"""
        self.eval()
        return self

    def training_mode(self):
        """Set the model to training mode"""
        self.train()
        return self


