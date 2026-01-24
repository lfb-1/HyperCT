"""
Learnable Task Embedding System for Ablation Study

This module provides learnable task embeddings as an alternative to pre-defined text embeddings,
allowing for an ablation study comparing fixed vs. learnable task representations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from loguru import logger
from ct2echo.preprocess.task_prompts import v2_medical_task_descriptions as medical_task_descriptions
from ct2echo.preprocess.task_prompts import label_query_func, radio_label_query_func


class LearnableTaskEmbedding(nn.Module):
    """
    Learnable task embedding layer that replaces pre-defined text embeddings.
    
    This creates a lookup table of learnable embeddings for each medical task,
    which will be optimized during training rather than using fixed text-based embeddings.
    
    Args:
        task_names: List of task names
        embedding_dim: Dimension of task embeddings (should match pre-defined embedding size)
        init_std: Standard deviation for initialization
    """
    
    def __init__(
        self, 
        task_names: List[str], 
        embedding_dim: int = 1024,  # Default to gte_large size
        init_std: float = 0.02
    ):
        super().__init__()
        
        self.task_names = task_names
        self.embedding_dim = embedding_dim
        self.num_tasks = len(task_names)
        
        # Create task name to index mapping
        self.task_to_idx = {task: idx for idx, task in enumerate(task_names)}
        self.idx_to_task = {idx: task for task, idx in self.task_to_idx.items()}
        
        # Learnable embedding table
        self.task_embeddings = nn.Embedding(self.num_tasks, embedding_dim)
        
        # Initialize embeddings with small random values
        with torch.no_grad():
            nn.init.normal_(self.task_embeddings.weight, mean=0.0, std=init_std)
        
        logger.info(f"🎯 Initialized LearnableTaskEmbedding:")
        logger.info(f"   📋 {self.num_tasks} tasks: {task_names}")
        logger.info(f"   📏 Embedding dimension: {embedding_dim}")
        logger.info(f"   🎲 Initialization std: {init_std}")
    
    def forward(self, task_names: List[str]) -> torch.Tensor:
        """
        Get embeddings for given task names
        
        Args:
            task_names: List of task names to get embeddings for
            
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        # Convert task names to indices
        task_indices = []
        for task_name in task_names:
            if task_name not in self.task_to_idx:
                raise ValueError(f"Unknown task: {task_name}. Available tasks: {self.task_names}")
            task_indices.append(self.task_to_idx[task_name])
        
        task_indices = torch.tensor(task_indices, device=self.task_embeddings.weight.device)
        
        # Get embeddings
        embeddings = self.task_embeddings(task_indices)
        
        return embeddings
    
    def get_task_embedding(self, task_name: str) -> torch.Tensor:
        """
        Get embedding for a single task
        
        Args:
            task_name: Name of the task
            
        Returns:
            Tensor of shape [embedding_dim]
        """
        if task_name not in self.task_to_idx:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {self.task_names}")
        
        task_idx = self.task_to_idx[task_name]
        task_tensor = torch.tensor([task_idx], device=self.task_embeddings.weight.device)
        embedding = self.task_embeddings(task_tensor).squeeze(0)
        
        return embedding
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get all task embeddings
        
        Returns:
            Tensor of shape [num_tasks, embedding_dim]
        """
        all_indices = torch.arange(self.num_tasks, device=self.task_embeddings.weight.device)
        return self.task_embeddings(all_indices)
    
    def save_embeddings(self, filepath: str):
        """Save learned task embeddings to file"""
        embeddings_dict = {
            'task_names': self.task_names,
            'embeddings': self.task_embeddings.weight.data.clone(),
            'task_to_idx': self.task_to_idx,
            'embedding_dim': self.embedding_dim
        }
        torch.save(embeddings_dict, filepath)
        logger.info(f"💾 Saved learnable task embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load learned task embeddings from file"""
        embeddings_dict = torch.load(filepath, map_location='cpu')
        
        # Verify compatibility
        if embeddings_dict['task_names'] != self.task_names:
            raise ValueError("Task names in saved file don't match current task names")
        if embeddings_dict['embedding_dim'] != self.embedding_dim:
            raise ValueError("Embedding dimension in saved file doesn't match current dimension")
        
        # Load the embeddings
        self.task_embeddings.weight.data = embeddings_dict['embeddings']
        logger.info(f"📂 Loaded learnable task embeddings from {filepath}")


def create_learnable_task_embeddings(
    embedding_dim: int = 1024,
    init_std: float = 0.02,
    device: str = "cpu",
    use_radio_labels: bool = False,
) -> LearnableTaskEmbedding:
    """
    Create learnable task embeddings for all medical tasks
    
    Args:
        embedding_dim: Dimension of task embeddings
        init_std: Standard deviation for initialization
        device: Device to place embeddings on
        
    Returns:
        LearnableTaskEmbedding instance
    """
    if use_radio_labels:
        label_source = {**label_query_func, **radio_label_query_func}
    else:
        label_source = label_query_func

    task_names = list(label_source.keys())
    
    learnable_embeddings = LearnableTaskEmbedding(
        task_names=task_names,
        embedding_dim=embedding_dim,
        init_std=init_std
    )
    
    learnable_embeddings = learnable_embeddings.to(device)
    
    label_mode = "radio+echo" if use_radio_labels else "echo"
    logger.info(f"✨ Created learnable task embeddings for ablation study ({label_mode} labels)")
    logger.info(f"   🎯 Tasks: {task_names}")
    logger.info(f"   📱 Device: {device}")
    
    return learnable_embeddings


def create_learnable_task_embedding_dict(
    learnable_embedding: LearnableTaskEmbedding,
    batch_size: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Create a dictionary of task embeddings compatible with existing data loading
    
    Args:
        learnable_embedding: LearnableTaskEmbedding instance
        batch_size: Batch size for each embedding (to match dataloader format)
        
    Returns:
        Dictionary mapping task names to embeddings [batch_size, embedding_dim]
    """
    task_embeddings = {}
    
    for task_name in learnable_embedding.task_names:
        # Get single embedding and expand to batch size
        single_embedding = learnable_embedding.get_task_embedding(task_name)
        batched_embedding = single_embedding.unsqueeze(0).expand(batch_size, -1)
        task_embeddings[task_name] = batched_embedding
    
    return task_embeddings
