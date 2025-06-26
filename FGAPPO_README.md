# FGAPPO: Feature Graph-Augmented PPO

## Overview

FGAPPO (Feature Graph-Augmented PPO) is a novel CTDE-compliant multi-agent reinforcement learning algorithm that combines diffusion-based action generation with feature-level graph neural networks. Unlike traditional graph-based MARL approaches that operate on agent-level graphs, FGAPPO treats each scalar feature within an agent's observation as a graph node, maintaining full CTDE compliance while leveraging graph structure to model relationships between observation features.

## Key Features

### 🎯 **CTDE Compliance**
- **Training**: Centralized training with access to global information
- **Execution**: Fully decentralized - each agent processes only their own observation features
- **No inter-agent communication** required during execution

### 🔗 **Feature-Level Graph Processing**
- Each scalar observation feature becomes a graph node
- Graph size = observation dimension (not number of agents)
- Models relationships between different aspects of observations

### 🌊 **Diffusion-Enhanced Action Generation**
- Extends DAPPO with feature graph structure
- Coupled diffusion process with feature graph enhancement
- Maintains GenPO properties (compression loss, mixing dynamics)

### 🎛️ **Flexible Feature Adjacency Types**
- **Spatial**: Connect nearby features in observation vector
- **Correlation**: Connect features based on statistical correlation
- **Learned**: Neural network learns feature relationships
- **Semantic**: Domain-specific feature groupings
- **Fully Connected**: All features connected

## Architecture

```
Individual Agent Observation [obs_dim]
           ↓
Feature Graph Builder (CTDE compliant)
           ↓
Adjacency Matrix [obs_dim × obs_dim]
           ↓
Feature Graph Layers (Conv/Attention)
           ↓
Feature Aggregation
           ↓
Diffusion Policy Network
           ↓
Action Generation [action_dim]
```

## Installation

Ensure FGAPPO is integrated into the HARL framework:

1. **Core Implementation**:
   - `harl/models/policy_models/feature_graph_diffusion_policy.py`
   - `harl/algorithms/actors/fgappo.py`

2. **Configuration**:
   - `harl/configs/algos_cfgs/fgappo.yaml`

3. **Registration**:
   - Added to `harl/algorithms/actors/__init__.py`
   - Added to `harl/runners/__init__.py`
   - Added to `examples/train.py`

## Usage

### Basic Training

```bash
python examples/train.py --algo fgappo --env pettingzoo_mpe --scenario simple_spread
```

### Advanced Configuration

```bash
python examples/train.py \
    --algo fgappo \
    --env pettingzoo_mpe \
    --scenario simple_spread \
    --feature_adjacency_type learned \
    --use_feature_attention True \
    --num_feature_graph_layers 3 \
    --feature_graph_hidden_dim 64
```

### Using the Example Script

```bash
# Spatial adjacency with convolution
python examples/fgappo_example.py --feature_adjacency_type spatial

# Learned adjacency with attention
python examples/fgappo_example.py \
    --feature_adjacency_type learned \
    --use_feature_attention True

# Debug feature graph structure
python examples/fgappo_example.py \
    --feature_adjacency_type correlation \
    --debug_feature_graph True
```

## Configuration Parameters

### Feature Graph Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_adjacency_type` | str | `"spatial"` | Type of feature adjacency: spatial, correlation, learned, semantic, fully_connected |
| `use_feature_attention` | bool | `False` | Use attention mechanism instead of convolution |
| `num_feature_graph_layers` | int | `2` | Number of feature graph processing layers |
| `feature_graph_hidden_dim` | int | `32` | Hidden dimension for feature graph layers |
| `feature_embed_dim` | int | `16` | Embedding dimension for scalar features |
| `use_feature_normalization` | bool | `True` | Apply layer normalization to feature processing |

### Diffusion Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_diffusion_steps` | int | `10` | Number of diffusion denoising steps |
| `beta_start` | float | `0.0001` | Starting beta for diffusion schedule |
| `beta_end` | float | `0.02` | Ending beta for diffusion schedule |
| `mixing_p` | float | `0.5` | Mixing probability for GenPO |
| `lambda_ent` | float | `0.1` | Entropy loss coefficient |
| `nu_compression` | float | `0.01` | Compression loss coefficient |

## Feature Adjacency Types

### 1. Spatial Adjacency
- Connects nearby features in the observation vector
- Good for structured observations (e.g., image-like data)
- Weight inversely proportional to feature distance

### 2. Correlation Adjacency  
- Connects features based on statistical correlation across batch
- Adapts to data distribution
- Threshold-based connection (default: 0.3)

### 3. Learned Adjacency
- Neural network learns pairwise feature relationships
- Most flexible but requires more computation
- Includes regularization for sparsity and symmetry

### 4. Semantic Adjacency
- Domain-specific feature groupings
- Example: position features connected to velocity features
- Requires domain knowledge for optimal performance

### 5. Fully Connected
- All features connected to all other features
- Maximum information flow but higher computation cost
- Good baseline for comparison

## Performance Considerations

### Memory Optimization
- Reduced hidden dimensions: `[64, 64]` instead of `[256, 256]`
- Fewer rollout threads: `2` instead of `32`
- Shorter diffusion steps: `10` instead of `50`

### Computational Efficiency
- Feature graph size scales with observation dimension, not agent count
- Parallel processing of feature relationships
- Optional gradient checkpointing for large graphs

### Numerical Stability
- Temperature scaling in attention mechanisms
- Gradient clipping for feature graph layers
- Fallback mechanisms for degenerate cases

## Comparison with Agent-Level Graphs

| Aspect | FGAPPO (Feature-Level) | GAPPO (Agent-Level) |
|--------|------------------------|---------------------|
| **CTDE Compliance** | ✅ Fully compliant | ❌ Requires agent observations |
| **Graph Size** | Observation dimension | Number of agents |
| **Scalability** | Scales with obs complexity | Scales with agent count |
| **Information** | Intra-agent feature relationships | Inter-agent relationships |
| **Execution** | Fully decentralized | Semi-centralized |

## Advanced Features

### Feature Graph Debugging

Enable debugging to visualize feature relationships:

```python
# In your training script
runner.debug_feature_graph(obs_batch, verbose=True)
```

Output provides:
- Adjacency matrix statistics
- Connection sparsity
- Feature degree distribution
- Relationship visualization

### Custom Feature Adjacency

Implement custom adjacency types by extending `FeatureGraphBuilder`:

```python
class CustomFeatureGraphBuilder(FeatureGraphBuilder):
    def _custom_adjacency(self, obs):
        # Your custom logic here
        return adjacency_matrix
```

## Hyperparameter Tuning

### For Different Observation Types

**High-dimensional observations** (e.g., images):
- `feature_adjacency_type: "spatial"`
- `num_feature_graph_layers: 3-4`
- `feature_graph_hidden_dim: 64-128`

**Low-dimensional observations** (e.g., state vectors):
- `feature_adjacency_type: "correlation"` or `"learned"`
- `num_feature_graph_layers: 2-3`
- `feature_graph_hidden_dim: 32-64`

**Domain-specific observations**:
- `feature_adjacency_type: "semantic"`
- Custom adjacency implementation
- Domain knowledge for feature groupings

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```yaml
   num_rollout_threads: 1
   feature_graph_hidden_dim: 16
   actor_hidden_sizes: [32, 32]
   ```

2. **NaN Actions**
   - Check attention temperature settings
   - Verify adjacency matrix validity
   - Enable gradient clipping

3. **Slow Training**
   - Reduce `num_diffusion_steps`
   - Use `feature_adjacency_type: "spatial"`
   - Disable feature attention

4. **Poor Performance**
   - Try different adjacency types
   - Adjust regularization weights
   - Tune feature graph depth

## Research Applications

### Suitable Environments
- **Continuous control**: Robot swarms, vehicle coordination
- **Discrete action spaces**: Multi-agent navigation, resource allocation
- **Mixed domains**: Games with both spatial and abstract features

### Experimental Directions
- Compare adjacency types across different environments
- Analyze feature importance through graph structure
- Study scalability with varying observation dimensions
- Investigate transfer learning between similar domains

## Citation

If you use FGAPPO in your research, please cite:

```bibtex
@misc{fgappo2024,
  title={FGAPPO: Feature Graph-Augmented PPO for CTDE-Compliant Multi-Agent Reinforcement Learning},
  author={Your Name},
  year={2024},
  howpublished={HARL Framework Implementation}
}
```

## Contributing

To contribute to FGAPPO development:

1. Follow the existing code structure in HARL
2. Add comprehensive tests for new features
3. Update documentation for new parameters
4. Benchmark against existing methods

## License

FGAPPO follows the same license as the HARL framework.

---

**FGAPPO represents a novel approach to graph-based MARL that maintains CTDE compliance while leveraging the power of graph neural networks at the feature level. This enables more sophisticated observation processing while ensuring practical deployment in decentralized settings.** 