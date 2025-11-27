# SFC RL: Service Function Chaining Virtual Network Embedding with Reinforcement Learning

A modular, research-grade framework for Service Function Chaining (SFC) virtual network embedding using Reinforcement Learning (DQN) with baseline comparisons (Random and Exhaustive Search).

## Features

- **Modular Architecture**: Pluggable components for state encoders, action spaces, reward functions, and QoE models
- **DQN Implementation**: Deep Q-Network with configurable features (double-DQN, dueling architecture, n-step returns)
- **Baseline Policies**: Random and Exhaustive/Violent search solvers for comparison
- **Config-Driven**: Hydra/OmegaConf configuration system for easy experimentation
- **Comprehensive Metrics**: Acceptance ratio, response time, QoE computation
- **Extensible**: Easy to add new state encoders, action spaces, reward functions, and QoE models

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- See `pyproject.toml` for full dependency list

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd Thesis-Smart-Slicing-rl-revised

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Training

Train a DQN agent on synthetic data:

```bash
python -m sfc_rl.cli
```

This uses the default configuration in `config/experiment.yaml`.

### Custom Configuration

Override configuration via command line:

```bash
python -m sfc_rl.cli \
    data=synthetic_small \
    train=medium \
    model=dqn_dueling \
    env=default \
    seed=42
```

### Run Tests

```bash
pytest tests/
```

## Project Structure

```
sfc_rl/
├── __init__.py
├── cli.py                 # CLI entry point
├── data/                  # Data schemas, generators, loaders
│   ├── schemas.py
│   ├── generators.py
│   ├── loaders.py
|   ├── lazy_loadr.py
│   └── dataset_provider.py
├── env/                   # Environment components
│   ├── sfc_env.py         # Main Gym-like environment
│   ├── state_encoders.py
│   ├── action_space.py
│   ├── reward.py
│   └── qoe.py
├── models/                # DQN models
│   ├── networks.py
│   ├── dqn.py
│   └── replay_buffer.py
├── baselines/             # Baseline policies
│   ├── random_policy.py
│   └── exhaustive_solver.py
├── train/                 # Training and evaluation
│   ├── trainer.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── plots.py
└── utils/                 # Utilities
    ├── seed.py
    ├── logging.py
    ├── registry.py
    └── serialization.py

config/
├── experiment.yaml        # Main experiment config
├── data/                  # Dataset configs
├── env/                   # Environment configs
├── model/                 # Model configs
├── train/                 # Training configs
└── eval/                  # Evaluation configs

tests/                     # Unit tests
```

## Configuration

The framework uses Hydra for configuration management. Key configuration files:

### Experiment Config (`config/experiment.yaml`)

```yaml
defaults:
  - data: synthetic_small
  - env: default
  - model: dqn_default
  - train: small
  - eval: default

seed: 42
output_dir: outputs
project_name: test_lazyloader #Training_dqn #data_generation_for_test
End_phase: 'All' #'data'
# These will be merged from data config
# p_net_setting: ...
# v_sim_setting: ...

```

### Data Config (`config/data/synthetic_small.yaml`)

```yaml
type: synthetic

p_net_setting:
  #dataset_dir : 'datasets/p_net/5-erdos_renyi-cpu-bandwidth-delay-seed_42' # IF to be load rather than generated, from where to be loaded
  #dataset_dir: "/home/mehrab/Workspaces/Thesis-Smart-Slicing-rl-revised/datasets/p_net/5nodes-erdos_renyi-cpu-bandwidth-delay-seed_42"
  dataset_dir: 'datasets/p_net/5nodes-erdos_renyi-cpu-bandwidth-delay-seed_42'
  topology:
    type: erdos_renyi # some common topologies e.g. generate_erdos_renyi_pn, fully connected, ...
    num_nodes: 5 #
    p: 1
  node_attrs_setting:
    - name: cpu
      distribution: uniform
      dtype: float
      low: 8
      high: 16
  link_attrs_setting:
    - name: bandwidth
      distribution: uniform
      dtype: float
      low: 60
      high: 100
    - name: delay
      distribution: uniform
      dtype: float
      low: 1
      high: 5
  output:
    if_save: True
    save_dir: datasets/p_net ## Location of saving the data

v_sim_setting:
  #dataset_dir : 'datasets/v_nets/200-[3-5]-1.0-cpu--seed_42' # IF to be load rather than generated, from where to be loaded
  #dataset_dir: 'datasets/v_nets/1000-[3-5]-1.0-cpu--seed_42'
  #daaset_dir: "/home/mehrab/Workspaces/Thesis-Smart-Slicing-rl-revised/datasets/v_nets/100sfcReqs-20groups-[5-5]lengths-arrival_rate1.0-cpu-bandwidth-latency-seed_42"
  #dataset_Dir: 'datasets/v_nets/100sfcReqs-2000groups-\[3-3\]lengths-arrival_rate1.0-cpu-bandwidth-latency-seed_42'
  dataset_Dir: 'datasets/v_nets/100sfcReqs-200groups-[3-3]lengths-arrival_rate1.0-cpu-bandwidth-latency-seed_42'
  num_v_nets: 100
  num_groups: 200
  cache_size: ${data.v_sim_setting.num_groups}
  cache_path: '.vnReqs_cache'
  sfc_len:
    low: 3
    high: 3
    distribution: uniform
  vnf_types:
    - fw
    - nat
    - ids
    - wanopt
  node_attrs_setting:
    - name: cpu
      distribution: uniform
      dtype: float
      low: 0
      high: 0
  qos_attrs_setting:
    - name: bandwidth
      distribution: uniform
      dtype: float
      low: 1
      high: 5
    - name: latency
      distribution: uniform
      dtype: float
      low: 8
      high: 15
  arrival_rate:
    distribution: poisson
    lam: 1.0
  lifetime:
    distribution: exponential
    scale: 100.0
  output:
    if_save: True
    save_dir: datasets/v_nets # Location of saving the data
    events_file_name: events.json
    setting_file_name: v_sim_setting.json
```

### Model Config (`config/model/dqn_default.yaml`)

```yaml
type: dqn #random #Violent
network:
  type: mlp
  hidden_sizes:
  - 256
  - 256
  activation: relu
optimizer:
  name: adam
  lr: 0.0003
dqn:
  gamma: 0.99
  buffer_size: 50
  batch_size: 16
  eps_start: 1.0
  eps_end: 0.05
  eps_decay_steps: 200000
  target_update: 10
  double: false
  dueling: false
  n_step: 1

```

## Usage Examples




### Setting Environment

```bash
state:
  encoder: NormalizedStateEncoder
  encoder_config:
    #include_pn_features: true
    #include_vn_features: true
    #include_embedding_status: true
    max_latency_budget: 100
action:
  type: node_selection
  mask_illegal: false
reward:
  name: "qoe_qos"
  penalty_failure: 10.0 # Penalty P for failed embeddings (Eq. 11 & paper text)
  penalty_exponent: false #true # Whether to apply the exponential penalty Pen_rn^sfcic even on success
  penalty_exp_factor: 1.0 # Tuning factor for the exponential penalty calculation (internal use)
  penalty_weight: 1.0 # Weight multiplier for the exponential penalty term
  qoe_weight: 1.0 # Weight multiplier for the QoE_sfcic term
qoe_model:
  name: "qoe_qos_paper"
  qoe_weight_delay: 0.5 # Weight (w_t) for the delay component (negative metric)
  qoe_weight_bandwidth: 0.5 # Weight (w_t) for the bandwidth component (positive metric)
  # --- IQX Hypothesis Parameters for Delay (Negative QoS Metric) ---
  alpha_n: -0.006931471805599453 # -ln(2) / 100
  beta_n: 0.0
  gamma_n: 4.0
  theta_n: 1.0
  # --- Weber-Fechner Law (WFL) Parameters for Bandwidth (Positive QoS Metric) ---
  alpha_p: 0.02484
  beta_p: 1.185
  gamma_p: 1.923
  theta_p: 1.0
```

### Training a DQN Agent

```bash
# Small experiment (quick test)
python -m sfc_rl.cli train=small

# Medium experiment
python -m sfc_rl.cli train=medium model=dqn_dueling

train: true #false
episodes: ${data.v_sim_setting.num_groups}
#episodes: 20
max_steps_per_episode: 200000
log_interval: 5
eval_every: 1
save_every: 10
```

### Evaluation

Evaluation runs automatically after training. To evaluate specific policies:

```yaml
# config/eval/default.yaml

enabled: false #true
policies:
  - name: dqn
    type: dqn
    checkpoint: null
  - name: random
    type: random
  - name: violent
    type: exhaustive
    timeout_seconds: 5.0
    max_embeddings: 1000
metrics:
  - acceptance_ratio
  - response_time
  - qoe
runs: ${data.v_sim_setting.num_groups}
seed: 42
report:
  csv: true
  tensorboard: false
  plots: true
output_dir: output_eval
```

## Extending the Framework

### Adding a New State Encoder

1. Create a new class inheriting from `StateEncoder` in `sfc_rl/env/state_encoders.py`
2. Implement `encode()` and `get_state_dim()` methods
3. Update environment configuration to use the new encoder

### Adding a New QoE Model

1. Create a new class inheriting from `QoEModel` in `sfc_rl/env/qoe.py`
2. Implement `compute()` method
3. Update environment configuration to use the new QoE model

### Adding a New Baseline

1. Create a new policy class in `sfc_rl/baselines/`
2. Implement `act()` or `solve()` method
3. Add to evaluation configuration

## Metrics

The framework computes the following metrics:

- **Acceptance Ratio**: Fraction of VN requests successfully embedded
- **Response Time**: Average time to process a request
- **QoE (Quality of Experience)**: Black-box QoE computation (configurable)
- **Mean Reward**: Average episode reward
- **Mean Episode Length**: Average number of steps per episode

## Outputs

Results are saved to `outputs/YYYYMMDD_HHMMSS/`:

- `config.yaml`: Experiment configuration
- `run.log`: Training logs
- `checkpoint_ep*.pt`: Model checkpoints (if DQN)
- `evaluation/`: Evaluation results
  - `evaluation_results.json`: Metrics in JSON format
  - `evaluation_results.csv`: Metrics in CSV format
  - `plots/metrics_comparison.png`: Comparison plots

## Testing

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_env.py
```

Run with coverage:

```bash
pytest --cov=sfc_rl tests/
```

## Development

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking (basic)

Format code:

```bash
black sfc_rl/ tests/
ruff check sfc_rl/ tests/
```

### Adding Tests

Add new tests in `tests/` following the existing test structure. Tests use pytest.

## License

MIT License

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{sfc_rl,
  title = {SFC RL: Service Function Chaining Virtual Network Embedding with Reinforcement Learning},
  author = {Your Name},
  year = {2024},
  url = {<repository-url>}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

This framework is designed for research in Service Function Chaining and Virtual Network Embedding using Reinforcement Learning.
