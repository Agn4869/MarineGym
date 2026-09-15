from setuptools import find_packages, setup

setup(
    name="marinegym",
    version="1.1.0.dev0",
    author="chusg@zju.edu.cn",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    python_requires=">=3.11,<3.12",
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "imageio",
        "plotly",
        "einops",
        "pandas",
        "moviepy",
        "av",
        # Isaac Sim 5.0 provides PyTorch 2.7. Keep the TorchRL and TensorDict
        # versions aligned without asking pip to replace Isaac Sim's PyTorch.
        "torchrl>=0.8.1,<0.9.0",
        "tensordict>=0.8.3,<0.9.0",
    ],
)
