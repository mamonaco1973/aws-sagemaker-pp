"""Where the demo finds its infrastructure: Terraform outputs, never pasted ARNs.

Lookup order:
  1. the path in $SAGEMAKER_DEMO_CONFIG, if set
  2. demo-config.json at the project root -- written by apply.sh on your
     machine, and by the lifecycle script on the notebook instance
  3. `terraform output -json demo_config` in 01-infrastructure
"""

import json
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "demo-config.json")
TERRAFORM_DIR = os.path.join(PROJECT_ROOT, "01-infrastructure")

REQUIRED = (
    "region", "prefix", "bucket", "execution_role_arn", "sklearn_image_uri",
    "training_instance_type", "training_max_runtime_seconds",
    "serverless_memory_mb", "serverless_max_concurrency",
)


class ConfigError(RuntimeError):
    pass


class Config(dict):
    """The demo_config Terraform output, with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    # Every name the Python side creates starts with the prefix, which is
    # what lets cleanup find them again after a crash or a lost state file.
    @property
    def endpoint_name(self):
        return "%s-endpoint" % self.prefix

    @property
    def endpoint_log_group(self):
        return "/aws/sagemaker/Endpoints/%s" % self.endpoint_name


def _from_terraform():
    try:
        out = subprocess.run(
            ["terraform", "-chdir=%s" % TERRAFORM_DIR, "output", "-json", "demo_config"],
            capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return json.loads(out.stdout)


def load(path=None):
    path = path or os.environ.get("SAGEMAKER_DEMO_CONFIG")
    if path:
        source = path
        with open(path) as f:
            values = json.load(f)
    elif os.path.exists(CONFIG_FILE):
        source = CONFIG_FILE
        with open(CONFIG_FILE) as f:
            values = json.load(f)
    else:
        source = "terraform output (01-infrastructure)"
        values = _from_terraform()
        if values is None:
            raise ConfigError(
                "No demo configuration found. Run ./apply.sh first (it writes "
                "demo-config.json), or set SAGEMAKER_DEMO_CONFIG to its path.")

    missing = [k for k in REQUIRED if k not in values]
    if missing:
        raise ConfigError("%s is missing %s; re-run ./apply.sh to regenerate it."
                          % (source, ", ".join(missing)))
    config = Config(values)
    config["source"] = source
    return config
