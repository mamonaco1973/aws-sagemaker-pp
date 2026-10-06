locals {
  name = var.prefix

  # ----------------------------------------------------------------------------
  # The scikit-learn container, used for BOTH training and inference
  # ----------------------------------------------------------------------------
  # One image for both sides is the simplest way to guarantee that the
  # scikit-learn which unpickles model.joblib is the one that pickled it.
  # 1.4-2-py312 = scikit-learn 1.4.2 on Python 3.12, CPU only. The image
  # supports SageMaker training, real-time and Serverless Inference.
  #
  # Registry accounts are AWS-owned, one per Region, taken from the SageMaker
  # Python SDK's image_uri_config/sklearn.json (version 1.4-2-py312). Only
  # commercial-partition Regions are listed.
  sklearn_framework_version = "1.4-2-py312"
  sklearn_registry = {
    "af-south-1"     = "510948584623"
    "ap-east-1"      = "651117190479"
    "ap-northeast-1" = "354813040037"
    "ap-northeast-2" = "366743142698"
    "ap-northeast-3" = "867004704886"
    "ap-south-1"     = "720646828776"
    "ap-south-2"     = "628508329040"
    "ap-southeast-1" = "121021644041"
    "ap-southeast-2" = "783357654285"
    "ap-southeast-3" = "951798379941"
    "ap-southeast-4" = "106583098589"
    "ca-central-1"   = "341280168497"
    "ca-west-1"      = "190319476487"
    "eu-central-1"   = "492215442770"
    "eu-central-2"   = "680994064768"
    "eu-north-1"     = "662702820516"
    "eu-south-1"     = "978288397137"
    "eu-south-2"     = "104374241257"
    "eu-west-1"      = "141502667606"
    "eu-west-2"      = "764974769150"
    "eu-west-3"      = "659782779980"
    "il-central-1"   = "898809789911"
    "me-central-1"   = "272398656194"
    "me-south-1"     = "801668240914"
    "sa-east-1"      = "737474898029"
    "us-east-1"      = "683313688378"
    "us-east-2"      = "257758044811"
    "us-west-1"      = "746614075791"
    "us-west-2"      = "246618743249"
  }
  sklearn_repository = "sagemaker-scikit-learn"
  sklearn_image_uri = format("%s.dkr.ecr.%s.amazonaws.com/%s:%s-cpu-py3",
    local.sklearn_registry[var.region], var.region, local.sklearn_repository,
  local.sklearn_framework_version)

  # Notebook instances keep only /home/ec2-user/SageMaker across restarts;
  # the project and the kernel's virtualenv both live under it.
  notebook_project_dir = "/home/ec2-user/SageMaker/aws-sagemaker-pp"

  # The project files the notebook needs, uploaded by Terraform and copied
  # onto the instance by the lifecycle script. Paths are relative to the
  # project root, and the S3 keys mirror them under notebook-source/.
  notebook_files = toset(concat(
    ["demo.py", "requirements.txt", "README.md"],
    [for f in fileset("${path.module}/..", "ml/*.py") : f],
    [for f in fileset("${path.module}/..", "ml/requirements.txt") : f],
    [for f in fileset("${path.module}/..", "sagemaker_demo/*.py") : f],
    [for f in fileset("${path.module}/..", "notebook/*.ipynb") : f],
  ))
}
