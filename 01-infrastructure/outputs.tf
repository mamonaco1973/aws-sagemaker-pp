# The single source of configuration for demo.py and the notebook.
output "demo_config" {
  value = local.demo_config
}

output "region" { value = var.region }
output "bucket" { value = aws_s3_bucket.demo.id }
output "notebook_name" { value = aws_sagemaker_notebook_instance.demo.name }
output "execution_role_arn" { value = aws_iam_role.execution.arn }
output "notebook_role_arn" { value = aws_iam_role.notebook.arn }
output "sklearn_image_uri" { value = local.sklearn_image_uri }

output "notebook_console_url" {
  value = "https://${var.region}.console.aws.amazon.com/sagemaker/home?region=${var.region}#/notebook-instances/${aws_sagemaker_notebook_instance.demo.name}"
}
