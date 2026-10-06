# ==============================================================================
# Demo Bucket -- data, training source, model artifacts, and run state
# ==============================================================================
# Layout (prefixes):
#   notebook-source/   project files for the notebook (Terraform-owned)
#   data/train/, data/test/   CSV channels           (written by Python)
#   code/              training source tarballs      (written by Python)
#   training-output/   <job>/output/model.tar.gz     (written by SageMaker)
#   state/             names of Python-created resources, for cleanup
#
# force_destroy is deliberately false. `demo.py cleanup --purge-data` removes
# the Python-written prefixes above and nothing else; if anything unrelated
# was put in this bucket, terraform destroy stops on it instead of silently
# deleting it.

resource "aws_s3_bucket" "demo" {
  # bucket_prefix, not bucket: S3 names are global, and a fixed name would
  # collide the moment this is deployed from a second account.
  bucket_prefix = "${local.name}-"
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "demo" {
  bucket                  = aws_s3_bucket.demo.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "demo" {
  bucket = aws_s3_bucket.demo.id
  rule { object_ownership = "BucketOwnerEnforced" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "demo" {
  bucket = aws_s3_bucket.demo.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

# Refuse plain-HTTP access. Every client here (boto3, the AWS CLI, SageMaker)
# already uses TLS; this makes it a guarantee rather than a habit.
resource "aws_s3_bucket_policy" "demo" {
  bucket = aws_s3_bucket.demo.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.demo.arn, "${aws_s3_bucket.demo.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.demo]
}

# The project files the notebook instance copies at start-up. etag makes an
# edited file re-upload on the next apply.
resource "aws_s3_object" "notebook_source" {
  for_each = local.notebook_files
  bucket   = aws_s3_bucket.demo.id
  key      = "notebook-source/${each.value}"
  source   = "${path.module}/../${each.value}"
  etag     = filemd5("${path.module}/../${each.value}")
}
