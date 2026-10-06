"""Orchestration for the SageMaker random forest demo.

Shared by the notebook and by demo.py, so both drive SageMaker through the
same boto3 calls. Nothing here holds credentials: boto3 uses the default
credential chain (the notebook's execution role on the notebook instance,
your AWS CLI profile on a laptop).
"""
