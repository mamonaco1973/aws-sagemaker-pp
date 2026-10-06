#!/usr/bin/env python3
"""Command-line version of the notebook: the same steps, the same code.

    python demo.py train            generate + upload data, run the training job
    python demo.py evaluate         held-out metrics of the latest job
    python demo.py artifact         download model.tar.gz, list it, show a tree
    python demo.py deploy           serve the latest model on Serverless Inference
    python demo.py predict          send the sample readings to the endpoint
    python demo.py predict --json '{"temperature": 80, "vibration": 4.5, "operating_hours": 9000}'
    python demo.py status           what exists right now
    python demo.py cleanup          delete endpoint, config, model (destroy.sh adds --purge-data)
    python demo.py notebook-stop    stop the notebook instance; files are kept

Configuration comes from demo-config.json (written by apply.sh) or the
Terraform outputs. Credentials come from the default AWS credential chain.
"""

import argparse
import json
import os
import sys

from sagemaker_demo import cleanup as cleanup_mod
from sagemaker_demo import config as config_mod
from sagemaker_demo import workflow
from sagemaker_demo.workflow import DemoError, log


def _job(cfg, args):
    return args.job or workflow.latest_completed_job(cfg)


def cmd_config(cfg, args):
    print(json.dumps(cfg, indent=2))


def cmd_train(cfg, args):
    paths = workflow.make_dataset()
    data_uris = workflow.upload_dataset(cfg, paths)
    source_uri = workflow.package_source(cfg)
    name = workflow.start_training(cfg, source_uri, data_uris)
    if args.no_wait:
        log("not waiting; follow it with: python demo.py wait %s" % name)
        return
    workflow.wait_for_training(cfg, name)
    print(json.dumps(workflow.training_summary(cfg, name), indent=2))
    log("the training instance is gone; the model is the S3 object above")


def cmd_wait(cfg, args):
    name = args.job or (workflow.list_training_jobs(cfg) or [{}])[0].get("TrainingJobName")
    if not name:
        raise DemoError("No training job found for prefix %s." % cfg.prefix)
    workflow.wait_for_training(cfg, name)
    print(json.dumps(workflow.training_summary(cfg, name), indent=2))


def cmd_evaluate(cfg, args):
    name = _job(cfg, args)
    summary = workflow.training_summary(cfg, name)
    _, extracted = workflow.download_artifact(cfg, name)
    with open(os.path.join(extracted, "evaluation.json")) as f:
        evaluation = json.load(f)
    log("training job %s (%s billable seconds on %s)"
        % (name, summary["billable_seconds"], summary["instance_type"]))
    print("  held-out rows : %d" % evaluation["test_rows"])
    for metric in ("accuracy", "precision", "recall"):
        print("  %-14s: %.3f" % (metric, evaluation[metric]))
    labels = evaluation["confusion_matrix"]["labels"]
    matrix = evaluation["confusion_matrix"]["matrix"]
    print("  confusion matrix (rows = actual, columns = predicted)")
    print("  %-10s %9s %9s" % ("", labels[0], labels[1]))
    for label, row in zip(labels, matrix):
        print("  %-10s %9d %9d" % (label, row[0], row[1]))


def cmd_artifact(cfg, args):
    name = _job(cfg, args)
    archive, extracted = workflow.download_artifact(cfg, name)
    log("s3 artifact: %s" % workflow.training_summary(cfg, name)["model_artifact"])
    log("downloaded to %s" % os.path.relpath(archive))
    for member, size in workflow.archive_listing(archive):
        print("  %8d  %s" % (size, member))
    with open(os.path.join(extracted, "metadata.json")) as f:
        print(json.dumps(json.load(f), indent=2))
    try:
        import joblib
        from sklearn.tree import export_text
    except ImportError:
        log("install requirements.txt to load model.joblib and print a tree")
        return
    model = joblib.load(os.path.join(extracted, "model.joblib"))
    log("tree 0 of %d, top three levels of splits:" % len(model.estimators_))
    print(export_text(model.estimators_[0], feature_names=list(model.feature_names_in_),
                      max_depth=2))


def cmd_deploy(cfg, args):
    name = _job(cfg, args)
    endpoint = workflow.deploy(cfg, name)
    log("endpoint %s is %s" % (endpoint["EndpointName"], endpoint["EndpointStatus"]))


def cmd_predict(cfg, args):
    if args.json:
        try:
            payload = json.loads(args.json)
        except ValueError as e:
            raise DemoError("--json is not valid JSON: %s" % e)
        body, seconds = workflow.invoke(cfg, payload)
        print(json.dumps(body, indent=2))
        log("%.2fs round trip" % seconds)
        return
    # The sample readings, and what the model said about them when it was
    # trained, both travel inside the artifact.
    _, extracted = workflow.download_artifact(cfg, _job(cfg, args))
    samples = workflow.load_samples(extracted)
    for i, sample in enumerate(samples):
        body, seconds = workflow.invoke(cfg, sample["reading"])
        result = body["predictions"][0]
        note = " (includes cold start)" if i == 0 and seconds > 3 else ""
        print("%-13s %s -> %-8s p(failure)=%.3f  [at training: %.3f]  %.2fs%s"
              % (sample["name"], json.dumps(sample["reading"]), result["label"],
                 result["failure_probability"], sample["expected_failure_probability"],
                 seconds, note))


def cmd_status(cfg, args):
    log("config from %s" % cfg.source)
    jobs = workflow.list_training_jobs(cfg)
    for job in jobs[:5]:
        print("  training job   %-45s %s" % (job["TrainingJobName"], job["TrainingJobStatus"]))
    if not jobs:
        print("  training job   (none)")
    found = cleanup_mod.find_resources(cfg)
    endpoint = workflow.describe_endpoint(cfg)
    print("  endpoint       %-45s %s" % (cfg.endpoint_name,
                                         endpoint["EndpointStatus"] if endpoint else "(none)"))
    for name in found["endpoint_configs"]:
        print("  endpoint conf  %s" % name)
    for name in found["models"]:
        print("  model          %s" % name)
    notebook = cfg.get("notebook_name")
    if notebook:
        sm = workflow.clients(cfg)["sagemaker"]
        nb = workflow._exists(sm.describe_notebook_instance, NotebookInstanceName=notebook)
        print("  notebook       %-45s %s" % (notebook, nb["NotebookInstanceStatus"] if nb else "(none)"))


def cmd_cleanup(cfg, args):
    cleanup_mod.cleanup(cfg, purge_data=args.purge_data, delete_logs=not args.keep_logs)
    log("SageMaker resources created by this demo are gone. The notebook is "
        "Terraform's: stop it with 'python demo.py notebook-stop', or remove "
        "everything with ./destroy.sh")


def cmd_notebook_stop(cfg, args):
    cleanup_mod.stop_notebook(cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="path to demo-config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="print the configuration in use")
    p = sub.add_parser("train", help="generate + upload data, run the training job")
    p.add_argument("--no-wait", action="store_true")
    for name, text in (("wait", "wait for a training job"),
                       ("evaluate", "held-out metrics"),
                       ("artifact", "download and inspect model.tar.gz"),
                       ("deploy", "deploy to Serverless Inference")):
        p = sub.add_parser(name, help=text)
        p.add_argument("job", nargs="?", help="training job name (default: latest completed)")
    p = sub.add_parser("predict", help="invoke the endpoint")
    p.add_argument("--json", help="one reading or {\"instances\": [...]} as JSON")
    p.add_argument("--job", help="training job whose samples to send (default: latest)")
    sub.add_parser("status", help="list the demo's resources")
    p = sub.add_parser("cleanup", help="delete endpoint, configs and models")
    p.add_argument("--purge-data", action="store_true",
                   help="also delete the demo's S3 prefixes (data, code, artifacts, state)")
    p.add_argument("--keep-logs", action="store_true")
    sub.add_parser("notebook-stop", help="stop the notebook instance")

    args = parser.parse_args(argv)
    from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError
    try:
        cfg = config_mod.load(args.config)
        globals()["cmd_" + args.command.replace("-", "_")](cfg, args)
    except (DemoError, config_mod.ConfigError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    except (NoCredentialsError, NoRegionError) as e:
        print("ERROR: %s. Configure the AWS CLI (aws configure, or AWS_PROFILE)." % e,
              file=sys.stderr)
        return 1
    except ClientError as e:
        error = e.response["Error"]
        hint = ""
        if error["Code"] in ("AccessDenied", "AccessDeniedException"):
            hint = (" The notebook role only covers names starting with %s-; from a "
                    "laptop, your own credentials need SageMaker, S3 and IAM PassRole."
                    % cfg.prefix)
        elif error["Code"] == "ResourceLimitExceeded":
            hint = " Request a quota increase in Service Quotas (SageMaker)."
        print("ERROR: %s: %s.%s" % (error["Code"], error["Message"], hint), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
