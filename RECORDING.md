# Recording Walkthrough (5-8 minutes)

A shot list for a technical video of this project. Each scene shows its
on-screen time and the waits to **cut**. The real elapsed time is about 25
to 35 minutes; the edited video is about 7.

**Before recording:** run `./apply.sh` and open the notebook once, so the
kernel is already built (`kernel-setup.log` says `KERNEL READY`). Run
`python demo.py cleanup` so no endpoint exists. Set JupyterLab's font size
up one step.

---

### 1. The idea (0:00 - 0:40)

**Screen:** the README's architecture diagram.

> "This is conventional machine learning on SageMaker: a random forest, not
> an LLM. We'll train it on a machine that disappears when training ends,
> save it as one file in S3, and then serve predictions from that file in a
> different container, with no retraining. Terraform built the bucket, two IAM
> roles and this notebook. Python creates everything SageMaker runs."

### 2. Infrastructure (0:40 - 1:20)

**Screen:** terminal, end of `./apply.sh` output (the quick-start banner),
then `01-infrastructure/iam.tf` scrolled to the notebook role.

> "Two roles. The notebook can only touch resources whose names start with
> our prefix, and the only role it can hand to SageMaker is the execution
> role, which can read the data and write the model, and nothing else."

**Cut:** `terraform apply` itself (**5-7 min**). Show only the banner.

### 3. Data (1:20 - 2:10)

**Screen:** notebook §1 setup cell output (versions, config, caller ARN), then
§2 cells: table head, the scatter plot.

> "No pasted ARNs or keys: the config came from Terraform, and the
> credentials are the notebook's role. The data is synthetic, with three
> sensor readings and a label drawn from a probability, so it's noisy on
> purpose. See how the classes overlap? No model can be perfect here."

### 4. Training (2:10 - 3:20)

**Screen:** §3 upload output, §4 the printed `CreateTrainingJob` request
(scroll to `HyperParameters`, `InputDataConfig`, `StoppingCondition`), then
the status lines ticking through `Starting / Downloading / Training /
Uploading / Completed`.

> "Script mode: the AWS scikit-learn image downloads train.py and runs it.
> Two channels, train and test, so the test rows never reach fit(). And a
> hard 15-minute cap on billed runtime."

**Cut:** the wait between `Starting` and `Completed` (**3-5 min**). Keep the
first and last status lines.

> "Completed. We were billed for the seconds shown, on an m5.large, and that
> machine no longer exists."

### 5. Evaluation (3:20 - 3:50)

**Screen:** §5 metrics and confusion matrix.

> "85% accuracy against a 71% always-healthy baseline. It catches about two
> thirds of failures. Not perfect, and it shouldn't be, with labels this
> noisy."

### 6. The artifact (3:50 - 4:50)

**Screen:** §6 S3 listing, §7 archive listing and model summary, §8 the tree
plot.

> "Everything that's left of training: a one-megabyte model.tar.gz. Inside,
> model.joblib is 200 fitted trees: thresholds, splits, leaf counts. It's a
> pickle, so it needs the same scikit-learn to load, which is why the
> endpoint uses the same image. Here's the top of one tree: temperature
> first, then vibration."

### 7. Serverless deployment (4:50 - 5:40)

**Screen:** §9 the `model_fn` source, then the three printed requests,
highlighting `ServerlessConfig`.

> "SageMaker extracts the archive to /opt/ml/model and calls model_fn, and
> this joblib.load is the only load in the serving process. Serverless,
> 2 GB, and no provisioned concurrency, so nothing runs between requests."

**Cut:** endpoint `Creating` to `InService` (**3-6 min**).

### 8. Predictions (5:40 - 6:50)

**Screen:** §10 sample table (point at the `seconds` column), batch output,
bad-request output, then the CloudWatch `joblib.load` line.

> "Three readings the model has never seen. The endpoint agrees with the
> local copy to four decimals, because it's the same file. The first call
> paid for a cold start; the next ones didn't. Bad input gets a 400 with a
> reason. And the endpoint's log shows the model loaded once, in one
> process. No training data, no retraining."

### 9. Cleanup (6:50 - 7:30)

**Screen:** §11 output; terminal `./destroy.sh` start and final line.

> "Cleanup deletes the endpoint first, then its config and model. Then
> destroy.sh hands over to Terraform for the notebook, the roles and the
> bucket. It never force-empties a bucket that has someone else's files in
> it."

**Cut:** endpoint deletion (**1-2 min**) and `terraform destroy` (**about
5 min**, mostly stopping the notebook).

---

### Timing summary

| Wait | Typical | Edit |
|---|---|---|
| `./apply.sh` (notebook to InService) | 5-7 min | cut, show banner |
| Kernel build on first start | ~3 min | do before recording |
| Training job | 3-5 min | cut middle |
| Endpoint creation | 3-6 min | cut middle |
| First invocation (cold start) | seconds | **keep**, it's a teaching point |
| Endpoint deletion | 1-2 min | cut |
| `terraform destroy` | ~5 min | cut |

The waits are estimates; they have not been timed on a live run yet.
