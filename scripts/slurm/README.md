# Future Slurm execution

GPU-backed inference, evaluation, and possible fine-tuning will run as batch jobs rather than on
cluster login nodes. Once a job script exists, it will be submitted with `sbatch <script>` and its
queued or running state monitored with `squeue` (using the appropriate user or job filter for the
site). Job output and exit status must be inspected before treating a run as successful.

There is intentionally no runnable GPU job script yet. Before writing one, obtain the following
NUS SoC cluster-specific details from current documentation or administrators:

- permitted partition or queue names and available GPU resource requests;
- required account, project, or quality-of-service fields;
- available GPU types and the correct generic-resource syntax;
- supported CUDA, driver, compiler, and environment-module combinations;
- approved Python/uv environment and package-cache locations;
- project, dataset, scratch, and artifact filesystem locations and quotas;
- wall-time, CPU, memory, GPU, job-array, and concurrency limits;
- login-node, data-transfer, internet-access, and storage-retention policies;
- monitoring, cancellation, and accounting commands required by the site.

Do not copy guessed values into a job script. A future script should fail clearly, log its software
and hardware environment, use explicit resource requests, and keep large data and outputs outside
the Git repository.
