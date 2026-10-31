# Brev experiment data and credit protection

Prepared October 29, 2026. This is an independent design/review document. It does not configure a VM, credentials, a Windows service, a scheduled task, or a billing setting. The root task is handling instance setup and implementation separately; actual deployment and live verification must be recorded there.

## What must be protected

The current repository has `tools/vision_gpu_probe.py` and preparation guides. It does not yet have a validated VLM trainer, trainer checkpoint callback, or recovery-tested optimizer checkpoint. A GPU probe report can be backed up now. Automatic model checkpointing and resumable training below are proposed requirements for a future trainer, not capabilities supplied by the probe.

| Layer | Purpose | Status established by this review |
| --- | --- | --- |
| Workspace on the VM | Survive an ordinary supported stop/start | Documented provider behavior; verify actual mount/path |
| Atomic completed checkpoint | Survive a process dying during a save | Proposed trainer requirement |
| Verified copy outside Brev | Survive VM/disk deletion or capacity unavailability | Required design; no deployed periodic pull verified here |
| Local run deadline and explicit instance stop | Bound paid runtime | Proposed guard requirements; root implements separately |
| Restore exercise | Establish that copied artifacts are usable | Required acceptance check; not yet a model-resume result |

NVIDIA documents `/home/ubuntu/workspace` as surviving stop, while deletion removes instance data. A stopped instance can become inaccessible until matching provider/region GPU capacity returns. Therefore persistent-on-VM storage is useful working storage, not an independent backup. [GPU instance lifecycle and storage](https://docs.nvidia.com/brev/concepts/gpu-instances).

Credits belong to the organization. NVIDIA's console guide says zero credits can pause stoppable instances, delete non-stoppable instances, and eventually delete remaining resources if funding is not restored. Do not make credit exhaustion the shutdown plan or rely on a grace period as a data guarantee. Auto recharge is a separate billing choice and must not be enabled merely to implement this protection plan. [Billing and credit exhaustion](https://docs.nvidia.com/brev/guides/console-reference#billing--usage).

## Storage and artifact layout

Use a dedicated working directory under the actual persistent workspace, for example:

```text
/home/ubuntu/workspace/chameleon-experiments/
  source/                 reviewed source checkout / working-tree provenance
  inputs/                 corpus and frozen split/label manifests
  models/                 reproducible pinned base-model cache
  runs/<run-id>/
    checkpoints/          immutable completed checkpoints
    staging/              unfinished writes; never considered recoverable
    logs/                 run logs and metrics
    manifests/            config, environment, code/data/model hashes
  exports/<run-id>/        immutable completed off-VM export bundles
```

This is a proposed layout, not an assertion about the running VM. Verify the workspace mount and any container/host mapping before choosing the output path. Brev's default container and custom-image paths can differ. [Brev file-transfer and path guidance](https://docs.nvidia.com/brev/latest/guides/development-tools/file-transfer-scp).

On your workstation, use a project-owned directory such as `output/brev-backups` relative to the local repository, or a dedicated directory under your home folder. Keep incoming transfers distinct from verified snapshots. The authoritative local copy must be on storage outside the Brev instance; a second folder on the same VM does not supply that protection.

Keep original code, labels, split definitions and source assets locally before beginning. During a run, back up all irreplaceable outputs: changed adapter/head weights, metrics, run/config manifests, human labels produced remotely, and enough state for the claimed recovery mode. Pinned public base weights can normally be fetched again; record exact revisions and hashes instead of repeatedly copying unchanged multi-GB caches. A private or otherwise irreplaceable base checkpoint must itself have an independent copy.

Use an explicit artifact allowlist. Do not copy SSH keys, Brev credentials, HF tokens, `.env` files, shell history or the entire home directory. A Git push preserves tracked source history; it does not preserve ignored adapters, data, optimizer state or untracked working-tree files. Record the code SHA plus the relevant local patch/untracked-source manifest when the run did not use a clean commit.

## Atomic checkpoint publication

For a future trainer, save only at a coherent optimizer-step boundary through one writer:

1. Create a new uniquely named staging directory on the **same filesystem** as the final checkpoint directory.
2. Write the adapter/head, configuration and trainer state. For exact training continuation also save optimizer, scheduler, RNG, global step and data-order/sampler state. An inference-only adapter is a different recovery artifact.
3. Close and flush files. Record expected filenames, sizes and SHA-256 hashes in a manifest. Validate that every required file is present and parseable.
4. Where supported, sync file contents and parent-directory metadata before acknowledging durability. Atomically rename the staging directory to a **new, previously unused** final checkpoint name. Do not try replacing an existing nonempty checkpoint directory in place.
5. Publish/update a small latest-completed pointer atomically only after the completed directory exists. Readers accept the completed manifest, not merely a directory name or an unverified `latest` pointer.
6. Keep the previous valid checkpoint if saving, hashing, space allocation or publication fails. Never label a partial save completed.

Python provides `os.fsync` and same-filesystem rename/replacement primitives; the complete multi-file transaction and validation remain our responsibility. Cross-filesystem moves do not provide this publication guarantee. [Python filesystem operations](https://docs.python.org/3/library/os.html).

PEFT's adapter weights alone do not contain the original base model; loading needs the matching base plus adapter configuration. Preserve those identities and processor settings. The optimizer/RNG state needed to resume training is additional to an adapter export. [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).

The first trainer acceptance test should save, reload, and continue a short real-model run. No generic wrapper or signal can guarantee a final checkpoint from a trainer that has not implemented a safe save callback. Abrupt provider deletion may allow no final callback at all.

## Off-VM backup protocol

Publish immutable export bundles from completed snapshots, not from a checkpoint directory being rewritten. A bundle should contain a manifest tying its contents to `run_id`, checkpoint step, code/model/data revisions and a SHA-256 digest of the archive. Logs can be exported as a bounded snapshot alongside it.

For each pull:

1. Read the completed export's identity and manifest.
2. Transfer into a new local `incoming` name; never overwrite the last verified copy.
3. Check transfer exit status, expected files, lengths and SHA-256 values. Reject truncated files, mismatched identities, unsafe archive paths and symlink escapes.
4. Validate the export's internal manifest after safe extraction, or verify its listed files with an equivalent archive reader. An outer archive hash establishes transfer integrity, not sufficient resume contents.
5. Promote the local snapshot into `verified` and atomically record a local receipt containing the verified hash, step and UTC time.
6. Only then consider pruning redundant old remote snapshots. Retain at least two verified generations; do not automatically delete the only prior recovery point.

For the current GPU-probe-only stage, this can be much simpler: copy the probe JSON plus environment/run manifest, parse the JSON locally and compare its digest. Do not add invented training metadata or claim that an adapter exists.

Brev documents `brev copy` and managed SCP aliases. For example, from the **local WSL terminal**, after the actual instance name and export path have been verified:

```bash
CHAMELEON_INSTANCE='chameleon-l40s'
CHAMELEON_LOCAL_BACKUP="$HOME/chameleon-backups"
mkdir -p "$CHAMELEON_LOCAL_BACKUP/incoming" "$CHAMELEON_LOCAL_BACKUP/verified"

# Example protocol names: publish these remotely before running this command.
brev copy "$CHAMELEON_INSTANCE:/home/ubuntu/workspace/chameleon-experiments/exports/RUN_ID/SNAPSHOT.tar.gz" \
  "$CHAMELEON_LOCAL_BACKUP/incoming/SNAPSHOT.tar.gz.part"
```

`RUN_ID`/`SNAPSHOT` above identify an export produced by the future publisher; they are not existing artifacts. The backup implementation should pass validated paths as structured subprocess arguments and write a receipt only after verification. No key or SSH configuration change is proposed by this example. [Official transfer commands](https://docs.nvidia.com/brev/latest/guides/development-tools/file-transfer-scp).

For larger immutable trees, `rsync` can reduce repeated transfer work. Keep its partial/staging area outside verified snapshots; do not use `--inplace`, `--delete` or `--remove-source-files` for this recovery copy. A successful incremental copy still needs manifest/hash verification. Its normal size/time shortcut is not itself a content checksum. [Rsync manual](https://download.samba.org/pub/rsync/rsync.1).

Begin with an immediate verified off-VM copy of the run's input/provenance and probe output. For later training, proposed starting intervals are a checkpoint every five minutes and a local pull every five minutes or on new completed checkpoints, whichever the runner supports. The healthy-path recovery window includes checkpoint interval, polling delay and transfer time. Network outages can extend it; a nominal interval is not a promise of zero loss.

Windows and WSL have separate process/SSH environments. A Windows controller should explicitly call the intended WSL distribution and CLI, or use a separately verified native connection. Do not assume a WSL `brev` login creates a native Windows SSH alias. Do not change machine-level keys or settings just to hide that distinction.

## Run cap and stop verification

Use a local controller outside the VM as the primary stop owner. Its durable ledger should contain:

- Exact organization and immutable instance identity, alongside the readable name.
- Observed provider/type, running and stopped-storage rates, initial balance observation and balance timestamp.
- Absolute UTC stop deadline, conservative elapsed-time/cost estimate and explicitly selected run allowance.
- Last verified off-VM backup receipt, last backup error, stop requests and authoritative final state.

The runtime allowance must include setup and idle time after the VM starts, not only the trainer's subprocess duration. The organization's remaining credits can change because of other instances or users. A local rate-times-runtime calculation is an estimate, not an account-wide hard financial cap.

Within one controller process use a monotonic clock for elapsed intervals. Preserve the absolute deadline and previous spend in the local ledger so restarting the controller cannot reset the allowance. Handle clock rollback conservatively. Reject missing, nonfinite, negative or ambiguous rate/budget values instead of silently disabling the guard. Scope every action to the original identity; never stop a similarly named replacement instance or every instance in the organization.

Suggested stop sequence:

1. Arm and verify the local controller before running GPU work; persist its deadline before launch.
2. Independently bound the owned workload on the VM. A process timeout limits a runaway command but does **not** establish that Brev compute billing stops.
3. Before the deadline, request a cooperative save if the trainer supports it, then perform a final bounded pull and verification.
4. At the deadline, stop the exact Brev instance even if backup failed. Record that failure and preserve the last verified copy. Do not extend paid compute indefinitely while waiting for an unreachable backup destination.
5. Use the provider's stop operation and retry transient failures within the guard's recovery procedure. Treat a successful CLI response as a request acknowledgment, then inspect the authoritative lifecycle state until stopped or another terminal state is confirmed.
6. Persist stop confirmation and last backup status locally. Report unknown/pending stop state explicitly; do not mark the session safely stopped merely because SSH disconnected.

The documented command for one instance is `brev stop <instance-name>` followed by lifecycle inspection. Do not use `brev stop --all`, `brev delete`, guest `shutdown`, closing SSH, or exiting a trainer as substitutes for exact-instance provider stop verification. A deleted state observed externally is terminal but is not an authorized deletion action by the guard. [Brev lifecycle commands](https://docs.nvidia.com/brev/cli/instance-management).

Stopping avoids compute charges but can leave storage charges; it does not equal zero future charges or an indefinitely retained backup. Review retention separately after verifying an independent copy. This plan does not alter auto recharge or delete stored data.

The local controller depends on the workstation staying awake and connected. A VM-side workload timeout remains useful when it is not, but neither layer can ensure a provider stop if the control plane or connection is unavailable. State this operational limit, keep a conservative reserve and start with a short probe session. An independent provider-side maximum-runtime feature would be preferable if available and explicitly verified; none is asserted here.

## Meaningful guard and backup tests

Run these with fake provider responses, temporary directories and injected clocks first; no paid cloud resources are needed for the failure-path tests.

| Test | Required result |
| --- | --- |
| Kill a writer halfway through a checkpoint | Previous completed checkpoint remains valid; partial generation is never published |
| Disk fills or hashing fails during save | New checkpoint is rejected; previous recovery point is retained |
| Copy truncates or transport drops | No verified receipt; retry targets incoming storage; old verified generation is untouched |
| Hash differs, required optimizer file is absent, or archive has `../`/absolute/symlink paths | Verification fails; no promotion or unsafe extraction |
| Two checkpoint publishers or backup pulls race | Unique IDs/lock prevent destructive overwrite; receipt identifies one complete consistent generation |
| Run cap expires while the command is still busy | Stop is requested for the exact instance within configured tolerance |
| Workload exits early but instance remains running | Controller still performs backup/stop lifecycle; process exit is not success |
| Stop returns success while status remains running/stopping | Guard keeps checking/retrying as appropriate; does not report stopped |
| Stop/status API times out or returns malformed state | Local error/deadline persists; no false stop confirmation or delete fallback |
| Instance name is reused with a different ID, or active organization changes | No action on replacement/unrelated resources; identity mismatch is explicit |
| Controller restarts or wall clock moves backward | Original allowance/deadline is not reset or extended |
| Backup never succeeds near the deadline | Stop is not delayed indefinitely; final report names the newest actually verified snapshot |
| Invalid budget/rate/interval, insufficient local disk, or unwritable ledger | Guard refuses an unprotected launch rather than running silently |
| Restore copied probe bundle into a fresh directory | JSON parses, hashes match, run identity matches; no dependency on the original VM |
| Later restore actual trainer checkpoint on a fresh worker | Matching base/processor loads; adapter is usable; claimed optimizer continuation resumes at the recorded step |

For the live acceptance check, use the existing small GPU probe: observe exact instance identity/rate, create a report, pull and verify it, stop through Brev, confirm stopped in the control plane, and reopen the local report after disconnecting from the VM. Only then describe the guard and backup path as deployed and verified. Model-save/resume acceptance remains a separate future test.
