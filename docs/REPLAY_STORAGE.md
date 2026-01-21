# Replay Storage for Cloud Training

How to store and sync replay files for RunPod/cloud GPU training.

## Recommended: Backblaze B2

**Why B2:**
- Very cheap: $0.005/GB/month storage, $0.01/GB egress
- S3-compatible API (works with rclone, aws cli, etc.)
- 10GB free tier
- Simple setup, no complex IAM policies

**Cost example:** 10GB of replays = $0.05/month storage + ~$0.10/month if you pull daily

### Setup

1. **Create Backblaze account** at [backblaze.com](https://www.backblaze.com/b2/cloud-storage.html)

2. **Create a bucket:**
   - B2 Cloud Storage → Buckets → Create a Bucket
   - Name: `exphil-replays` (must be globally unique, add random suffix if needed)
   - Files in Bucket: **Private**

3. **Create application key:**
   - Account → App Keys → Add a New Application Key
   - Name: `exphil-rclone`
   - Allow access to: Select your bucket
   - Type of Access: Read and Write
   - **Save the keyID and applicationKey** (shown only once!)

4. **Configure rclone locally:**
   ```bash
   rclone config
   # n (new remote)
   # name: b2
   # Storage: b2 (Backblaze B2)
   # account: YOUR_KEY_ID
   # key: YOUR_APPLICATION_KEY
   # hard_delete: (leave blank)
   # Leave rest as defaults
   ```

5. **Upload your replays:**
   ```bash
   # Upload Mewtwo replays
   rclone sync ~/git/melee/slp/mewtwo b2:exphil-replays/mewtwo --progress

   # Upload all characters (when you have more)
   rclone sync ~/git/melee/slp b2:exphil-replays --progress
   ```

6. **Verify:**
   ```bash
   rclone ls b2:exphil-replays
   ```

### Environment Variables for Pod

Set these in RunPod pod config or export before running:

```bash
export B2_KEY_ID="your_key_id"
export B2_APP_KEY="your_application_key"
export B2_BUCKET="exphil-replays"
```

## Alternative: Cloudflare R2

**Why R2:**
- **Free egress** (huge if pulling frequently)
- S3-compatible
- 10GB/month free storage
- Slightly more complex setup than B2

### Setup

1. Create Cloudflare account, enable R2
2. Create bucket: `exphil-replays`
3. Create R2 API token with read/write access
4. Configure rclone:
   ```bash
   rclone config
   # n → r2 → s3 → Cloudflare R2
   # access_key_id: YOUR_ACCESS_KEY
   # secret_access_key: YOUR_SECRET_KEY
   # endpoint: https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com
   ```

### Environment Variables

```bash
export R2_ACCESS_KEY="your_access_key"
export R2_SECRET_KEY="your_secret_key"
export R2_ENDPOINT="https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com"
export R2_BUCKET="exphil-replays"
```

## Quick Setup on New Pod

If rclone isn't configured, create config and sync in one command:

```bash
rclone config create b2 b2 account="$B2_KEY_ID" key="$B2_APP_KEY" && rclone sync b2:exphil-replays /workspace/replays --progress
```

Or with credentials inline (not recommended for shared environments):

```bash
rclone config create b2 b2 account="YOUR_KEY_ID" key="YOUR_APP_KEY" && rclone sync b2:YOUR_BUCKET /workspace/replays --progress
```

## Secrets Management

**Option 1: RunPod Environment Variables (Recommended)**

Set secrets in RunPod pod template under "Environment Variables":
- `B2_KEY_ID` = your key ID
- `B2_APP_KEY` = your application key
- `B2_BUCKET` = your bucket name

These are encrypted at rest and injected at runtime. Then your fetch script just uses `$B2_KEY_ID` etc.

**Option 2: RunPod Secrets**

RunPod has a Secrets feature (Account → Secrets) that's more secure than env vars:
1. Create secrets in RunPod dashboard
2. Reference them in pod template as `{{RUNPOD_SECRET_B2_KEY_ID}}`

**Option 3: One-time interactive setup**

SSH in, run `rclone config` interactively, credentials stay in `/root/.config/rclone/rclone.conf` on the pod. Lost when pod terminates unless you use persistent storage.

**Never commit credentials to git or Docker images.**

## Pod Startup Workflow

### Option A: Manual (run after SSH)

```bash
# On pod, after SSH in:
/app/scripts/fetch_replays.sh
```

### Option B: RunPod Start Command

In RunPod pod config, set **Start Command** to:

```bash
/app/scripts/fetch_replays.sh && sleep infinity
```

This pulls replays automatically when pod starts.

### Option C: Docker Compose Override

```yaml
services:
  train:
    environment:
      - B2_KEY_ID=${B2_KEY_ID}
      - B2_APP_KEY=${B2_APP_KEY}
      - B2_BUCKET=exphil-replays
    command: >
      bash -c "/app/scripts/fetch_replays.sh &&
               mix run scripts/train_from_replays.exs --preset production --replays /workspace/replays"
```

## Directory Structure

```
/workspace/
├── replays/           # Synced from cloud storage
│   ├── mewtwo/
│   ├── ganondorf/
│   ├── link/
│   └── ...
└── checkpoints/       # Training outputs (keep on pod or sync back)
```

## Syncing Checkpoints Back

After training, push checkpoints to storage:

```bash
# Push to B2
rclone sync /workspace/checkpoints b2:exphil-replays/checkpoints --progress

# Or download locally
rclone sync b2:exphil-replays/checkpoints ~/git/melee/exphil/checkpoints --progress
```

## Quick Reference

```bash
# === LOCAL MACHINE ===

# Upload replays to B2
rclone sync ~/git/melee/slp b2:exphil-replays --progress

# Download checkpoints from B2
rclone sync b2:exphil-replays/checkpoints ~/git/melee/exphil/checkpoints --progress

# === ON POD ===

# Fetch replays (uses env vars)
/app/scripts/fetch_replays.sh

# Or manual rclone
rclone sync b2:$B2_BUCKET /workspace/replays --progress
```

## Storage Comparison

| Provider | Storage | Egress | Free Tier | Best For |
|----------|---------|--------|-----------|----------|
| Backblaze B2 | $0.005/GB | $0.01/GB | 10GB | Simple setup, low cost |
| Cloudflare R2 | $0.015/GB | **Free** | 10GB | Frequent pulls |
| AWS S3 | $0.023/GB | $0.09/GB | None | If already using AWS |
| Google Drive | Free | Free | 15GB | Already have rclone setup |

**Recommendation:** Start with **Backblaze B2** for simplicity. Switch to R2 if egress costs become significant.
