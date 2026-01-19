# Downloading from Google Drive with rclone

Guide for downloading large files from Google Drive on RunPod or other cloud instances, bypassing rate limits.

## Why rclone?

- **gdown** gets rate-limited on popular files ("Too many users have viewed or downloaded this file")
- **rclone** with your own Google API credentials bypasses shared quota limits
- Works reliably for 50GB+ files

## Setup

### 1. Create Google Cloud Project & Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable Google Drive API:
   - APIs & Services → Library
   - Search "Google Drive API" → Enable
4. Create OAuth credentials:
   - APIs & Services → Credentials
   - Create Credentials → OAuth client ID
   - Application type: **Desktop app**
   - Copy the **Client ID** and **Client Secret**

### Finding Your Credentials Later

If you already created credentials and need to find them:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. **APIs & Services** → **Credentials**
3. Under "OAuth 2.0 Client IDs", click on your client name
4. Both **Client ID** and **Client Secret** are shown on that page
5. Or click the download icon (⬇) to get a JSON file with both values

### 2. Install rclone on Cloud Instance

```bash
curl https://rclone.org/install.sh | bash
```

### 3. Configure rclone

```bash
rclone config
```

Follow prompts:
1. `n` - new remote
2. Name: `gdrive`
3. Storage: `drive` (Google Drive)
4. **client_id**: Paste your Client ID
5. **client_secret**: Paste your Client Secret
6. Scope: `1` (full access)
7. Root folder ID: (leave blank)
8. Service account: (leave blank)
9. Advanced config: `n`
10. Auto config: `n` (remote machine)

You'll get a command like:
```
rclone authorize "drive" "eyJjbGllbnRf..."
```

### 4. Authorize on Local Machine

Run the authorize command on your **local machine** (with browser):

```bash
# Install rclone locally if needed
# Manjaro: sudo pacman -S rclone
# Ubuntu: sudo apt install rclone
# Mac: brew install rclone

rclone authorize "drive" "eyJjbGllbnRf..."
```

Browser opens → authorize with Google → copy the token back to cloud instance.

### 5. Finish Config

- Shared Drive: `n`
- Keep remote: `y`
- Quit: `q`

## Downloading Files

### Step 1: Add Files to "Shared with me"

For public Google Drive links, **open each link in your browser first**. This adds them to your "Shared with me" which rclone can access.

**Slippi Ranked Replay Archives:**
1. https://drive.google.com/file/d/1pFjgh1dapX34s0T-Q1TC7JUO-qjYbQZf/view (ranked-anonymized-1)
2. https://drive.google.com/file/d/1jEIzvhpV3778J2s2-Np9vCVqSLf9lZnk/view (ranked-anonymized-2)
3. https://drive.google.com/file/d/1glzlkAPxHC58oXZljJXQV8dsTBKmlhkE/view (ranked-anonymized-3)
4. https://drive.google.com/file/d/1qdIZUW4Er_Vu6rD3-VUvyak3lKa1KxVk/view (ranked-anonymized-4)
5. https://drive.google.com/file/d/1Hqmj6C8g1BzuRAIqOrQcMDL0MX4GtffE/view (ranked-anonymized-5)
6. https://drive.google.com/file/d/1g8yZ-Q4ldyhDEmXLSPBoWxywJRMRVGc3/view (ranked-anonymized-6)

Open each link in your browser, then proceed to download.

### Step 2: List Available Files

```bash
rclone lsf --drive-shared-with-me gdrive:
```

This shows all files now in your "Shared with me".

### Step 3: Download All 7z Files

```bash
rclone copy --progress --drive-shared-with-me "gdrive:" /workspace/downloads/ --include "*.7z"
```

### Alternative: Download Single File

```bash
rclone copy --progress --drive-shared-with-me "gdrive:ranked-anonymized-1-116248.7z" /workspace/downloads/
```

### Alternative: Download by Pattern

```bash
rclone copy --progress --drive-shared-with-me "gdrive:" /workspace/downloads/ --include "ranked-anonymized-*"
```

## Processing After Download

Once files are downloaded, run the filter script:

```bash
cd /workspace

# Extract and filter for low-tier characters
python3 cloud_filter_replays.py \
  --urls-file links.txt \
  --output /workspace/lowtier \
  --download-dir /workspace/downloads \
  --cleanup
```

Or manually extract and filter:

```bash
# Extract
7z x /workspace/downloads/ranked-1.7z -o/workspace/extracted

# Filter (using the script's filter function)
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/workspace')
from cloud_filter_replays import filter_replays
filter_replays('/workspace/extracted', '/workspace/lowtier', num_workers=8)
PYEOF

# Cleanup
rm -rf /workspace/extracted
```

## Troubleshooting

### "Too many users have viewed or downloaded this file"
- Use your own Google API Client ID (see setup above)
- Or wait 24 hours for rate limit reset

### "directory not found" errors
- Don't use `--drive-root-folder-id` with file IDs
- Open the link in browser first to add to "Shared with me"
- Use `--drive-shared-with-me` flag

### Slow downloads
- Google Drive can throttle; speeds vary
- Large files (50GB+) may take 30-60 min

### Auth token expired
```bash
rclone config reconnect gdrive:
```

## RunPod-Specific Notes

### After Pod Restart

RunPod container filesystems are **ephemeral** - rclone and configs are lost on restart. Your `/workspace` volume persists. When resuming:

```bash
# 1. Reinstall rclone
curl https://rclone.org/install.sh | bash

# 2. Create directories on persistent storage
mkdir -p /workspace/tmp /workspace/downloads

# 3. Reconfigure rclone
rclone config
# Follow prompts: n → gdrive → drive → paste client_id → paste client_secret → 1 → blank → blank → n → n
# Copy the authorize command, run on local machine, paste token back
# Then: n → y → q
```

### Avoiding "No Space Left" Errors

RunPod containers have small root disks (5GB). Rclone uses temp files that can fill this up.

**Solution 1: Redirect temp files to volume**
```bash
TMPDIR=/workspace/tmp rclone copy --progress --drive-shared-with-me "gdrive:" /workspace/downloads/ --include "*.7z"
```

**Solution 2: Disable multi-threading (no temp files)**
```bash
rclone copy --progress --multi-thread-streams 0 --drive-shared-with-me "gdrive:" /workspace/downloads/ --include "*.7z"
```

### Resuming Interrupted Downloads

Rclone automatically resumes partial downloads. Just run the same command again - it will skip completed files and resume partials.

## Quick Reference

```bash
# List shared files
rclone lsf --drive-shared-with-me gdrive:

# Download all 7z files (with temp fix for RunPod)
TMPDIR=/workspace/tmp rclone copy --progress --drive-shared-with-me "gdrive:" /workspace/downloads/ --include "*.7z"

# Check download progress in another terminal
watch -n 5 'ls -lh /workspace/downloads/'

# Check disk space
df -h /workspace
```
