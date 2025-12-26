# WSL2 Migration Guide: Moving from C: to E: Drive

## Overview
WSL2 (Windows Subsystem for Linux 2) can consume significant disk space on your C: drive. This guide provides comprehensive steps to relocate your WSL2 installation to your E: drive to free up space on C:.

---

## Space Analysis

**Typical WSL2 Space Usage:**
- Minimal installation: 1-5 GB
- With development tools: 10-30 GB
- With datasets/models: 50+ GB

Your situation: Limited C: drive space, 200 GB available on E: drive ✓

---

## Option 1: Export and Re-Import (Recommended - Safest)

### Step 1: Check Your Current WSL Distributions
```powershell
wsl --list -v
```
This shows all installed distributions and their versions.

### Step 2: Backup Your Distribution
Create a backup on E: drive:
```powershell
# Create backup directory
mkdir E:\wsl-backup

# Export your distribution (replace Ubuntu with your distro name if different)
wsl --export Ubuntu E:\wsl-backup\ubuntu-backup.tar
```
**Note:** This may take 5-10 minutes depending on your WSL2 size.

### Step 3: Unregister from C: Drive
```powershell
wsl --unregister Ubuntu
```
⚠️ **Warning:** This removes the WSL2 installation from C: drive. The backup protects your data.

### Step 4: Verify Uninstallation
```powershell
wsl --list -v
```
Ubuntu should no longer appear in the list.

### Step 5: Create Install Directory on E: Drive
```powershell
mkdir E:\wsl-installs\ubuntu
```

### Step 6: Import Distribution to E: Drive
```powershell
wsl --import Ubuntu E:\wsl-installs\ubuntu E:\wsl-backup\ubuntu-backup.tar
```
**Note:** This may take 5-10 minutes.

### Step 7: Verify Installation
```powershell
wsl --list -v
wsl --set-default Ubuntu
```
You should see Ubuntu listed with version 2 and location on E: drive.

### Step 8: Test WSL2
```powershell
wsl
```
You should enter your Linux environment. Type `exit` to return to PowerShell.

### Step 9: Clean Up (Optional)
Once verified, delete the backup:
```powershell
Remove-Item E:\wsl-backup\ubuntu-backup.tar
```

---

## Option 2: Move vhdx File (Advanced)

**Pros:** Faster than export/import, keeps settings  
**Cons:** More complex, requires registry modification

### Step 1: Stop WSL2
```powershell
wsl --shutdown
```

### Step 2: Locate vhdx File
The default location is:
```
%USERPROFILE%\AppData\Local\Packages\CanonicalGroupLtd.UbuntuonWindows_79rhkp1fndgsc\LocalState\ext4.vhdx
```

Navigate there to find your exact path (GUID may differ).

### Step 3: Create Directory on E: Drive
```powershell
mkdir E:\wsl-data
```

### Step 4: Copy vhdx File
```powershell
Copy-Item "$env:USERPROFILE\AppData\Local\Packages\CanonicalGroupLtd.UbuntuonWindows_*\LocalState\ext4.vhdx" -Destination "E:\wsl-data\ext4.vhdx"
```

### Step 5: Update WSL Configuration
Create or edit `%USERPROFILE%\.wslconfig`:
```ini
[interop]
enabled = true
appendWindowsPath = true

[wsl2]
kernel=C:\Windows\System32\lxss\tools\kernel
```

Add custom distribution configuration in:
```
%USERPROFILE%\AppData\Local\Packages\CanonicalGroupLtd.UbuntuonWindows_*\LocalState\wsl.conf
```

### Step 6: Restart WSL2
```powershell
wsl
```

### Step 7: Verify
Check disk usage and confirm it's working properly.

---

## Option 3: Fresh Install on E: Drive (Clean Start)

Use this if you want a fresh WSL2 installation without migration.

### Step 1: Uninstall Existing WSL2
```powershell
wsl --unregister Ubuntu
```

### Step 2: Install from Microsoft Store
Or use command line:
```powershell
wsl --install -d Ubuntu
```

### Step 3: Set Default Location
Before first run, configure `.wslconfig` to prefer E: drive for new distributions.

---

## Recommended Approach Summary

| Option | Time | Difficulty | Data Loss Risk | Recommendation |
|--------|------|-----------|-----------------|-----------------|
| Option 1 (Export/Import) | 15-20 min | Easy | None | ✅ **BEST** |
| Option 2 (vhdx Move) | 10-15 min | Hard | Low | Advanced users |
| Option 3 (Fresh Install) | 10 min | Very Easy | Total | If starting fresh |

---

## Post-Migration Checklist

- [ ] WSL2 boots normally: `wsl`
- [ ] File system intact: `ls -la` in WSL2
- [ ] Docker works (if installed): `docker --version`
- [ ] Git access works: `git --version`
- [ ] Development tools functional
- [ ] Old installation removed from C: drive
- [ ] Backup on E: drive kept for 1-2 weeks before deletion
- [ ] Disk space verified on both drives

---

## Troubleshooting

### WSL2 Won't Start After Migration
```powershell
# Restart WSL service
wsl --shutdown
# Wait 10 seconds
wsl
```

### High CPU Usage After Migration
```powershell
# Restart the entire WSL service
Restart-Service LxssManager
```

### Storage Still High on C: Drive
Check for:
```powershell
# Other packages in AppData
Get-ChildItem "$env:USERPROFILE\AppData\Local\Packages" -Filter "*Ubuntu*" -Directory
```

### Permission Denied Errors
Run PowerShell as Administrator before executing WSL commands.

---

## Space Recovery

**After successful migration, you can reclaim C: drive space:**

1. Delete old WSL2 package directory:
   ```powershell
   # Run as Administrator
   Remove-Item "$env:USERPROFILE\AppData\Local\Packages\CanonicalGroupLtd.Ubuntu*" -Recurse
   ```

2. Run Disk Cleanup:
   - Press `Win + R`
   - Type `cleanmgr`
   - Select C: drive
   - Check "Temporary files" and other safe options

3. Expected space recovery: **5-50 GB** depending on your setup

---

## Important Notes

⚠️ **Backup First:** Always backup important files before migration  
⚠️ **Administrator Rights:** You need admin privileges for WSL commands  
⚠️ **Docker Desktop:** If using Docker Desktop with WSL2 backend, it may need reconfiguration  
⚠️ **WSL2 Version:** Ensure you're using WSL2, not WSL1: `wsl --list -v`

---

## For Your Project Setup

Since you're working on the Urban Climate Modeling project:

1. After migration, reinstall dependencies:
   ```bash
   cd /path/to/project
   pip install -r requirements.txt
   ```

2. Rebuild Docker images if needed:
   ```bash
   docker-compose up --build
   ```

3. Verify GPU setup (if applicable):
   ```bash
   python test_gpu.py
   ```

---

## Quick Reference Commands

```powershell
# Check distributions
wsl --list -v

# Stop WSL
wsl --shutdown

# Start specific distro
wsl -d Ubuntu

# Set default distro
wsl --set-default Ubuntu

# Check version
wsl --version
```

---

**Last Updated:** December 2025  
**Tested On:** Windows 10/11 with WSL2
