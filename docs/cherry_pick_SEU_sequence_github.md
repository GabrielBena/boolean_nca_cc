# Cherry-Pick SEU Sequence Strategy

## Branch Structure Overview

### Current State (Post-Rollback)
- **`rebuild`** (main working branch): Clean state at commit `91d9377` + new development
- **`backup-before-rollback`** (archive): Contains 6 SEU-related commits that were rolled back
- **`new-feature-branch`** (development): Clean state at `91d9377` for new work

### Commit History in `backup-before-rollback`
```
2a42856 - KO attempt 1 rollback to perfect accuracy
d4eab93 - reversible KO attempt 1: full trajectories (post-damage pre recovery logged)
9611925 - stash bit-level granularity SEU
fc0380f - Implemented sequential bit flip injection during inner loop eval trajectory
29fab38 - SEU eval complete
8c233f6 - WIP: finish SEU train loop integration, eval and overall plumbing
```

## Workflow Strategy

### 1. Development Phase
```bash
# Work on new features from clean state
git checkout new-feature-branch
# ... make changes ...
git add .
git commit -m "New feature description"
```

### 2. Integration Phase
When ready to integrate new work into main branch:
```bash
# Update rebuild branch with new work
git checkout rebuild
git reset --hard new-feature-branch
git push --force-with-lease origin rebuild
```

### 3. Selective Cherry-Picking
To selectively add back specific SEU features from backup:
```bash
# Cherry-pick specific commits from backup
git checkout rebuild
git cherry-pick <commit-hash>

# Example: Add only SEU eval functionality
git cherry-pick 29fab38

# Example: Add bit flip injection
git cherry-pick fc0380f
```

## Branch Purposes

### `rebuild` Branch
- **Purpose**: Main working branch, definitive version
- **State**: Clean `91d9377` + integrated new work + selectively cherry-picked features
- **Remote**: Synced with `origin/rebuild`

### `backup-before-rollback` Branch
- **Purpose**: Archive of rolled-back commits
- **State**: Contains all 6 SEU-related commits
- **Remote**: Synced with `origin/backup-before-rollback`
- **Usage**: Source for cherry-picking specific features

### `new-feature-branch` Branch
- **Purpose**: Development sandbox
- **State**: Clean `91d9377` + new development work
- **Usage**: Safe space for experimentation before integration

## Cherry-Pick Commands Reference

### Available Commits for Cherry-Picking
```bash
# SEU evaluation functionality
git cherry-pick 29fab38

# Bit flip injection during evaluation
git cherry-pick fc0380f

# Bit-level granularity SEU
git cherry-pick 9611925

# Reversible KO with full trajectories
git cherry-pick d4eab93

# KO rollback to perfect accuracy
git cherry-pick 2a42856

# SEU train loop integration (WIP)
git cherry-pick 8c233f6
```

### Cherry-Pick Best Practices
1. **Test individually**: Cherry-pick one commit at a time and test
2. **Resolve conflicts**: Handle merge conflicts as they arise
3. **Document decisions**: Note which features were selected and why
4. **Preserve backup**: Never delete `backup-before-rollback` branch

## Emergency Recovery

### If you need to restore all SEU commits:
```bash
git checkout rebuild
git reset --hard backup-before-rollback
git push --force-with-lease origin rebuild
```

### If you need to start over from clean state:
```bash
git checkout rebuild
git reset --hard 91d9377ce96301bfbdbbbb19c72d90b6abade2d7
git push --force-with-lease origin rebuild
```

## Notes
- The `backup-before-rollback` branch is permanently preserved on remote
- All SEU-related commits remain available for selective integration
- The rollback was performed on 2025-01-17 to clean state `91d9377`
- This strategy allows maximum flexibility in feature selection while maintaining a clean main branch
