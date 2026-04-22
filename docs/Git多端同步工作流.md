# Git 多端同步工作流

> 解决 Obsidian 多设备同步时的分支合并问题

## 问题分析

```
PC1 (Windows) ──┐
                ├──→ GitHub ──→ PC2 (Mac)
PC3 (WSL) ──────┘         └──→ 冲突/合并噩梦
```

**常见场景**:
1. PC1 修改了笔记A，PC2 修改了笔记B → 自动合并 ✅
2. PC1 和 PC2 同时修改了同一笔记 → 冲突 ❌
3. 忘记 pull 就开始编辑 → 覆盖之前的内容 ❌

---

## 推荐工作流

### 核心原则

```
1. 每次开始编辑前 → Pull 确认最新
2. 编辑完成后 → 立即 Commit + Push
3. 换设备前 → 确认 Sync indicator 为绿色
4. 只用 main 分支 → 不创建其他分支
```

### 详细步骤

#### 场景1: 开始一天的工作

```bash
# 1. 打开 Obsidian
# 2. 等待 Obsidian Git 插件自动 Pull (或手动 Ctrl+P → "Obsidian Git: Pull")
# 3. 确认 sync indicator 为绿色
# 4. 开始阅读/编辑笔记
```

#### 场景2: 完成编辑后

```bash
# 1. 确认所有修改已保存
# 2. Obsidian Git 应该会自动 commit + push
# 3. 或手动: Ctrl+P → "Obsidian Git: Commit" → "Obsidian Git: Push"
# 4. 确认 sync indicator 为绿色
```

#### 场景3: 切换设备

```bash
# 在旧设备上:
# 1. 关闭所有正在编辑的笔记
# 2. 等待 sync indicator 变为绿色
# 3. 关闭 Obsidian

# 在新设备上:
# 1. 打开 Obsidian
# 2. 等待自动 Pull 完成
# 3. 开始工作
```

---

## Obsidian Git 插件配置

### 推荐设置

```json
{
  "autoSaveInterval": 5,        // 每5分钟自动保存
  "autoPull": true,              // 自动拉取
  "autoPush": true,              // 自动推送
  "commitMessage": "sync: {{date}} {{time}}",  // 提交信息
  "pullBeforePush": true,        // 推送前拉取
  "conflictResolveBehavior": "ask"  // 冲突时询问
}
```

### 冲突解决

当发生冲突时，Obsidian Git 会：

1. 弹出对话框询问: "Conflict detected in [filename.md]. Choose resolution strategy:"
2. 选项:
   - **Keep current**: 保留你的版本
   - **Accept remote**: 接受远程版本
   - **Keep both**: 保留两个版本（推荐）

**推荐**: 选择 "Keep both"，然后手动合并

```markdown
<!-- 合并后的文件可能变成 -->
# 笔记标题

<<<<<<< HEAD (Your version)
你的修改内容
=======
远程的修改内容
>>>>>>> abc123 (Remote version)

```

---

## Git 钩子自动化

创建 `hooks/post-commit` 自动推送：

```bash
#!/bin/bash
# .git/hooks/post-commit

# 自动推送到远程
git push origin $(git branch --show-current) 2>/dev/null || true
```

```bash
# 设置执行权限
chmod +x .git/hooks/post-commit
```

---

## 高级策略: 分支隔离（可选）

如果你经常遇到冲突，可以使用分支策略：

```bash
# 按设备创建分支
git checkout -b windows
git checkout -b mac
git checkout -b wsl

# 工作流程
git add . && git commit -m "update" && git push origin windows
# 在其他设备上
git pull origin windows
```

**缺点**: 需要手动合并分支，增加复杂度

---

## 备份策略

### 定期备份到 ZIP

```bash
# 每周手动执行一次
cd /home/yanzm/sspo
zip -r backup_$(date +%Y%m%d).zip docs/ -x "*.git/*"
```

### GitHub Actions 自动备份

创建 `.github/workflows/backup.yml`:

```yaml
name: Backup Vault

on:
  schedule:
    - cron: '0 2 * * 0'  # 每周日凌晨2点
  workflow_dispatch:      # 也可手动触发

jobs:
  backup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          
      - name: Zip vault
        run: zip -r vault_backup.zip docs/
        
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: backup-${{ github.run_number }}
          assets: vault_backup.zip
```

---

## 最佳实践总结

| 场景 | 做法 |
|------|------|
| 日常使用 | 依赖 Obsidian Git 自动 sync |
| 长时间编辑 | 每30分钟检查一次 sync 状态 |
| 换设备 | 旧设备确保 sync 完成再离开 |
| 冲突发生 | 保留两份，手动合并 |
| 重要修改 | 手动 commit，不要只依赖 auto-save |

---

## 常见问题

### Q: sync indicator 是黄色/红色怎么办？
**A**: 
1. 点击 sync indicator 查看错误信息
2. 手动执行 Pull/Push
3. 检查网络连接

### Q: 冲突太多了怎么办？
**A**: 
1. 禁用 auto-push
2. 手动管理 commit
3. 必要时回退到某个稳定的 commit

### Q: 误删了笔记怎么恢复？
**A**: 
```bash
git log --all -- docs/deleted_note.md
git checkout <commit-hash> -- docs/deleted_note.md
```

### Q: 可以不用 Git 吗？
**A**: 可以用 Remotely Save + OneDrive/iCloud，但失去版本控制能力
