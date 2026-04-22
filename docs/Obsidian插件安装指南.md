# Obsidian 插件安装指南

> 让 Obsidian 成为强大的论文阅读和知识管理工具

## 推荐的插件列表

### 核心插件（必装）

#### 1. Templater
- **作用**: 模板引擎，支持变量、脚本、动态内容
- **安装**: Community Plugins 搜索 "Templater"
- **配置**: 设置模板文件夹 `docs/templates/`

```markdown
<!-- 模板示例: 新论文笔记 -->
---
title: <% tp.file.title %>
paper: 
authors: 
date: <% tp.date.now() %>
tags: [paper, ]
status: to-read
---

# <% tp.file.title %>

## 论文信息
- **作者**: 
- **会议/期刊**: 
- **GitHub**: 

## 一句话总结

## 核心贡献

## 方法

## 实验

## 相关工作

## 待研究问题

```

#### 2. Dataview
- **作用**: 类SQL查询，从笔记中提取数据
- **安装**: Community Plugins 搜索 "Dataview"
- **用途**: 索引所有论文、管理阅读状态

```dataview
TABLE title, authors, date, status
FROM "docs"
WHERE type = "paper"
SORT date DESC
```

#### 3. Admonition
- **作用**: 美观的提示块（tip/info/warning/danger）
- **安装**: Community Plugins 搜索 "Admonition"
- **使用**:

```markdown
> [!tip] 提示
> 这是提示内容

> [!info] 信息
> 一般信息

> [!warning] 警告
> 注意事项

> [!danger] 危险
> 重要警告
```

#### 4. Style Settings
- **作用**: 主题自定义
- **安装**: Community Plugins 搜索 "Style Settings"
- **配置**: 微调字体、颜色、布局

#### 5. Quick Switcher++
- **作用**: `Ctrl+O` 快速跳转，支持正则
- **安装**: Community Plugins 搜索 "quick switcher++"

---

### 科研论文插件

#### 6. Citations
- **作用**: 读取bibtex，自动补全引用
- **安装**: Community Plugins 搜索 "Citations"
- **配置**: 设置bib文件路径

```bibtex
@article{lee2024sspo,
  title={Semi-Supervised Preference Optimization with Limited Feedback},
  author={Lee, Seonggyun and Lim, Sungjun and Park, Seojin and Cheon, Soeun and Song, Kyungwoo},
  year={2024},
  journal={arXiv preprint arXiv:2511.00040}
}
```

#### 7. Zotero Integration
- **作用**: 联动Zotero提取PDF注释
- **安装**: Community Plugins 搜索 "Zotero Integration"
- **用途**: 自动同步Zotero文献库

#### 8. Paper Import
- **作用**: arXiv ID直接下载+解析论文
- **安装**: Community Plugins 搜索 "paper-import"

#### 9. Markmind
- **作用**: PDF批注+思维导图
- **安装**: Community Plugins 搜索 "Markmind"

---

### 代码阅读插件

#### 10. Codeblock Shortcuts
- **作用**: 代码块增强
- **安装**: Community Plugins 搜索 "codeblock-shortcuts"

#### 11. Syntax Highlight
- **作用**: 200+语言高亮
- **安装**: Community Plugins 搜索 "syntax-highlighting"

#### 12. Advanced Tables
- **作用**: Markdown表格编辑器
- **安装**: Community Plugins 搜索 "advanced-tables"

---

### 效率提升插件

#### 13. Kanban
- **作用**: 看板视图
- **安装**: Community Plugins 搜索 "kanban"
- **用途**: 管理复现任务进度

```markdown
## SSPO 复现进度

- [ ] Phase 1: 环境搭建
- [ ] Phase 2: 数据准备
- [x] Phase 3: 算法实现
```

#### 14. Excalidraw
- **作用**: 手绘风格图表
- **安装**: Community Plugins 搜索 "excalidraw"
- **用途**: 画架构图、流程图

#### 15. Tracker
- **作用**: 数据可视化
- **安装**: Community Plugins 搜索 "tracker"
- **用途**: 追踪学习进度

#### 16. Iconize
- **作用**: 为笔记/文件夹添加emoji图标
- **安装**: Community Plugins 搜索 "iconize"

---

## 一键安装脚本

在 Obsidian 的 Community Plugins 中搜索以下插件名称安装：

```
核心插件（5个）:
1. Templater
2. Dataview
3. Admonition
4. Style Settings
5. Quick Switcher++

科研插件（4个）:
6. Citations
7. Zotero Integration (如果你用Zotero)
8. Paper Import
9. Markmind

代码插件（3个）:
10. Codeblock Shortcuts
11. Syntax Highlight
12. Advanced Tables

效率插件（4个）:
13. Kanban
14. Excalidraw
15. Tracker
16. Iconize
```

---

## 插件配置建议

### Templater 配置

1. 创建模板文件夹: `docs/templates/`
2. 设置模板触发命令: `Ctrl+Shift+E`
3. 启用自动触发

### Dataview 配置

1. 启用JavaScript查询
2. 设置默认视图为表格

### Admonition 配置

1. 自定义颜色主题
2. 添加自定义类型

---

## 常见问题

### Q: 插件冲突怎么办？
**A**: 禁用部分插件，逐个排查冲突源

### Q: 启动很慢怎么办？
**A**: 减少插件数量，禁用不必要的社区插件

### Q: 如何备份插件配置？
**A**: 使用 Obsidian Git 插件自动同步 `.obsidian/plugins/` 目录
