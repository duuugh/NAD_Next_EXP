# NAD_Next Web

基于当前 `NAD_Next/web` 现有 React + Vite 结构的研究成果展示站，不重建架构，直接消费 `web/public/data` 下的导出 JSON。

## 当前页面

- `/`：首页，总览 Early Stop / Best-of-N / Timeline
- `/early-stop`：Early Stop 主线、重点方案、router 结论
- `/best-of-n`：Best-of-N 主线、重点方案、方案切换对比
- `/timeline`：两条研究主线的时间线
- `/data`：关键 JSON / notes / report 下载页

## 数据来源

前端直接读取以下导出文件：

- `public/data/early_stop_cards.json`
- `public/data/best_of_n_cards.json`
- `public/data/research_timeline.json`
- `public/data/data_index.json`

这些文件由仓库根目录下的导出脚本生成：

- `../scripts/export_web_data.py`

脚本会：

- 从 `../result/` 读取现有 result / notes / report JSON
- 生成前端消费的 cards / timeline / data index
- 将关键原始产物复制到 `public/data/files/`，供 `/data` 页面下载

## 本地使用

在仓库根目录 `/home/jovyan/work/NAD_Next` 下已有结果文件时，推荐顺序：

```bash
cd /home/jovyan/work/NAD_Next
python scripts/export_web_data.py
cd web
npm install
npm run build
```

开发模式：

```bash
cd /home/jovyan/work/NAD_Next
python scripts/export_web_data.py
cd web
npm run dev
```

## 已验证

当前工作区已完成一次本地构建验证：

```bash
cd /home/jovyan/work/NAD_Next/web
npm run build
```

构建产物输出到：

- `dist/`

## 备注

- `public/data/` 属于前端静态资源，更新 `result/` 后需要重新运行 `python scripts/export_web_data.py`
- 当前 build 可通过，但 Vite 会提示主 bundle 超过 500 kB；这只是 chunk size warning，不影响本次构建成功
- 导出脚本是面向当前已有研究产物的薄层整理，不替换原始实验脚本与结果目录
