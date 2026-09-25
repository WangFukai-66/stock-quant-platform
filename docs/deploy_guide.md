# 网站部署指引：从本地项目到公网可访问的网址

本项目已为免费云端部署做好适配。按以下步骤操作，即可得到一个**输入网址直接打开**的在线网站
（默认网址形如 `https://你的应用名.streamlit.app`，免费、无需服务器、无需备案）。

## 原理

- 代码托管在 **GitHub**（免费），网站运行在 **Streamlit Community Cloud**（免费）。
- `main` 分支 = 本地开发版（完整依赖，用于训练）。
- `cloud` 分支 = 云端部署版（精简依赖 + 训练好的预测文件），**云端从这里部署**。

## 一、准备工作（一次性，约 10 分钟）

1. 注册 GitHub 账号：https://github.com/signup （免费，邮箱即可）
2. 在 GitHub 上**新建一个仓库**（New repository）：
   - 名称建议：`stock-quant-platform`
   - 选 **Public**（公开，免费版要求）
   - **不要**勾选 “Add a README”（保持空仓库）
3. 注册 Streamlit Cloud 账号：https://share.streamlit.io （直接用 GitHub 账号登录即可）

## 二、把本地代码推送到 GitHub（一次性）

在终端（项目根目录 `/Users/kk/雏雁项目`）执行，把 `你的GitHub用户名` 换成你的实际用户名：

```bash
git remote add origin https://github.com/你的GitHub用户名/stock-quant-platform.git

# 推送 main 分支（本地开发版）
git push -u origin main

# 推送 cloud 分支（云端部署版，包含训练好的预测文件）
git checkout cloud
git push -u origin cloud

# 切回 main 分支
git checkout main
```

## 三、在 Streamlit Cloud 上部署（一次性，约 5 分钟）

1. 打开 https://share.streamlit.io ，点右上角 **Create app** → **Deploy a public app from GitHub**
2. 按提示授权 Streamlit 访问你的 GitHub 仓库
3. 填写部署信息：
   - **Repository**：`你的GitHub用户名/stock-quant-platform`
   - **Branch**：选择 **cloud**（重要！）
   - **Main file path**：`app/main.py`
   - App URL：自定义一个名字（比如 `stock-quant-platform`）
4. 点 **Deploy**，等待 3~5 分钟构建完成（首次会自动安装依赖）
5. 部署完成后，浏览器打开 `https://你的应用名.streamlit.app` 即可访问 ✅

## 四、日常使用与更新

- **换设备访问**：任何设备浏览器输入网址即可，无需安装任何环境。
- **网站休眠**：免费版长时间无人访问会自动休眠，下次打开稍等十几秒会自动唤醒，属正常现象。
- **更新代码后重新上线**：
  ```bash
  # 在本地更新代码后
  git add -A
  git commit -m "更新说明"
  git push origin main        # 推 main（开发版）
  git checkout cloud
  git merge main              # 同步到 cloud
  git push origin cloud       # 推 cloud，Streamlit Cloud 会自动重新构建
  git checkout main
  ```
  > 在 Streamlit Cloud 的应用页面点右上角菜单 **Reboot app** 也可手动触发重建。

## 五、常见问题

| 问题 | 原因与解决 |
| --- | --- |
| 打开后一直转圈/报错 | 免费版休眠中，等待约 20 秒自动唤醒后刷新 |
| 页面显示"未找到预测文件" | 预测文件在 cloud 分支的 results/ 中；确认部署分支是 cloud |
| 行情数据加载失败 | 已内置东财→腾讯自动降级；若全部失败，多为数据源临时波动，稍后重试 |
| 构建失败 | 到 Streamlit Cloud 的 App → Logs 查看报错，最常见是分支选错或入口文件填错 |

## 六、升级到自己的 .com 域名（可选，后续再做）

免费版稳定运行后，若想要真正的 `.com` 网址，需要：

1. 购买域名（约 60~100 元/年，阿里云/腾讯云均可）
2. 租一台轻量云服务器（学生机约 100~300 元/年）
3. 国内服务器需 ICP 备案（个人实名，约 1~2 周）；海外服务器免备案但访问国内股票数据源可能不稳定

届时在服务器上跑 `streamlit run app/main.py`，再把域名解析到服务器即可。本项目代码无需改动。
