# Docker挂载
按你当前项目配置，最省事就是用 Docker Compose，因为挂载已经在 docker-compose.yaml 里写好了。
1. 克隆项目  
在你希望存放代码的目录执行：
    ```bash
    git clone 你的仓库地址 transformer  
    cd transformer
    ```
2. 确认本地目录结构
确保至少有这些目录（没有就创建）：
    ```bash
    mkdir -p data/speech_enhancement/clean  
    mkdir -p data/speech_enhancement/noise  
    mkdir -p data/speech_enhancement/noisy  
    mkdir -p experiments  
    mkdir -p enhancement_report
    ```
3. 启动容器并自动挂载
直接执行：
    ```bash
    docker compose up --build
    ```
当前配置会自动把本机目录挂载到容器内：  
- ./data -> /app/data  
- ./experiments -> /app/experiments  
- ./enhancement_report -> /app/enhancement_report

对应配置可见 docker-compose.yaml。

4. 验证挂载是否生效  
另开终端执行：
    ```bash
    docker exec -it dual_transformer bash  
    ls /app/data  
    ls /app/experiments  
    ls /app/enhancement_report
    ```
如果能看到本机同名目录内容，说明挂载成功。

5. 停止容器  
在 compose 终端里按 Ctrl+C，或执行：
```bash
docker compose down
```
补充说明：  
当前容器默认执行命令来自 docker-compose.yaml，即运行训练脚本。基础镜像和依赖安装逻辑在 Dockerfile。