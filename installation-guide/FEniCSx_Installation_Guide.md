# FEniCSx Installation Guide
**Using Docker on Windows 11 and Conda on macOS**  
**Author:** Farhad Kamarei

This guide provides practical setup instructions for:

- **Windows 11:** FEniCSx in Docker
- **macOS:** FEniCSx in Conda
- **Visualization:** ParaView installation and basic use

---

## Windows 11 Installation (Docker)

### Step 1 — Install Docker Desktop
1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Install and restart your computer if prompted.
3. Verify installation:
   ```powershell
   docker --version
   ```

### Step 2 — Install VS Code
1. Download VS Code: https://code.visualstudio.com/
2. Install extensions:
   - Docker (Microsoft)
   - Python (Microsoft)

Optional terminal install:
```powershell
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-python.python
```

### Step 3 — Prepare Project Folder
```powershell
cd "C:\Users\YourUsername\Documents\fenicsx_project"
code .
```

### Step 4 — Pull FEniCSx Image
```powershell
docker pull dolfinx/dolfinx:nightly
```

Optional stable tags: https://hub.docker.com/r/dolfinx/dolfinx/tags

### Step 5 — Run Container
```powershell
docker run -it --name dolfinximage -v ${PWD}:/home/shared dolfinx/dolfinx:nightly
```

- `${PWD}` maps your current folder
- Files appear in container at `/home/shared`

### Step 6 — Verify and Run
```bash
python3 -c "import dolfinx; print(f'DOLFINx version: {dolfinx.__version__}')"
cd /home/shared
python3 test.py
```

If needed:
```bash
/dolfinx-env/bin/python test.py
```

### Step 7 — Reuse Container
Exit:
```bash
exit
```

Restart later:
```powershell
docker start -ai dolfinximage
```

### Troubleshooting (Windows)
- Check Docker Desktop file-sharing permissions.
- Remove container:
  ```powershell
  docker rm dolfinximage
  ```
- Remove image:
  ```powershell
  docker rmi dolfinx/dolfinx:nightly
  ```

---

## macOS Installation (Conda)

### Step 1 — Install Anaconda or Miniconda
Download: https://www.anaconda.com/products/distribution

Verify:
```bash
conda --version
```

### Step 2 — Install VS Code + Python Extension
```bash
code --install-extension ms-python.python
```

### Step 3 — Create FEniCSx Environment
```bash
conda create -n fenicsx-env python=3.11
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
python3 -c "import dolfinx; print(f'DOLFINx version: {dolfinx.__version__}')"
```

### Step 4 — Working Directory
```bash
mkdir -p ~/Documents/fenicsx_project
cd ~/Documents/fenicsx_project
code .
```

In VS Code: select Python interpreter from `fenicsx-env`.

### Step 5 — Test
```bash
conda activate fenicsx-env
python3 test.py
```

### Troubleshooting (macOS)
- List environments:
  ```bash
  conda env list
  ```
- Remove environment:
  ```bash
  conda env remove -n fenicsx-env
  ```

### Alternative: Docker on macOS
```bash
docker run -it --name dolfinximage -v $(pwd):/home/shared dolfinx/dolfinx:nightly
```

---

## ParaView Installation (Visualization)

### Why ParaView?
Use ParaView for post-processing FEniCSx outputs (`.xdmf`, `.vtu`, `.pvd`), contour plots, crack fields, and animations.

### Windows
1. Download: https://www.paraview.org/download/
2. Install and open result files.

### macOS
1. Download: https://www.paraview.org/download/
2. Drag to Applications and launch.

### Optional (Conda on macOS)
```bash
conda activate fenicsx-env
conda install -c conda-forge paraview
```

### First Use
1. **File > Open**
2. Click **Apply**
3. Use **Color by** to choose variables
4. Use filters like **Slice**, **Clip**, and **Warp By Vector**

---
