# ============================================================================
# Traffic Sign Detection -- reproducible runtime
#
# Headless by design: training, evaluation, batch prediction and the Jupyter
# notebook all run inside the container. The Tkinter desktop app (gui.py) is
# NOT run here -- a GUI toolkit in a container needs X11 forwarding and is more
# trouble than it is worth on macOS. Run gui.py natively instead; it needs
# nothing the container has.
#
# Build:  docker build -t traffic-sign-detection .
# Run:    docker compose up          -> Jupyter at http://localhost:8888
#         docker compose run --rm tsd python train.py --epochs 40
# ============================================================================
FROM python:3.11-slim

# Pinned so a rebuild six months from now produces the same environment.
# numpy, scipy and Pillow all ship manylinux and aarch64 wheels, so this
# installs from wheels on both Intel and Apple Silicon -- no compiler needed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, as their own layer: editing project code does not
# invalidate the (slow) pip install layer on rebuild.
COPY requirements.txt .
# notebook 7 is built on the JupyterLab components, so this one package gives
# both the classic and the lab interface -- no separate jupyterlab install.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "notebook>=7.0,<8"

# Project code. dataset/ and models/ are excluded by .dockerignore and are
# mounted as volumes at run time, so a rebuild never re-copies the dataset and
# a trained model survives image rebuilds.
COPY src/ ./src/
COPY train.py evaluate.py predict.py augment.py gui.py ./
COPY Project_Walkthrough.ipynb README.md SETUP.md REPORT.md ./

# Mount points for the two things that must outlive the container
RUN mkdir -p /app/dataset /app/models
VOLUME ["/app/dataset", "/app/models"]

EXPOSE 8888

# Default: Jupyter, reachable from the host browser.
# --ip 0.0.0.0 is required for the port publish to work.
# Token auth is disabled because the server is bound to a published port on a
# development machine only; do not expose this container to a network.
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/app"]
