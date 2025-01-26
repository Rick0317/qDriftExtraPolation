FROM ubuntu

# Use root user
USER root

# Install dependencies in a single RUN command to reduce image layers
RUN apt-get update && apt-get install -y \
    bzip2 \
    cmake \
    git \
    wget \
    libblas-dev \
    liblapack-dev \
    python3 \
    python3-pip \
    python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Update PATH environment variable
ENV PATH="/opt/conda/bin:$PATH"

# Create a Conda environment and install Psi4
RUN conda create -y -n psi4env psi4 -c psi4 && \
    conda clean -a -y

# Activate the Conda environment
SHELL ["conda", "run", "-n", "psi4env", "/bin/bash", "-c"]

# Install PySCF
RUN git clone https://github.com/sunqm/pyscf /root/pyscf && \
    mkdir /root/pyscf/pyscf/lib/build && \
    cd /root/pyscf/pyscf/lib/build && \
    cmake .. && \
    make

# Set environment variables for PySCF
ENV PYTHONPATH="/root/pyscf:$PYTHONPATH"


# Install OpenFermion, Cirq, and plugins
RUN pip install --upgrade pip setuptools wheel

RUN pip install openfermion 
RUN pip install cirq 
RUN pip install openfermioncirq 
RUN pip install openfermionpsi4 
RUN pip install openfermionpyscf
RUN pip install qiskit
RUN pip install tqdm
RUN pip install seaborn

# Ensure python points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Jupyter within the Conda environment
RUN conda run -n psi4env pip install jupyter notebook && \
    conda clean -a -y

# Set entrypoint to bash
ENTRYPOINT ["/bin/bash"]
