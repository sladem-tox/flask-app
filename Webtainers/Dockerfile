# Use an official Miniconda image as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the local directory contents into the container at /app
COPY . /app

# Create a new Conda environment and install Python packages
RUN conda create -c conda-forge -n my-rdkit-env rdkit && \
    echo "conda activate my-rdkit-env" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && conda install -n my-rdkit-env pandas" && \
    /bin/bash -c "source ~/.bashrc && conda activate my-rdkit-env && pip install torch flask"  # Add Flask installation

# Activate the Conda environment and run a Python script
CMD ["/bin/bash", "-c", "source ~/.bashrc && conda activate my-rdkit-env  && python BabyModel.py"]
