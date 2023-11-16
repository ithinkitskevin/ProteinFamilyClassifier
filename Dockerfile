# Start from a PyTorch base image with CUDA 12.3
FROM pytorchlightning/pytorch_lightning:latest-py3.7-torch1.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define entrypoint
ENTRYPOINT ["python", "./src/main.py"]