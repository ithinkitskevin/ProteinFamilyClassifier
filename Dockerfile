# Use Python 3.7.4 image as the base image
FROM python:3.7.4

RUN apt-get update && apt-get install -y wget
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.19.0/protoc-3.19.0-linux-x86_64.zip
RUN unzip protoc-3.19.0-linux-x86_64.zip -d /usr/local bin/protoc
RUN unzip protoc-3.19.0-linux-x86_64.zip -d /usr/local 'include/*'
RUN rm protoc-3.19.0-linux-x86_64.zip

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements.txt file into the container at /usr/src/app
COPY requirements.txt ./
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN pip install jupyter

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
