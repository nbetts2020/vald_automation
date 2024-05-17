# Python runtime as a parent image
FROM python:3.9-slim

# Container working directory
WORKDIR /usr/src/app

# Copying the requirements.txt file into the container at /usr/src/app
COPY requirements.txt ./

# Installing any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the working directory contents into the container at /usr/src/app
COPY . .

# Making port 80 available to the world outside this container
EXPOSE 80

# Defining environment variable
ENV NAME World

# Running main.py when the container launches
CMD ["python", "main.py"]
