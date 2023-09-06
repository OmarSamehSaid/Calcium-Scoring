<<<<<<< HEAD
# Use the official Python image as a builder stage
FROM python:3.9.13 AS builder
COPY . /app
# Set the working directory in the container
WORKDIR /app
# Install the Python dependencies
RUN pip install -r requirements.txt
# Expose the port for Flask (if needed)
EXPOSE 5000
# Specify the command to run when the container starts
CMD ["python","app.py"]
=======
# Use the official Python image as a builder stage
FROM python:3.9.13 AS builder
COPY . /app
# Set the working directory in the container
WORKDIR /app
# Install the Python dependencies
RUN pip install -r requirements.txt
# Expose the port for Flask (if needed)
EXPOSE 5000
# Specify the command to run when the container starts
CMD ["python","app.py"]
>>>>>>> e236d54e5f7c72f71815e431cb358b72e5265829
