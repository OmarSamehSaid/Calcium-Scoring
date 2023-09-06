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
