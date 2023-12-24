#Use an official PySpark image as a base
FROM jupyter/pyspark-notebook:latest

# Set the working directory
WORKDIR /app

# Copy your PySpark model files  and other to the container
COPY model /app/model
COPY run.py /app


# Install Flask and any other necessary Python dependencies
RUN pip install pyspark numpy pandas matplotlib seaborn scipy flask

# Expose the Flask app port
EXPOSE 5000

# Start the Flask app
CMD ["python3", "run.py"]