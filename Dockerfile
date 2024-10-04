# Base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

RUN apt-get update \
  && apt-get -y install tesseract-ocr \ 
  && apt-get -y install ffmpeg libsm6 libxext6 \
  && apt-get -y install poppler-utils

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Download necessary nltk data
COPY nltk_downloads.py .
RUN python nltk_downloads.py

# Expose the port that Streamlit uses
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.port=8501"]