FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for Poetry and building packages
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Disable Poetry virtualenv creation (use system Python in Docker)
RUN poetry config virtualenvs.create false

# Copy only dependency files first
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the app
COPY . .

# (Optional) Install the package itself if needed
RUN poetry install

# Expose Streamlit port
EXPOSE 8503

# Run the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8503", "--server.address=0.0.0.0"]