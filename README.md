# Facial Recognition API

## Overview

This repository is part of the TechCom project and was used for experimentation purposes. It is designed to function as a standalone microservice or be easily integrated into the broader TechCom backend. The API provides facial recognition capabilities, including onboarding new users and recognizing faces from images.

## Features

- **Onboarding Users**: Capture and store facial embeddings for new users.
- **Face Recognition**: Identify users based on their facial embeddings.
- **Web Interface**: A simple HTML interface for onboarding and recognition.
- **Database Integration**: Uses PostgreSQL with pgvector for storing and querying facial embeddings.

## Technologies Used

- **Backend Framework**: FastAPI
- **Database**: PostgreSQL with pgvector extension
- **Image Processing**: OpenCV and face_recognition
- **Frontend**: HTML, CSS, and JavaScript
- **ORM**: SQLModel

## Project Structure

```
├── database.py          # Database connection and session management
├── main.py              # FastAPI application with endpoints for onboarding and recognition
├── models.py            # SQLModel definitions for Users and FaceEncoding
├── templates/
│   └── index.html       # Web interface for onboarding and recognition
├── myvenv/              # Virtual environment for dependencies
└── README.md            # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.11
- PostgreSQL with pgvector extension installed

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd facial_recon
   ```
2. Create a virtual environment:
   ```bash
   python -m venv myvenv
   source myvenv/Scripts/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Database Setup

1. Ensure PostgreSQL is running and pgvector extension is installed.
2. Update the `DATABASE_URL` in `database.py` with your PostgreSQL credentials.
3. Create the database and tables:
   ```bash
   python -c "from database import create_db_and_tables; create_db_and_tables()"
   ```

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
2. Open the web interface at `http://127.0.0.1:8000/static/index.html`.

## API Endpoints

### `/onboard_user`

- **Method**: POST
- **Description**: Onboards a new user by storing their facial embedding.
- **Request Body**:
  ```json
  {
    "name": "John Doe",
    "image_data": "<Base64-encoded image>"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Successfully onboarded John Doe.",
    "user_id": 1,
    "face_encoding_id": 1
  }
  ```

### `/recognize`

- **Method**: POST
- **Description**: Recognizes a face from the provided image.
- **Request Body**:
  ```json
  {
    "image_data": "<Base64-encoded image>"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "recognized_as": "John Doe",
    "distance": 0.45
  }
  ```

## Notes

- This project is experimental and may require additional security measures for production use.
- The facial recognition threshold can be adjusted in `main.py` (`RECOGNITION_THRESHOLD`).

## Future Enhancements

- Add authentication and authorization for API endpoints.
- Improve error handling and logging.
- Extend the web interface for better user experience.
- Optimize database queries for large-scale usage.
