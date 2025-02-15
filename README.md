# Course Recommendation System

## Overview
This is a real-time course recommendation system that suggests relevant courses based on user inputs, such as education, skills, career goals, and learning objectives.

## How to Use??

### Send POST Request
Send a POST request to `/predict` with a JSON body:
```json
{
    "education": "Btech",
    "skills": "Python, AI",
    "career_goals": "Python developer, web analyst",
    "learning_objectives": "Better learning and career opportunities",
    "top_n": 5
}
```

### Response Format
```json
{
    "status": "success",
    "message": "Found 5 relevant courses",
    "recommendations": [
        {
            "course_id": 631128,
            "title": "Complete Python Web Course",
            "similarity": 85.5,
            "url": "https://...",
            "level": "Beginner",
            "duration": 8.5
        }
    ]
}
```

## Installation
To set up the project, install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application
Run the Flask app:
```bash
python app.py
```

## API Endpoints
- `POST /predict` - Get course recommendations based on user input.

## Technologies Used
- Python, Flask
- Sentence Transformers (SBERT)
- MySQL
- Scikit-learn
- MLflow
- DVC for data versioning

## Future Enhancements
- Collaborative filtering
- User feedback loop
- Mobile app integration