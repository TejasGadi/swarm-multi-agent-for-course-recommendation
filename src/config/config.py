"""Configuration settings for the Course Recommendation System."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Model Configuration
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Vector DB Configuration
VECTOR_DB_PATH = "data/vector_db"

# Agent Configuration
MAX_ITERATIONS = 10
TEMPERATURE = 0.7

# Course Metadata Schema
COURSE_METADATA_SCHEMA = {
    "title": str,
    "provider": str,
    "duration": str,
    "daily_commitment": str,
    "start_date": str,
    "schedule": str,
    "mode": str,
    "language": str,
    "prerequisites": list,
    "level": str,
    "cost": str,
    "certification": bool,
    "career_outcomes": list
}

# Student Profile Schema
STUDENT_PROFILE_SCHEMA = {
    "education_level": str,
    "academic_background": list,
    "interests": list,
    "preferred_mode": str,
    "availability": dict,
    "career_goals": list
} 