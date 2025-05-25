import os
import json
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Step 1: Define CourseSchema for structured output
class CourseSchema(BaseModel):
    title: str = Field(..., description="Name of the course")
    description: str = Field(..., description="Description of the course")
    provider_of_course: str = Field(..., description="Provider/Platform or institution of the course")
    daily_commitment_hours: str = Field(..., description="e.g., 2 hours/day")
    start_date: str = Field(..., description="Start date i.e. When it begins")
    duration: str = Field(..., description="Total length of course (e.g., 6 months or 4 years)")
    time_of_the_day: str = Field(..., description="Time of day (e.g., evening)")
    mode: str = Field(..., description="online / offline / hybrid")
    language: str = Field(..., description="Medium of instruction")
    prerequisites: List[str] = Field(default_factory=list, description="Required skills/knowledge List")
    suitable_academic_level_required: str = Field(..., description="Suitable academic level required")
    course_rating: str = Field(..., description="Course rating in range 1 to 5 (float values)")

# Step 2: Generate prompt
def generate_prompt(course: dict) -> str:
    return (
        "You are an education advisor AI. Based on the course information below, "
        "extract the fields for a course schema. Return a valid JSON with the following fields:\n"
        "- title\n"
        "- description\n"
        "- provider_of_course\n"
        "- daily_commitment_hours\n"
        "- start_date\n"
        "- duration\n"
        "- time_of_the_day\n"
        "- mode\n"
        "- language\n"
        "- prerequisites (as list of strings)\n"
        "- suitable_academic_level_required\n"
        "- course_rating (string between 1.0 to 5.0)\n\n"
        "Available input data is as follows, extract the fields and add additional data(most likely value) in missings fields on your own"
        f"Course Info:\n"
        f"Title: {course['title']}\n"
        f"Description: {course['description']}\n"
        f"Instructor: {course['instructor']}\n"
        f"Rating: {course['rating']}\n"
        f"Review Count: {course['reviewcount']}\n"
        f"Duration: {course['duration']}\n"
        f"Lectures: {course['lectures']}\n"
        f"Level: {course['level']}\n"
    )

# Step 3: Process CSV and generate JSONL
def process_courses(input_csv: str, output_jsonl: str):
    df = pd.read_csv(input_csv)
    llm = ChatGroq(model="gemma2-9b-it", temperature=0.2)
    structured_llm = llm.with_structured_output(CourseSchema)
    start_index=200

    with open(output_jsonl, "w", encoding="utf-8") as outfile:
        for _, row in tqdm(list(df.iloc[start_index:].iterrows()), total=len(df) - start_index,
            initial=start_index, desc="Processing Courses"):
            course_data = row.to_dict()
            prompt = generate_prompt(course_data)
            try:
                result = structured_llm.invoke([{"role": "user", "content": prompt}])
                outfile.write(result.json() + "\n")
                print(f"Processed: {course_data['title']}")
            except Exception as e:
                print(f"Failed to process course '{course_data['title']}': {e}")

# Step 4: Run the pipeline
if __name__ == "__main__":
    input_csv = "./input_data_csv/udemy_courses.csv"

    output_jsonl = "./output_data_jsonl/udemy_courses_2.jsonl"

    process_courses(input_csv, output_jsonl)