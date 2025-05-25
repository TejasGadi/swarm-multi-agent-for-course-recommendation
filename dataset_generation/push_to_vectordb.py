import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import Field, BaseModel
from tqdm.auto import tqdm

# === CONFIG ===
PINECONE_INDEX_NAME = "course-index"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY")

class CourseSchema(BaseModel):
    """Model for course schema"""
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


# === Convert course data to LangChain Documents ===
def convert_to_documents(courses: List[dict]) -> List[Document]:
    documents = []

    for course in tqdm(courses):
        lines = []

        # Title and Description first
        title = course.get('title', '')
        description = course.get('description', '')
        lines.append(f"Title: {title}")
        lines.append(f"Description: {description}")

        # Add rest as full sentences
        lines.append(f"This course is provided by {course.get('provider_of_course', 'an unknown provider')}.")
        lines.append(f"It requires a daily commitment of {course.get('daily_commitment_hours', 'unspecified time')}.")
        lines.append(f"The course starts on {course.get('start_date', 'an unspecified date')}.")
        lines.append(f"It runs for a duration of {course.get('duration', 'an unspecified period')}.")
        lines.append(f"Classes are scheduled during the {course.get('time_of_the_day', 'unspecified time of day')}.")
        lines.append(f"The course is delivered in {course.get('mode', 'unspecified')} mode.")
        lines.append(f"The medium of instruction is {course.get('language', 'unspecified')}.")
        
        prerequisites = course.get('prerequisites', [])
        if prerequisites:
            lines.append(f"Prerequisites for this course include: {', '.join(prerequisites)}.")
        else:
            lines.append("There are no specific prerequisites for this course.")
        
        lines.append(f"It is suitable for learners at the {course.get('suitable_academic_level_required', 'unspecified')} level.")
        lines.append(f"The course has a rating of {course.get('course_rating', 'unrated')} out of 5.")

        content = "\n".join(lines)
        documents.append(Document(page_content=content, metadata=course))

    return documents

def push_to_pinecone(courses: List[dict]):
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in environment.")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    documents = convert_to_documents(courses)
    
    # Process documents in batches of 100
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Pushed batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size} to Pinecone")
    
    print(f"Successfully pushed all {len(documents)} courses to Pinecone index '{PINECONE_INDEX_NAME}'.")

# === Main ===
if __name__ == "__main__":
    # === Sample Course Data ===
    # You can also load this from a file
    file_path = "./output_data_jsonl/udemy_courses.jsonl"

    courses_dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            courses_dataset.append(json.loads(line))
    push_to_pinecone(courses_dataset)