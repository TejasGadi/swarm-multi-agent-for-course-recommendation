from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from typing import List
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config.config import OPENAI_API_KEY, LLM_MODEL, TAVILY_API_KEY

# --- Define Input Schema for the tool ---
class CourseValidationInput(BaseModel):
    courses: List[str] = Field(..., description="List of courses i.e. Metadata of the courses to validate, mention all courses list, each course in form of detailed text")


from typing import Dict
import json

class CourseSuitability(BaseModel):
    courseName: str = Field(
        ...,
        title="Course Name",
        description="The name of the course being evaluated for suitability."
    )
    suitabilityDescription: str = Field(
        ...,
        title="Suitability Description",
        description="A detailed explanation of how well the course aligns (or does not align) with the student's profile."
    )

def validate_course_tool(input: CourseValidationInput) -> Dict:
    # Load the profile from file
    with profile_json_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)

    # Convert the profile dict into a readable string format (optional but useful)
    profile_text = ". ".join(f"{k}: {v}" for k, v in profile.items())

    # Construct the validation query
    prompt = (
        "You are a suitability analysis agent. Evaluate how suitable each given course is for the student, "
        "based on their profile, constraints, and preferences. "
        "If necessary, ask clarifying questions. For each course, indicate how well it aligns with the profile. "
        "If it doesn't align, explain why.\n\n"
        f"Courses: {input.courses}\n"
        f"Student Profile: {profile_text}"
    )

    structured_llm = llm.with_structured_output(List[CourseSuitability])
    # Call the LLM
    result = structured_llm.invoke([{"role": "user", "content": prompt}])

    return {"suitability_result": result.dict()}

course_suitability_agent_tools=[
    Tool(
        name="validate_course_tool",
        func=validate_course_tool,
        description="Validate course suitability against a student's profile and provide matches, mismatches",
        args_schema=CourseValidationInput
    )
]

retriever = PineconeVectorStore.from_existing_index(index_name="course-index", embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY)).as_retriever(search_kwargs={"k": 8})

retriever_tool = create_retriever_tool(
    retriever,
    "database_search",
    "A tool to semantically search the course database using natural language queries. Make sure the when calling this tool, you MUST provide a natural language query that reflects course preferences, constraints, or requirements.",
)

class TavilySearchInput(BaseModel):
    query: str = Field(..., description="The query to search the web for relevant course information.")


course_discovery_agent_tools=[
    Tool(
        name="web_search",
        func=TavilySearchResults(max_results=5),
        description="A tool to search the web for relevant course information.",
        args_schema=TavilySearchInput
    ),
    retriever_tool,
]


# --- Career Path Agent Tools Implementation

llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)

class CareerPathAnalysisInput(BaseModel):
    query: str = Field(..., description="The query to analyze the career path.")

def ask_llm(query: str) -> str:
    return llm.invoke(query)

career_path_agent_tools=[
    Tool(
        name="career_path_analysis",
        func=ask_llm,
        description="A tool to give insights into the career path as per selected courses or user's ask in the query",
        args_schema=CareerPathAnalysisInput
    )
]


## StudentProfile schema

from typing import List, Optional, Dict
from pydantic import BaseModel

class StudentProfile(BaseModel):
    name: str = Field(..., description="Full name of the student.")
    educational_level: str = Field(..., description="Current level of education, e.g., High School, Undergraduate, etc.")
    age: Optional[int] = Field(None, description="Age of the student.")
    course_description: str = Field(..., description="A short description of the course the student is looking for.")
    course_mode: str = Field(..., description="Preferred mode of learning: online, offline, hybrid, or any.")
    daily_hours: Optional[int] = Field(None, description="Number of hours the student can dedicate to learning each day.")
    preferred_timing: str = Field(..., description="Preferred time of the day to take the course: morning, afternoon, or evening.")
    max_duration_months: Optional[int] = Field(None, description="Maximum duration (in months) the student is willing to commit to the course.")
    language: List[str] = Field(default_factory=lambda: ["English"], description="Preferred languages for the course content.")
    certification_needed: bool = Field(default=False, description="Whether the student wants a certification after completing the course.")
    location_preference: str = Field(default="any", description="Location preference for the course: online, offline, hybrid, or any.")


# Input schema for extract_student_profile tool
from typing import List, Dict, Optional, Any
import json
from pathlib import Path

from pydantic import BaseModel, Field

profile_json_path = Path("./profile.json")

class ExtractProfileInput(BaseModel):
    conversation: str = Field(
        ..., description="Flattened conversation between student and agent as a single string."
    )


def extract_student_profile(conversation: str) -> dict:
    """Extract structured student profile data from plain conversation text."""
    prompt = (
        "You are an expert profile extractor. Extract the following fields from the conversation below.\n"
        "Return a valid JSON with the fields (use null/empty values where not mentioned):\n"
        "- name\n"
        "- educational_level\n"
        "- age\n"
        "- course_description (as a detailed string)\n"
        "- course_mode\n"
        "- daily_hours\n"
        "- preferred_timing\n"
        "- max_duration_months\n"
        "- language (as list)\n"
        "- certification_needed (true/false)\n"
        "- location_preference\n\n"
        "Conversation:\n"
        f"{conversation}"
    )
    structured_llm = llm.with_structured_output(StudentProfile)
    partial_profile_response = structured_llm.invoke([{"role": "user", "content": prompt}])

    try:
        new_profile = json.loads(partial_profile_response.content)
        new_profile = StudentProfile(**profile).dict()
    except Exception:
        new_profile = None

    
    new_profile = partial_profile_response.dict()

    # Save to file
    with profile_json_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)

    # Newly extract fields(new_profile) add to profile, given that add only null/empty fields
    # Merge: only update missing fields
    for key, value in new_profile.items():
        if key in profile:
            profile[key] = value

    # Save updated profile
    with profile_json_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    total_fields = len(profile)
    filled_fields = sum(1 for v in profile.values() if v not in [None, "","null", [], {}])
    profile["completion_percent"] = round((filled_fields / total_fields) * 100)

    return profile


# Tool

class EmptyInput(BaseModel):
    description: str = Field(..., description="Description of tool action")


def check_profile_completeness(description: str) -> Dict[str, Any]:
    """Check what percentage of the profile is completed by reading from file."""
    required_fields = [
        "name", "educational_level", "age", "course_description", "course_mode",
        "daily_hours", "preferred_timing", "max_duration_months", "language",
        "certification_needed", "location_preference"
    ]
    
    # Load the profile from file
    with profile_json_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)
    
    total_fields = len(required_fields)
    filled_fields = sum(
        1 for field in required_fields 
        if profile.get(field) not in [None, "", [], {}]
    )

    percent_complete = round((filled_fields / total_fields) * 100)
    return {
        "percent_complete": percent_complete,
        "is_complete": percent_complete == 100
    }



def determine_next_missing_field(description: str) -> Optional[str]:
    """Determine the next field to ask about based on missing data."""
    ordered_fields = [
        "name", "educational_level", "age", "course_description", "course_mode",
        "daily_hours", "preferred_timing", "max_duration_months", "language",
        "certification_needed", "location_preference"
    ]

     # Load the profile from file
    with profile_json_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)

    for field in ordered_fields:
        value = profile.get(field)
        if value in [None, "", [], {}]:
            return field  # Return the first missing field

    return None  # All fields are filled

student_profile_agent_tools=[
    Tool(
        name="extract_student_profile",
        func=extract_student_profile,
        description="Extract structured student profile data from a conversation chat history.",
        args_schema=ExtractProfileInput
    ),
    Tool(
        name="check_profile_completeness",
        func=check_profile_completeness,
        description="Check how much percentage fields are completed in the student profile.",
        args_schema=EmptyInput
    ),
    Tool(
        name="determine_next_missing_field",
        func=determine_next_missing_field,
        description="Determine the next missing profile field to ask the student/user",
        args_schema=EmptyInput
    )
]


from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool

# --- Handoff Tools ---
transfer_to_course_discovery_agent = create_handoff_tool(
    agent_name="course_discovery_agent",
    description="Transfer user to the course discovery agent."
)
# transfer_to_course_suitability_agent = create_handoff_tool(
#     agent_name="course_suitability_agent",
#     description="Transfer user to the course suitability agent."
# )
transfer_to_career_path_agent = create_handoff_tool(
    agent_name="career_path_agent",
    description="Transfer user to the career path agent."
)
transfer_to_student_profile_agent = create_handoff_tool(
    agent_name="student_profile_agent",
    description="Transfer user to the student profile agent."
)

# Adding handoff tools to tools
student_profile_agent_tools.append(transfer_to_course_discovery_agent)

course_discovery_agent_tools.append(transfer_to_career_path_agent)
course_discovery_agent_tools.append(transfer_to_student_profile_agent)

course_suitability_agent_tools.append(transfer_to_career_path_agent)

career_path_agent_tools.append(transfer_to_student_profile_agent)