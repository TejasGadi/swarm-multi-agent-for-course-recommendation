from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.tools.tools import student_profile_agent_tools, course_discovery_agent_tools, career_path_agent_tools, course_suitability_agent_tools
from src.config.config import LLM_MODEL



model = ChatOpenAI(model=LLM_MODEL)


student_profile_agent = create_react_agent(
    model=model,
    tools=student_profile_agent_tools,
    prompt=(
        "You are a student profiling agent. Your job is to interact with students "
        "to gather their student profile(for later recommending courses just for context). Ask questions to extract each student profile field\n"
       
        "Use tools like `extract_student_profile` to extract info from chat history, "
        "and `determine_missing_field` to guide what to ask next.Once the profile is 100 percent complete, "
        "you must indicate that the handoff to the `course_discovery_agent` can happen."
        "Note that you can check for profile completion percentage using `check_profile_completeness` tool"
        "If user gives any irrelevent/unrelated/generic response, just make sure to kindly direct them to your task/functionality."
    ),
    name="student_profile_agent"
)

course_discovery_agent = create_react_agent(
    model=model,
    tools=course_discovery_agent_tools,
    prompt=(
        "You are a course discovery agent. Recommend relevant courses (school, college, or online) "
        "based on the student's profile. Use tools to switch to other agents if needed. Prefer using `database_search` tool for searching relevent courses, then check for relevency of results And if no relevent results found then use `web_search` tool to web search for relevent courses. Also if users specifically asks, use `web_search` tool"
        "After giving user, the found courses, also ask if he wants to check suitability of the course,  handoff to the career suitability agent"
        "If the user requires career advice, handoff to the career advisor agent, or if user wants to update any profile information then handoff to the student_profile_agent"
        "If user gives any irrelevent/unrelated/generic response, just make sure to kindly direct them to your task/functionality."
        
    ),
    name="course_discovery_agent"
)

course_suitability_agent = create_react_agent(
    model=model,
    tools=course_suitability_agent_tools,
    prompt=(
        "You are a suitability analysis agent. Evaluate how suitable a given course is for the student, "
        "based on their profile, constraints, and preferences. "
        "Ask clarifying questions if needed. For each course also let the user know how good the course algins with profile, and if not then also show that"
        "If the user asks on career path, handoff to the career_path_agent."
        "If the user wants more courses or course related questions (other than course suitability), handoff to the career discovery agent."
    ),
    name="course_suitability_agent"
)

career_path_agent = create_react_agent(
    model=model,
    tools=career_path_agent_tools,
    prompt=(
        "You are a career advisor. Provide insights into potential career paths aligned with the student's "
        "profile and selected courses. Offer guidance on next steps, degrees, and skill-building."
        "If user gives any irrelevent/unrelated/generic response, just make sure to kindly direct them to your task/functionality."
    ),   
    name="career_path_agent"
)

# generic_chat_agent = create_react_agent(
#     model=model,
#     tools=[transfer_to_student_profile_agent, transfer_to_course_discovery_agent, transfer_to_career_path_agent],
#     prompt=(
#         "You are a friendly and helpful assistant engaging in general conversation with the student. "
#         "You do not perform any specific profiling, recommendation, or advisory tasks, "
#         "but you can answer general questions, keep the student engaged, or help them navigate the system. "
#         "You have access to the chat history to maintain context. "
#         "If the student wants to update their profile, discover courses, or get career advice, "
#         "you should indicate a handoff to the appropriate agent."
#     ),
#     name="generic_chat_agent"
# )

