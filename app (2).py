
import os
import fitz
import json
import requests
from duckduckgo_search import DDGS # Added DDGS import

from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

app = FastAPI(
    title="Placement Ready AI Agent",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# GEMMA MODEL
# -------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="models/gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# -------------------------------------------------------
# GLOBAL VARIABLES
# -------------------------------------------------------

resume_text = ""
github_username = ""
target_role = ""

# -------------------------------------------------------
# RESUME PARSER
# -------------------------------------------------------

def extract_resume(pdf_bytes):

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    return text

# -------------------------------------------------------
# TOOL 1
# Resume Analyzer
# -------------------------------------------------------

@tool
def resume_tool(dummy: str) -> str:
    """
    Analyze the uploaded resume.
    """

    global resume_text

    prompt = f"""
You are an ATS Resume Analyzer.

Resume:

{resume_text}

Extract:

1. Skills
2. Programming Languages
3. Frameworks
4. Projects
5. Certifications
6. Strengths
7. Weaknesses

Return structured markdown.
"""

    response = llm.invoke(prompt)

    return response.content

# -------------------------------------------------------
# TOOL 2
# GitHub Analyzer
# -------------------------------------------------------

@tool
def github_tool(username: str) -> str:
    """
    Analyze a GitHub profile.
    """

    user_url = f"https://api.github.com/users/{username}"

    repo_url = f"https://api.github.com/users/{username}/repos"

    user = requests.get(user_url).json()

    repos = requests.get(repo_url).json()

    repo_count = len(repos)

    languages = {}

    stars = 0

    forks = 0

    for repo in repos:

        lang = repo.get("language")

        if lang:
            languages[lang] = languages.get(lang, 0) + 1

        stars += repo.get("stargazers_count", 0)

        forks += repo.get("forks_count", 0)

    report = {
        "Name": user.get("name"),
        "Followers": user.get("followers"),
        "Following": user.get("following"),
        "Public Repositories": repo_count,
        "Stars": stars,
        "Forks": forks,
        "Languages": languages
    }

    return json.dumps(report, indent=4)

# -------------------------------------------------------
# TOOL 3
# Job Search Tool
# -------------------------------------------------------

@tool
def job_search_tool(role:str)->str:
    """
    Search latest placement skills.
    """

    results=[]

    with DDGS() as ddgs:

        for r in ddgs.text(
            f"{role} placement skills 2026",
            max_results=5
        ):
            results.append(r["body"])

    return "\n".join(results)
# -------------------------------------------------------
# TOOL 4
# Skill Gap Analyzer
# -------------------------------------------------------

@tool
def skill_gap_tool(job_requirements: str) -> str:
    """
    Compare the student's resume with job requirements
    and identify missing skills.
    """

    global resume_text

    prompt = f"""
You are an expert placement mentor.

Resume:

{resume_text}

Job Requirements:

{job_requirements}

Compare both and provide:

1. Existing Skills
2. Missing Skills
3. Weak Areas
4. Strong Areas
5. Certifications to Pursue
6. Interview Preparation Topics

Return in markdown.
"""

    response = llm.invoke(prompt)

    return response.content


# -------------------------------------------------------
# TOOL 5
# Project Recommendation Tool
# -------------------------------------------------------

@tool
def project_tool(role: str) -> str:
    """
    Recommend placement-ready projects.
    """

    global resume_text

    prompt = f"""
Target Role:

{role}

Resume:

{resume_text}

Suggest FIVE projects.

For every project include:

Project Title

Description

Tech Stack

Difficulty

Skills Learned

Estimated Completion Time

Keep projects industry level.
"""

    response = llm.invoke(prompt)

    return response.content


# -------------------------------------------------------
# TOOL 6
# Placement Readiness Report
# -------------------------------------------------------

@tool
def placement_report_tool(data: str) -> str:
    """
    Generate the final placement readiness report.
    """

    prompt = f"""
You are a senior campus placement mentor.

Based on all collected information:

{data}

Generate a report with:

Placement Readiness Score (0-100)

Strengths

Weaknesses

Resume Improvements

GitHub Improvements

Projects to Build

Skills to Learn

30-Day Roadmap

60-Day Roadmap

90-Day Roadmap

Final Advice

Return a professional report.
"""

    response = llm.invoke(prompt)

    return response.content


# -------------------------------------------------------
# REGISTER TOOLS
# -------------------------------------------------------

tools = [
    resume_tool,
    github_tool,
    job_search_tool,
    skill_gap_tool,
    project_tool,
    placement_report_tool,
]


# -------------------------------------------------------
# CREATE REACT AGENT
# -------------------------------------------------------

agent = create_react_agent(
    model=llm,
    tools=tools,
)


# -------------------------------------------------------
# AGENT SYSTEM PROMPT
# -------------------------------------------------------

SYSTEM_PROMPT = """
You are an AI Placement Mentor.

Your objective is to prepare students for campus placements.

You have access to the following tools:

1. resume_tool
   Analyze uploaded resume.

2. github_tool
   Evaluate GitHub profile.

3. job_search_tool
   Identify skills expected for the target role.

4. skill_gap_tool
   Compare resume against job requirements.

5. project_tool
   Recommend resume-worthy projects.

6. placement_report_tool
   Generate the final placement report.

IMPORTANT:

Always follow this order:

Step 1:
Analyze Resume

Step 2:
Analyze GitHub

Step 3:
Search Job Requirements

Step 4:
Identify Skill Gap

Step 5:
Recommend Projects

Step 6:
Generate Placement Report

Never skip any step.

Always use the tools.

Never answer directly from your own knowledge unless necessary.
"""


# -------------------------------------------------------
# MAIN AGENT FUNCTION
# -------------------------------------------------------

def run_agent():

    global target_role
    global github_username

    user_prompt = f"""
Target Role:

{target_role}

GitHub Username:

{github_username}

Perform complete placement analysis.

Use every available tool.

Generate the final report.
"""

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=SYSTEM_PROMPT + "\n\n" + user_prompt
                )
            ]
        }
    )

    return response


# -------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "Running",
        "agent": "Placement Ready AI Agent",
        "model": "gemma-4-31b-it"
    }
# -------------------------------------------------------
# ANALYZE ENDPOINT
# -------------------------------------------------------

@app.post("/analyze")
async def analyze(
    role: str = Form(...),
    github: str = Form(...),
    resume: UploadFile = File(...)
):

    global resume_text
    global github_username
    global target_role

    try:

        target_role = role
        github_username = github

        pdf_bytes = await resume.read()

        resume_text = extract_resume(pdf_bytes)

        result = run_agent()

        final_output = ""

        messages = result.get("messages", [])

        if len(messages) > 0:

            final_output = messages[-1].content

        return {

            "success": True,

            "role": target_role,

            "github_username": github_username,

            "resume_characters": len(resume_text),

            "placement_report": final_output

        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)

        }


# -------------------------------------------------------
# GITHUB ONLY ANALYSIS
# -------------------------------------------------------

@app.get("/github/{username}")
def github_analysis(username: str):

    try:

        result = github_tool.invoke(username)

        return {

            "success": True,

            "analysis": result

        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)

        }


# -------------------------------------------------------
# RESUME ONLY ANALYSIS
# -------------------------------------------------------

@app.post("/resume")
async def resume_analysis(

    resume: UploadFile = File(...)

):

    global resume_text

    try:

        pdf = await resume.read()

        resume_text = extract_resume(pdf)

        result = resume_tool.invoke("resume")

        return {

            "success": True,

            "analysis": result

        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)

        }


# -------------------------------------------------------
# JOB SEARCH
# -------------------------------------------------------

@app.get("/jobs/{role}")
def jobs(role: str):

    try:

        result = job_search_tool.invoke(role)

        return {

            "success": True,

            "jobs": result

        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)

        }


# -------------------------------------------------------
# PROJECT RECOMMENDATION
# -------------------------------------------------------

@app.get("/projects/{role}")
def projects(role: str):

    global target_role

    target_role = role

    try:

        result = project_tool.invoke(role)

        return {

            "success": True,

            "projects": result

        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)

        }


# -------------------------------------------------------
# HEALTH
# -------------------------------------------------------

@app.get("/health")
def health():

    return {

        "status": "Healthy",

        "llm": "gemma-4-31b-it",

        "framework": "LangGraph",

        "tools": len(tools)

    }


# -------------------------------------------------------
# AVAILABLE TOOLS
# -------------------------------------------------------

@app.get("/tools")
def available_tools():

    return {

        "tools": [

            "resume_tool",

            "github_tool",

            "job_search_tool",

            "skill_gap_tool",

            "project_tool",

            "placement_report_tool"

        ]

    }


@app.get("/agent")
def agent_info():

    return{

        "Agent":"Placement Ready AI Agent",

        "Framework":"LangGraph",

        "Model":"gemma-4-31b-it",

        "Tools":[
            "resume_tool",
            "github_tool",
            "job_search_tool",
            "skill_gap_tool",
            "project_tool",
            "placement_report_tool"
        ]
    }


# -------------------------------------------------------
# ROOT
# -------------------------------------------------------

@app.get("/")
def root():

    return {

        "message": "Placement Ready AI Agent",

        "version": "1.0",

        "framework": "FastAPI",

        "agent": "LangGraph ReAct",

        "model": "gemma-4-31b-it"

    }


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )

def run_agent():

    global resume_text
    global github_username
    global target_role

    user_query = f"""
You are given:

Target Role:
{target_role}

GitHub Username:
{github_username}

Resume:
{resume_text}

Your task is to help the student become placement ready.

IMPORTANT:

You MUST use the available tools.

Follow this exact workflow.

1. Call resume_tool.

2. Call github_tool.

3. Call job_search_tool.

4. Call skill_gap_tool.

5. Call project_tool.

6. Call placement_report_tool.

After collecting outputs from every tool,
produce ONE comprehensive placement report.

Do not skip any tool.
"""

    result = agent.invoke(
        {
            "messages":[
                HumanMessage(content=user_query)
            ]
        }
    )

    return result
@tool
def github_tool(username: str) -> str:
    """
    Analyze GitHub profile quality.
    """

    user_url = f"https://api.github.com/users/{username}"
    repo_url = f"https://api.github.com/users/{username}/repos"

    user = requests.get(user_url).json()
    repos = requests.get(repo_url).json()

    total_stars = 0
    total_forks = 0
    total_watchers = 0

    languages = {}

    recent_projects = []

    for repo in repos:

        total_stars += repo["stargazers_count"]
        total_forks += repo["forks_count"]
        total_watchers += repo["watchers_count"]

        if repo["language"]:
            languages[repo["language"]] = languages.get(repo["language"],0)+1

        recent_projects.append({
            "name":repo["name"],
            "language":repo["language"],
            "stars":repo["stargazers_count"]
        })

    report=f"""
GitHub Profile

Name : {user.get("name")}

Followers : {user.get("followers")}

Repositories : {len(repos)}

Stars : {total_stars}

Forks : {total_forks}

Watchers : {total_watchers}

Languages Used

{languages}

Projects

{recent_projects[:5]}

Evaluate the GitHub profile.
"""

    response=llm.invoke(report)

    return response.content
