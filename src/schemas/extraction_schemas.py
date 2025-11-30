"""
Structured output schemas for LLM extraction
"""
from typing import List, Optional
from pydantic import BaseModel, Field

# Registry to map schema names to their Pydantic models
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {}


def register_schema(name: str, schema_class: type[BaseModel]):
    """Register a schema class with a name"""
    SCHEMA_REGISTRY[name] = schema_class


def get_schema(name: str) -> Optional[type[BaseModel]]:
    """Get a schema class by name"""
    return SCHEMA_REGISTRY.get(name)


def get_schema_json_schema(name: str) -> Optional[dict]:
    """Get the JSON schema for a registered schema name"""
    schema_class = get_schema(name)
    if schema_class:
        return schema_class.model_json_schema()
    return None


# Job Posting Schema
class JobPosting(BaseModel):
    """Schema for extracting detailed job posting information"""
    job_title: Optional[str] = Field(None, description="Official job title")
    location: Optional[str] = Field(None, description="Job location or city/country")
    employment_type: Optional[str] = Field(None, description="Full-time, part-time, contract, etc.")
    department: Optional[str] = Field(None, description="Team or department name")

    company_name: Optional[str] = Field(None, description="Company or hiring organization")
    job_overview: Optional[str] = Field(None, description="Short overview/summary of the role")

    responsibilities: List[str] = Field(
        default_factory=list,
        description="Main responsibilities or duties, as bullet points or sentences",
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Required hard/technical skills and tools",
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Nice-to-have or preferred skills",
    )
    soft_skills: List[str] = Field(
        default_factory=list,
        description="Soft skills such as communication, teamwork, leadership",
    )

    qualifications: List[str] = Field(
        default_factory=list,
        description="Formal qualifications such as degrees, certifications, years of experience",
    )
    languages: List[str] = Field(
        default_factory=list,
        description="Language requirements with proficiency levels if specified",
    )
    benefits: List[str] = Field(
        default_factory=list,
        description="Main benefits and perks mentioned in the posting",
    )

    application_instructions: Optional[str] = Field(
        None, description="How to apply or what to send"
    )


# Register the schema
register_schema("job_details", JobPosting)

# Default extraction instruction for job_details
JOB_DETAILS_INSTRUCTION = (
    "Extract all job information from the webpage content below. "
    "Read the entire page carefully and extract the following information:\n\n"
    "1. JOB TITLE: Find the main job title/position name (usually in a heading or title section)\n"
    "2. LOCATION: Extract city, country, or location where the job is based\n"
    "3. EMPLOYMENT TYPE: Find if it's full-time, part-time, contract, permanent, etc.\n"
    "4. DEPARTMENT: Identify the team, department, or division name\n"
    "5. COMPANY NAME: Extract the hiring company name\n"
    "6. JOB OVERVIEW: Create a brief 2-3 sentence summary of what this role entails\n"
    "7. RESPONSIBILITIES: Extract all main duties, tasks, and responsibilities (look for bullet lists, numbered lists, or paragraphs under sections like 'Responsibilities', 'What you will do', 'Your role', 'Key tasks', etc.)\n"
    "8. REQUIRED SKILLS: Extract all mandatory technical skills, tools, technologies, frameworks, programming languages, software, platforms mentioned as required or must-have\n"
    "9. PREFERRED SKILLS: Extract skills mentioned as nice-to-have, preferred, or bonus\n"
    "10. SOFT SKILLS: Extract interpersonal skills like communication, leadership, teamwork, problem-solving, etc.\n"
    "11. QUALIFICATIONS: Extract education requirements (degrees), certifications, years of experience, specific experience requirements\n"
    "12. LANGUAGES: Extract language requirements with proficiency levels if mentioned (e.g., 'English B2', 'Czech native')\n"
    "13. BENEFITS: Extract all benefits, perks, compensation details mentioned\n"
    "14. APPLICATION INSTRUCTIONS: Extract how to apply, what documents to send, application process\n\n"
    "IMPORTANT: Extract actual content from the page. Do not leave fields empty if the information exists. "
    "For list fields (responsibilities, skills, etc.), extract each item as a separate string. "
    "Be thorough and extract all available information. If a field is truly not present on the page, you may leave it null or empty."
)

# Instruction registry
INSTRUCTION_REGISTRY: dict[str, str] = {
    "job_details": JOB_DETAILS_INSTRUCTION,
}


def get_extraction_instruction(schema_name: str) -> Optional[str]:
    """Get the extraction instruction for a schema"""
    return INSTRUCTION_REGISTRY.get(schema_name)

