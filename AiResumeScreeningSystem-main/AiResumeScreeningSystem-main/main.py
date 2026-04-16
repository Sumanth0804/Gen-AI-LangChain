import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from chains.extract_chain import create_extract_chain
from chains.match_chain import create_match_chain
from chains.score_chain import create_score_chain
from chains.explain_chain import create_explain_chain

load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Create chains
extract_chain = create_extract_chain(llm)
match_chain = create_match_chain(llm)
score_chain = create_score_chain(llm)
explain_chain = create_explain_chain(llm)

# Load job description
job_description = open("job_description.txt").read()

resumes = {
    "Strong": "resumes/strong.txt",
    "Average": "resumes/average.txt",
    "Weak": "resumes/weak.txt"
}

for label, path in resumes.items():

    print("\n==============================")
    print(f"Evaluating {label} Candidate")
    print("==============================")

    resume = open(path).read()

    extracted = extract_chain.invoke({
        "resume": resume
    })

    matched = match_chain.invoke({
        "skills": extracted,
        "job_description": job_description
    })

    score = score_chain.invoke({
        "matched_skills": matched,
        "missing_skills": matched
    })

    explanation = explain_chain.invoke({
        "score": score,
        "matched_skills": matched,
        "missing_skills": matched
    })

    print("Extraction:", extracted)
    print("Matching:", matched)
    print("Score:", score)
    print("Explanation:", explanation)