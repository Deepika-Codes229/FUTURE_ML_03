import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SAMPLE DATASET
data = {
    "Candidate_Name": [
        "Rahul",
        "Priya",
        "Arjun",
        "Sneha",
        "Karan"
    ],

    "Resume": [

        "Python SQL Machine Learning Deep Learning Pandas NumPy Communication",

        "Java Excel Communication Teamwork",

        "Python Data Analysis Pandas Machine Learning",

        "Deep Learning TensorFlow Python SQL",

        "HR Management Communication Leadership"
    ]
}

# CREATE DATAFRAME
df = pd.DataFrame(data)

print("\nDATASET CREATED SUCCESSFULLY\n")

print(df)

# CLEAN TEXT FUNCTION
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text

# CLEAN RESUMES
df['cleaned_resume'] = df['Resume'].apply(clean_text)

# SKILLS
skills = [
    "python",
    "machine learning",
    "deep learning",
    "sql",
    "java",
    "tensorflow",
    "excel",
    "communication",
    "pandas",
    "numpy",
    "data analysis"
]

# EXTRACT SKILLS
def extract_skills(text):

    found_skills = []

    for skill in skills:
        if skill.lower() in text.lower():
            found_skills.append(skill)

    return found_skills

df['skills'] = df['cleaned_resume'].apply(extract_skills)

# JOB DESCRIPTION
job_description = """
Looking for a Data Scientist with Python,
Machine Learning, SQL, Pandas, NumPy,
and Deep Learning skills.
"""

cleaned_jd = clean_text(job_description)

# TF-IDF
vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(
    [cleaned_jd] + list(df['cleaned_resume'])
)

# COSINE SIMILARITY
similarity = cosine_similarity(vectors[0:1], vectors[1:])

# SCORES
df['score'] = similarity.flatten()

# JOB SKILLS
job_skills = extract_skills(cleaned_jd)

# MISSING SKILLS
def missing_skills(candidate_skills):
    return list(set(job_skills) - set(candidate_skills))

df['missing_skills'] = df['skills'].apply(missing_skills)

# RANKING
ranked_df = df.sort_values(by='score', ascending=False)

print("\nFINAL CANDIDATE RANKING\n")

for index, row in ranked_df.iterrows():

    print("Candidate:", row['Candidate_Name'])

    print("Resume:", row['Resume'])

    print("Score:", round(row['score'], 2))

    print("Skills:", row['skills'])

    print("Missing Skills:", row['missing_skills'])

    print("-" * 60)
    


