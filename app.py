import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE SETTINGS
st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)

# TITLE
st.title("AI Resume Screening System")
st.write("Resume Screening using NLP & Machine Learning")

# SIDEBAR
st.sidebar.title("AI Resume Screening System")

st.sidebar.info("""
This project uses:

✔ NLP  
✔ TF-IDF  
✔ Cosine Similarity  
✔ Skill Extraction  
✔ Candidate Ranking  
✔ Missing Skill Analysis  

Built using Python & Streamlit.
""")

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

# SKILLS LIST
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

# SKILL EXTRACTION FUNCTION
def extract_skills(text):

    found_skills = []

    for skill in skills:

        if skill.lower() in text.lower():

            found_skills.append(skill)

    return found_skills

# EXTRACT SKILLS
df['skills'] = df['cleaned_resume'].apply(extract_skills)

# JOB DESCRIPTION INPUT
job_description = st.text_area(
    "Enter Job Description",
    "Looking for a Data Scientist with Python, Machine Learning, SQL, Pandas, NumPy and Deep Learning skills."
)

# BUTTON
if st.button("Screen Resumes"):

    # CLEAN JOB DESCRIPTION
    cleaned_jd = clean_text(job_description)

    # TF-IDF VECTORIZATION
    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [cleaned_jd] + list(df['cleaned_resume'])
    )

    # COSINE SIMILARITY
    similarity = cosine_similarity(vectors[0:1], vectors[1:])

    # ADD SCORES
    df['score'] = similarity.flatten()

    # JOB SKILLS
    job_skills = extract_skills(cleaned_jd)

    # MISSING SKILLS FUNCTION
    def missing_skills(candidate_skills):

        return list(set(job_skills) - set(candidate_skills))

    # ADD MISSING SKILLS
    df['missing_skills'] = df['skills'].apply(missing_skills)

    # SORT CANDIDATES
    ranked_df = df.sort_values(
        by='score',
        ascending=False
    )

    # SHOW RESULTS
    st.subheader("Top Ranked Candidates")

    for index, row in ranked_df.iterrows():

        st.markdown("---")

        st.markdown(
            f"## 👤 Candidate: {row['Candidate_Name']}"
        )

        st.write("### Resume")
        st.info(row['Resume'])

        st.write("### Matching Score")

        score = round(row['score'], 2)

        if score > 0.5:

            st.success(score)

        elif score > 0.2:

            st.warning(score)

        else:

            st.error(score)

        st.write("### Skills")

        st.write(row['skills'])

        st.write("### Missing Skills")

        if len(row['missing_skills']) == 0:

            st.success("No Missing Skills")

        else:

            st.error(row['missing_skills'])

    # GRAPH
    st.subheader("Candidate Ranking Graph")

    top_candidates = ranked_df.head(5)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        top_candidates['Candidate_Name'],
        top_candidates['score']
    )

    ax.set_xlabel("Candidates")

    ax.set_ylabel("Matching Score")

    ax.set_title("Top Candidate Rankings")

    st.pyplot(fig)

    # BEST CANDIDATE
    best_candidate = ranked_df.iloc[0]

    st.subheader("Best Candidate Selected")

    st.success(
        f"{best_candidate['Candidate_Name']} "
        f"with score {round(best_candidate['score'], 2)}"
    )

    # DOWNLOAD BUTTON
    csv = ranked_df.to_csv(index=False)

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name='screened_candidates.csv',
        mime='text/csv'
    )

# FOOTER
st.markdown("---")

st.caption("Developed using Machine Learning & NLP")


    
    
    
    