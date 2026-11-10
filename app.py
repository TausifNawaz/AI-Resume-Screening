import streamlit as st
import matplotlib.pyplot as plt
from sklearn .feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag


# Dawnload NLTK Resources

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")

# page Setup
st.set_page_config(
    page_title="AI Resume Screening and Interview Analysis" ,
    page_icon="📄" ,
    layout="wide"

)

st.markdown("""
# AI-Based Resume Screening & Interview Analysis

Welcome! This intelligent system helps recruiters and job seekers evaluate resume compatibility with job descriptions.

📄 Upload your resume (PDF)
📝 Paste the job description
🤖 AI will analyze skills, keywords, and experience
📊 View the Resume Match Score.
""")

with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:

    -Measures how your resume matches a job description

    -Identify important job keywords

    -Improve your resume based on missing terms

    """)
    st.header("How It Works")
    st.write("""
    1. Upload your resume(PDF)
    2. Paste the job description
    3. Click **Analyze Match**
    4.Review score & suggestion
    """)

   # ---------------- Functions ---------------- #
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reder = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reder.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF:{e}")
        return ""

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    text=re.sub(r'\s+',' ',text).strip()
    return text

def remove_stopwords(text):
    stop_words=set(stopwords.words('english'))
    words=word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_text,job_description):
    resume_processed=remove_stopwords(clean_text(resume_text))
    job_processed=remove_stopwords(clean_text(job_description))
    vectorizer=TfidfVectorizer()
    tfidf_matrix=vectorizer.fit_transform([resume_processed,job_processed])
    score=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:2])[0][0]*100
    return round(score,2),resume_processed,job_processed

def main():

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)", type=["pdf"]
    )

    job_description = st.text_area(
        "Paste the job description", height=200
    )

    if st.button("Analyze Match"):
        if not uploaded_file:
             st.warning("please upload your resume")
             return
        if not job_description:
             st.warning("please paste the job discription")
             return

        
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text:
                st.error("Could not extract text from PDF. Please try another PDF.")
                return
            # Calculate Similarity
            similarity_score, resume_processed, job_processed=calculate_similarity(
                resume_text,
                job_description
            )
            st.session_state["resume_text_for_ai"] = resume_text
            st.session_state["job_description_for_ai"] = job_description
            st.session_state["resume_match_score_for_ai"] = similarity_score
               # ==========================
                       # Results
               # ==========================

            st.subheader("📊 Results")
            st.metric("Resume Match Score", f"{similarity_score:.2f}%")

                
            

                # Colorful Match Score Chart
            fig, ax = plt.subplots(figsize=(12, 2.5))

                # Background zones
            ax.barh(["Match Score"], [40], color="red", alpha=0.25)
            ax.barh(["Match Score"], [20], left=40, color="orange", alpha=0.35)
            ax.barh(["Match Score"], [20], left=60, color="gold", alpha=0.40)
            ax.barh(["Match Score"], [20], left=80, color="green", alpha=0.35)

                # Actual score
            if similarity_score < 40:
             score_color = "red"
            elif similarity_score < 60:
             score_color = "orange"
            elif similarity_score < 80:
             score_color = "gold"
            else:
             score_color = "green"

            ax.barh(
            ["Match Score"],
            [similarity_score],
            color=score_color,
            height=0.35
            )

             # Score text
            ax.text(
             similarity_score,
             0,
             f"  {similarity_score:.2f}%",
             va="center",
             ha="left",
             fontsize=12,
             fontweight="bold"
             )

            ax.set_xlim(0, 100)
            ax.set_xlabel("Match Percentage")
            ax.set_title("Resume Job Match", fontsize=18, fontweight="bold")

             # Remove unnecessary borders
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

                # Match Status
            if similarity_score < 50:
              st.error("❌ Low Match")
            elif similarity_score < 75:
             st.warning("⚠️ Good Match")
            else:
             st.success("✅ Excellent Match")

             st.subheader("💡 Resume Suggestions")

        if similarity_score < 50:
             st.write("""
                • Add more technical skills related to the job.
                • Include relevant projects.
                • Improve your resume summary.
                • Use keywords from the job description.
                   """)
        elif similarity_score < 75:
             st.write("""
                    • Your resume matches fairly well.
                    • Add more relevant skills and certifications.
                    • Highlight project achievements.
                    """)
        else:
             st.write("""
                     • Excellent resume!
                     • Keep updating your projects and skills.
                     """)

                    # ==========================
                          # Matched Skills
                    # ==========================

        skill_keywords = [
            "python", "java", "c", "c++", "html", "css",
            "javascript", "react", "node", "sql",
            "mysql", "mongodb", "git", "github",
            "communication", "teamwork", "problem solving"
                 ]

        
        matched_skills = []
        missing_skills = []

        resume_lower = resume_text.lower()
        jd_lower = job_description.lower()

        for skill in skill_keywords:
             if skill in jd_lower:
              if skill in resume_lower:
               matched_skills.append(skill.title())
             else: 
               missing_skills.append(skill.title())

        st.subheader("🟩 Matched Skills")

        if matched_skills:
           st.success(", ".join(matched_skills))
        else:
             st.warning("No matched skills found.")

        st.subheader("🟥 Missing Skills")

        if missing_skills:
             st.error(", ".join(missing_skills))
        else:
             st.success("No Missing Skills 🎉")

    st.divider()

    # ==============================
    # Advanced Interview Analysis
    # ==============================

    st.header("Interview Analysis")
    st.write(
        "Evaluate interview answers using relevance, answer quality, technical content, "
        "communication structure, and confidence indicators."
    )

    questions = [
        "Tell me about yourself.",
        "Explain one of your projects and your role in it.",
        "What are your strongest technical skills?",
        "Describe a challenging problem you faced in a project and how you solved it.",
        "Why should we hire you for this position?"
    ]

    # Q1
    st.subheader("Q1. Tell me about yourself.")
    answer1 = st.text_area(
        "Your Answer",
        placeholder="Type your answer here...",
        key="answer1"
    )

    # Q2
    st.subheader("Q2. Explain one of your projects and your role in it.")
    answer2 = st.text_area(
        "Your Answer",
        placeholder="Type your answer here...",
        key="answer2"
    )

    # Q3
    st.subheader("Q3. What are your strongest technical skills?")
    answer3 = st.text_area(
        "Your Answer",
        placeholder="Type your answer here...",
        key="answer3"
    )

    # Q4
    st.subheader("Q4. Describe a challenging problem you faced in a project and how you solved it.")
    answer4 = st.text_area(
        "Your Answer",
        placeholder="Type your answer here...",
        key="answer4"
    )

    # Q5
    st.subheader("Q5. Why should we hire you for this position?")
    answer5 = st.text_area(
        "Your Answer",
        placeholder="Type your answer here...",
        key="answer5"
    )

    if st.button("Analyze Interview"):

        answers = [answer1, answer2, answer3, answer4, answer5]

        keyword_groups = [
            ["education", "degree", "skills", "experience", "project", "goal", "career", "strength"],
            ["project", "role", "technology", "developed", "contribution", "challenge", "result", "team"],
            ["python", "java", "c++", "javascript", "react", "node", "sql", "mysql", "mongodb", "html", "css", "git"],
            ["challenge", "problem", "solution", "debugging", "testing", "solved", "result", "learned"],
            ["skills", "experience", "team", "communication", "leadership", "problem solving", "contribution", "company", "role"]
        ]

        technical_words = [
            "python", "java", "c++", "javascript", "react", "node", "sql",
            "mysql", "mongodb", "html", "css", "git", "github", "api",
            "database", "machine learning", "testing", "debugging"
        ]

        positive_confidence_words = [
            "achieved", "improved", "successfully", "led", "built",
            "developed", "solved", "implemented", "delivered", "confident"
        ]

        hesitation_words = [
            "maybe", "perhaps", "i think", "i guess", "probably",
            "not sure", "kind of", "sort of"
        ]

        answered_questions = sum(1 for answer in answers if answer.strip())

        if answered_questions == 0:
            st.warning("Please answer at least one interview question.")
            return

        question_scores = []
        answer_length_scores = []
        relevance_scores = []
        structure_scores = []
        technical_scores = []
        confidence_scores = []
        detailed_feedback = []
        strengths = []
        improvements = []

        for i, answer in enumerate(answers):
            if not answer.strip():
                continue

            lower_answer = answer.lower()
            words = answer.split()
            word_count = len(words)

            # 1. Answer length
            if word_count >= 80:
                length_score = 100
            elif word_count >= 50:
                length_score = 85
            elif word_count >= 30:
                length_score = 70
            elif word_count >= 15:
                length_score = 50
            else:
                length_score = 30

            # 2. Keyword relevance
            keywords = keyword_groups[i]
            matched_keywords = [k for k in keywords if k in lower_answer]
            relevance_score = min(
                100,
                (len(matched_keywords) / max(1, min(5, len(keywords)))) * 100
            )

            # 3. Structure and communication
            has_example = any(
                phrase in lower_answer
                for phrase in ["for example", "such as", "when i", "in my project"]
            )
            has_result = any(
                word in lower_answer
                for word in [
                    "result", "achieved", "improved", "learned",
                    "success", "outcome", "impact"
                ]
            )
            has_reason = any(
                word in lower_answer
                for word in ["because", "therefore", "so that", "which helped"]
            )

            structure_score = 45
            if word_count >= 30:
                structure_score += 20
            if has_example:
                structure_score += 15
            if has_result:
                structure_score += 10
            if has_reason:
                structure_score += 10
            structure_score = min(100, structure_score)

            # 4. Technical/professional content
            technical_matches = sum(
                1 for word in technical_words if word in lower_answer
            )

            if technical_matches >= 4:
                technical_score = 100
            elif technical_matches >= 2:
                technical_score = 80
            elif technical_matches == 1:
                technical_score = 65
            else:
                technical_score = 50

            # 5. Confidence indicators
            positive_matches = sum(
                1 for word in positive_confidence_words
                if word in lower_answer
            )
            hesitation_matches = sum(
                1 for phrase in hesitation_words
                if phrase in lower_answer
            )

            confidence_score = 65
            confidence_score += min(25, positive_matches * 5)
            confidence_score -= min(30, hesitation_matches * 10)
            confidence_score = max(0, min(100, confidence_score))

            # Weighted final score
            final_score = round(
                length_score * 0.20
                + relevance_score * 0.30
                + structure_score * 0.20
                + technical_score * 0.15
                + confidence_score * 0.15
            )

            question_scores.append(final_score)
            answer_length_scores.append(length_score)
            relevance_scores.append(relevance_score)
            structure_scores.append(structure_score)
            technical_scores.append(technical_score)
            confidence_scores.append(confidence_score)

            # Question-specific feedback
            if word_count < 30:
                detailed_feedback.append(
                    f"Q{i+1}: Make your answer more detailed and include a specific example."
                )
                improvements.append(f"Q{i+1}: Increase answer detail.")
            elif relevance_score < 50:
                detailed_feedback.append(
                    f"Q{i+1}: Add more points directly related to the question."
                )
                improvements.append(f"Q{i+1}: Improve relevance.")
            elif structure_score < 70:
                detailed_feedback.append(
                    f"Q{i+1}: Improve the structure by explaining the situation, action, and result."
                )
                improvements.append(f"Q{i+1}: Improve answer structure.")
            else:
                detailed_feedback.append(
                    f"Q{i+1}: Good response. Keep the answer focused and support it with evidence."
                )

            if final_score >= 80:
                strengths.append(f"Q{i+1}: Strong and well-developed response.")
            if technical_score >= 80:
                strengths.append(f"Q{i+1}: Good technical/professional content.")
            if confidence_score >= 80:
                strengths.append(f"Q{i+1}: Confident language detected.")

        # Overall metrics
        interview_score = round(sum(question_scores) / len(question_scores))
        total_words = sum(len(answer.split()) for answer in answers if answer.strip())
        average_words = total_words / answered_questions

        avg_length = round(sum(answer_length_scores) / len(answer_length_scores))
        avg_relevance = round(sum(relevance_scores) / len(relevance_scores))
        avg_structure = round(sum(structure_scores) / len(structure_scores))
        avg_technical = round(sum(technical_scores) / len(technical_scores))
        avg_confidence = round(sum(confidence_scores) / len(confidence_scores))

        # Hiring recommendation
        if interview_score >= 85:
            recommendation = " Strong Candidate"
            recommendation_text = "The candidate demonstrates strong relevance, communication, technical content, and confidence."
        elif interview_score >= 70:
            recommendation = " Potential Candidate"
            recommendation_text = "The candidate shows good potential but should improve some areas before the final interview."
        elif interview_score >= 55:
            recommendation = "Needs Improvement"
            recommendation_text = "The candidate has a reasonable foundation but needs stronger, more detailed and relevant answers."
        else:
            recommendation = "Weak Interview Performance"
            recommendation_text = "The candidate should improve answer quality, relevance, structure, and confidence."

        # Results
        st.divider()
        st.subheader("Interview Result")

        st.metric("Overall Interview Score", f"{interview_score}%")
        st.progress(interview_score / 100)

        if interview_score >= 85:
            st.success(recommendation_text)
        elif interview_score >= 70:
            st.info(recommendation_text)
        elif interview_score >= 55:
            st.warning(recommendation_text)
        else:
            st.error(recommendation_text)

        st.subheader("Hiring Recommendation")
        st.markdown(f"### {recommendation}")
        st.write(recommendation_text)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Questions Answered", f"{answered_questions}/5")

        with col2:
            st.metric("Average Answer Length", f"{average_words:.0f} words")

        with col3:
            st.metric("Answer Relevance", f"{avg_relevance}%")

        st.subheader("Question-wise Score")

        for i, score in enumerate(question_scores):
            st.write(f"**Q{i+1}: {score}%**")
            st.progress(score / 100)

        st.subheader("📝 Detailed Feedback")

        for item in detailed_feedback:
            st.write("• " + item)

        st.subheader("Candidate Strengths")

        if strengths:
            for item in strengths[:8]:
                st.write("• " + item)
        else:
            st.write("• No strong areas detected yet. More detailed answers are recommended.")

        st.subheader("Areas to Improve")

        if improvements:
            for item in improvements:
                st.write("• " + item)
        else:
            st.success("No major improvement areas detected.")

        st.subheader("Interview Skill Breakdown")

        st.write(f"**Answer Length:** {avg_length}%")
        st.progress(avg_length / 100)

        st.write(f"**Relevance:** {avg_relevance}%")
        st.progress(avg_relevance / 100)

        st.write(f"**Structure & Communication:** {avg_structure}%")
        st.progress(avg_structure / 100)

        st.write(f"**Technical Content:** {avg_technical}%")
        st.progress(avg_technical / 100)

        st.write(f"**Confidence Indicators:** {avg_confidence}%")
        st.progress(avg_confidence / 100)


    # ==============================================
    # Free Interview Evaluation (No API / No Payment)
    # ==============================================
    st.divider()
    st.header("🆓 Free Interview Evaluation")
    st.success(
        "This project evaluates interview answers locally using answer relevance, "
        "detail, examples, structure, and job-related technical keywords. "
        "No OpenAI API key or paid API credits are required. Python packages and NLTK data may need to download during initial setup."
    )
    st.info(
        "Enter your answers above and click **Analyze Interview** to view the "
        "overall score, question-wise feedback, strengths, and areas to improve."
    )
# FREE INTERVIEW EVALUATION
    if st.button("Analyze Free Interview", key="free_interview_btn"):

        free_answers = [
            answer1, answer2, answer3, answer4, answer5
        ]

        scores = []
        feedback = []

        for i, ans in enumerate(free_answers):

            if not ans.strip():
                score = 0
                msg = "Please provide an answer."

            else:
                words = len(ans.split())
                score = 30

                if words >= 20:
                    score += 20
                if words >= 40:
                    score += 15
                if words >= 60:
                    score += 10

                keywords = [
                    ["name", "education", "degree", "skills", "experience"],
                    ["project", "developed", "created", "role", "technology"],
                    ["python", "java", "html", "css", "programming", "technical"],
                    ["problem", "solution", "solved", "challenge", "result"],
                    ["team", "skills", "company", "contribute", "work"]
                ]

                matched = sum(
                    1 for word in keywords[i]
                    if word in ans.lower()
                )

                score += min(matched * 5, 25)
                score = min(score, 100)

                if words < 20:
                    msg = "Give a more detailed answer."
                elif matched < 2:
                    msg = "Add more relevant points and examples."
                else:
                    msg = "Good answer. Keep improving with specific examples."

            scores.append(score)
            feedback.append(msg)

        overall = sum(scores) // 5

        st.divider()
        st.header("Free Interview Results")

        st.metric("Overall Interview Score", f"{overall}%")
        st.progress(overall / 100)

        if overall >= 80:
            st.success("Hiring Recommendation: Strong Candidate")
        elif overall >= 60:
            st.warning("Hiring Recommendation: Needs Improvement")
        else:
            st.error("Hiring Recommendation: Not Ready Yet")

        st.subheader("Question-wise Scores")

        for i, score in enumerate(scores):
            st.write(f"Q{i+1}: {score}%")
            st.progress(score / 100)
            st.write(feedback[i])

        st.subheader("Interview Summary")
        st.write(f"Questions Answered: {sum(1 for a in free_answers if a.strip())}/5")
        st.write(f"Average Score: {overall}%")

if __name__=="__main__":
    main()
