import os
import csv
import argparse
from datetime import datetime

import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------------------
# 1. PROJECT CONFIGURATION
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_NAME = "gemma3:1b"
COLLEGE_NAME = "ABC College"

os.makedirs(DATASET_DIR, exist_ok=True)


# -----------------------------------------
# 2. INITIALIZE OLLAMA MODEL
# -----------------------------------------

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.3
)

output_parser = StrOutputParser()


# -----------------------------------------
# 3. CREATE FIVE PROMPT TEMPLATES
# -----------------------------------------

PROMPT_TEMPLATES = {

    "general": """
You are a helpful and friendly college assistant.

College Name: {college_name}

Answer the student's question clearly and accurately.
If information is unavailable, clearly say that you do not know.

Student Question: {user_input}

Answer:
""",

    "course": """
You are a college course information assistant.

College Name: {college_name}

Explain courses, departments, subjects and career opportunities
in simple language.

Student Question: {user_input}

Answer:
""",

    "examination": """
You are an examination assistance chatbot.

Provide helpful information about examinations,
exam preparation, schedules and academic procedures.

Do not invent official examination dates or rules.
Ask students to verify official announcements when necessary.

Student Question: {user_input}

Answer:
""",

    "library": """
You are a college library assistant.

Help students with library facilities, book borrowing,
reading rooms and library procedures.

Do not invent specific library timings or policies.

Student Question: {user_input}

Answer:
""",

    "attendance": """
You are a student attendance assistant.

Explain attendance calculations, attendance requirements
and attendance management in simple language.

Do not invent official college attendance policies.

Student Question: {user_input}

Answer:
"""
}


# -----------------------------------------
# 4. CREATE SAMPLE USER QUERIES
# -----------------------------------------

SAMPLE_QUERIES = [

    ("general", "What is the purpose of a college?"),
    ("general", "What facilities are available in a college?"),
    ("general", "How can students improve their communication skills?"),
    ("general", "Why are extracurricular activities important?"),

    ("course", "What is Computer Science?"),
    ("course", "What is Artificial Intelligence?"),
    ("course", "What are the benefits of learning Python?"),
    ("course", "What career opportunities are available in AI?"),

    ("examination", "How can I prepare for semester examinations?"),
    ("examination", "What is the importance of internal assessments?"),
    ("examination", "How can I manage examination stress?"),
    ("examination", "What is the best way to revise programming concepts?"),

    ("library", "What is the importance of a college library?"),
    ("library", "How does a library help students?"),
    ("library", "What is a digital library?"),
    ("library", "How should students maintain borrowed books?"),

    ("attendance", "What is attendance percentage?"),
    ("attendance", "How can I calculate my attendance percentage?"),
    ("attendance", "Why is regular attendance important?"),
    ("attendance", "How can students improve their attendance?")
]


# -----------------------------------------
# 5. SAVE PROMPT TEMPLATES AND QUERIES
# -----------------------------------------

def create_dataset_files():

    prompts_path = os.path.join(
        DATASET_DIR, "prompts.csv"
    )

    queries_path = os.path.join(
        DATASET_DIR, "user_queries.csv"
    )

    if not os.path.exists(prompts_path):

        with open(prompts_path, "w", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            writer.writerow([
                "template_name",
                "prompt_template"
            ])

            for name, template in PROMPT_TEMPLATES.items():

                writer.writerow([
                    name,
                    template.strip()
                ])

    if not os.path.exists(queries_path):

        with open(queries_path, "w", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            writer.writerow([
                "template_name",
                "user_query"
            ])

            writer.writerows(SAMPLE_QUERIES)

    print("Dataset files created successfully.")


# -----------------------------------------
# 6. BUILD LANGCHAIN PROMPT CHAIN
# -----------------------------------------

def build_chain(template_name="general"):

    template = PROMPT_TEMPLATES[template_name]

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | output_parser

    return chain


# -----------------------------------------
# 7. CREATE MAIN LLM CHAIN
# -----------------------------------------

main_chain = build_chain("general")


# -----------------------------------------
# 8. CREATE SUMMARIZATION CHAIN
# -----------------------------------------

summary_prompt = ChatPromptTemplate.from_template("""
You are a text summarization assistant.

Summarize the following text into a short,
clear and meaningful paragraph.

Text:
{text}

Summary:
""")

summarization_chain = summary_prompt | llm | output_parser


# -----------------------------------------
# 9. CREATE QUESTION ANSWERING CHAIN
# -----------------------------------------

qa_prompt = ChatPromptTemplate.from_template("""
You are a question answering assistant.

Answer the question using only the information
provided in the context.

If the answer is not available in the context,
say that the information is not provided.

Context:
{context}

Question:
{question}

Answer:
""")

question_answering_chain = qa_prompt | llm | output_parser


# -----------------------------------------
# 10. GENERATE RESPONSES FOR 20 QUERIES
# -----------------------------------------

def run_batch():

    responses = []

    print("\nGenerating responses for 20 queries...\n")

    for index, (template_name, query) in enumerate(
        SAMPLE_QUERIES, start=1
    ):

        chain = build_chain(template_name)

        response = chain.invoke({
            "user_input": query,
            "college_name": COLLEGE_NAME
        })

        responses.append({

            "query_id": index,
            "template_name": template_name,
            "user_query": query,
            "chatbot_response": response,
            "model": MODEL_NAME,
            "timestamp": datetime.now().isoformat()

        })

        print(f"\nQuery {index}: {query}")
        print(f"Response: {response}")
        print("-" * 60)

    response_path = os.path.join(
        DATASET_DIR, "chatbot_responses.csv"
    )

    pd.DataFrame(responses).to_csv(
        response_path,
        index=False
    )

    print("\n20 chatbot responses saved successfully.")


# -----------------------------------------
# 11. TEST SUMMARIZATION AND QA CHAINS
# -----------------------------------------

def run_additional_chains():

    results = []

    sample_text = """
    Artificial Intelligence is a branch of computer science.
    It enables machines to perform tasks that usually require
    human intelligence. AI is used in healthcare, education,
    transportation and many other industries.
    """

    summary = summarization_chain.invoke({
        "text": sample_text
    })

    results.append({

        "chain_name": "Summarization Chain",
        "input": sample_text.strip(),
        "output": summary

    })

    sample_context = """
    LangChain is a framework used to develop applications
    powered by Large Language Models. It provides prompt
    templates, chains and integrations with different models.
    """

    question = "What is LangChain?"

    answer = question_answering_chain.invoke({

        "context": sample_context,
        "question": question

    })

    results.append({

        "chain_name": "Question Answering Chain",
        "input": question,
        "output": answer

    })

    results_path = os.path.join(
        DATASET_DIR, "chain_results.csv"
    )

    pd.DataFrame(results).to_csv(
        results_path,
        index=False
    )

    print("\nAdditional chain results saved successfully.")

    print("\nSUMMARY:")
    print(summary)

    print("\nQUESTION ANSWERING:")
    print(answer)


# -----------------------------------------
# 12. INTERACTIVE CHATBOT
# -----------------------------------------

def interactive_chatbot():

    print("\n===================================")
    print("       LANGCHAIN COLLEGE BOT")
    print("===================================")

    print("Available templates:")

    for name in PROMPT_TEMPLATES:
        print("-", name)

    print("\nType 'exit' to stop the chatbot.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            print("Thank you for using College Assistant!")
            break

        if not query:
            print("Please enter a valid question.")
            continue

        print("\nChoose a template:")
        print("1. General")
        print("2. Course")
        print("3. Examination")
        print("4. Library")
        print("5. Attendance")

        choice = input("Enter choice (1-5): ").strip()

        template_names = [
            "general",
            "course",
            "examination",
            "library",
            "attendance"
        ]

        if choice not in ["1", "2", "3", "4", "5"]:
            print("Invalid choice. Using general template.")
            choice = "1"

        template_name = template_names[int(choice) - 1]

        chain = build_chain(template_name)

        response = chain.invoke({

            "user_input": query,
            "college_name": COLLEGE_NAME

        })

        print("\nChatbot Response:")
        print(response)


# -----------------------------------------
# 13. MAIN PROGRAM
# -----------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate responses for all 20 queries"
    )

    args = parser.parse_args()

    create_dataset_files()

    if args.batch:

        run_batch()
        run_additional_chains()

    else:

        interactive_chatbot()