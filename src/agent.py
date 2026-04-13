import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from src.vector_store import load_vector_store
from src.student_data import StudentData

load_dotenv()

class ARUGenie:
    def __init__(self, vector_store_path="data/vector_store/faiss_index", student_data_path="data/student_records/students.csv"):
        self.vector_store = load_vector_store(vector_store_path)
        self.student_data = StudentData(student_data_path)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.authenticated_student = None

    def search_regulations(self, query: str) -> str:
        """Search the ARU Academic Regulations for answers."""
        docs = self.vector_store.similarity_search(query, k=3)
        return "\n\n".join([d.page_content for d in docs])

    def authenticate_student(self, input_str: str) -> str:
        """Authenticate a student using SID and PIN. Input format: 'SID, PIN'"""
        try:
            sid, pin = [s.strip() for s in input_str.split(",")]
            student = self.student_data.verify_student(sid, pin)
            if student:
                self.authenticated_student = student
                return f"Authentication successful. Welcome, {student['Name']}!"
            return "Authentication failed. Incorrect SID or PIN."
        except ValueError:
            return "Invalid input format. Please provide SID and PIN separated by a comma (e.g., '12345678, 4821')."

    def get_personal_info(self, query: str = "") -> str:
        """Get personal student information (requires authentication)."""
        if not self.authenticated_student:
            return "This query requires access to personal records. Please provide your SID and PIN first."
        return f"Student: {self.authenticated_student['Name']} (SID: {self.authenticated_student['SID']})\nEmail: {self.authenticated_student['Email']}"

    def get_marks(self, query: str = "") -> str:
        """Get student marks/grades (requires authentication)."""
        if not self.authenticated_student:
            return "This query requires access to personal records. Please provide your SID and PIN first."
        return f"Your current grade for {self.authenticated_student['Module']} is: {self.authenticated_student['Grade']}%"

    def get_deadline(self, query: str = "") -> str:
        """Get next coursework deadline (requires authentication)."""
        if not self.authenticated_student:
            return "This query requires access to personal records. Please provide your SID and PIN first."
        return f"Your next deadline for {self.authenticated_student['Module']} is: {self.authenticated_student['Next_Deadline']}"

    def get_timetable(self, query: str = "") -> str:
        """Get student timetable (requires authentication)."""
        if not self.authenticated_student:
            return "This query requires access to personal records. Please provide your SID and PIN first."
        return f"Your timetable: {self.authenticated_student['Timetable']}"

    def get_mitigation_status(self, query: str = "") -> str:
        """Get student mitigation status (requires authentication)."""
        if not self.authenticated_student:
            return "This query requires access to personal records. Please provide your SID and PIN first."
        return f"Your mitigation status is: {self.authenticated_student['Mitigation_Status']}"

    def get_agent(self):
        tools = [
            Tool(
                name="search_regulations",
                func=self.search_regulations,
                description="Useful for answering general questions about ARU academic regulations, policies, and procedures."
            ),
            Tool(
                name="authenticate_student",
                func=self.authenticate_student,
                description="Use this to log in a student. Input should be 'SID, PIN'."
            ),
            Tool(
                name="get_personal_info",
                func=self.get_personal_info,
                description="Get basic info about the student. Requires login."
            ),
            Tool(
                name="get_marks",
                func=self.get_marks,
                description="Get student's current grades. Requires login."
            ),
            Tool(
                name="get_deadline",
                func=self.get_deadline,
                description="Get student's next coursework deadline. Requires login."
            ),
            Tool(
                name="get_timetable",
                func=self.get_timetable,
                description="Get student's class timetable. Requires login."
            ),
            Tool(
                name="get_mitigation_status",
                func=self.get_mitigation_status,
                description="Get student's mitigation or appeals status. Requires login."
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are ARU Genie, a helpful and highly intelligent assistant for Anglia Ruskin University students. "
                       "Your primary goal is to provide accurate, context-aware advice grounded in official university documents. \n\n"
                       "### CORE CAPABILITIES:\n"
                       "1. **Mitigation Advisor**: If a student describes a personal situation (e.g., illness, bereavement), you must: \n"
                       "   - Use 'search_regulations' to find the relevant mitigation criteria. \n"
                       "   - If authenticated, use 'get_deadline' and 'get_personal_info' to see their upcoming assessments. \n"
                       "   - Provide a reasoned recommendation (e.g., 'Based on the policy for [X], your [Y] deadline might qualify for mitigation because...'). \n"
                       "   - Always advise them to provide official evidence as per the policy. \n"
                       "2. **General Query**: Answer questions about ARU regulations using 'search_regulations'. \n"
                       "3. **Personal Records**: For data like marks or deadlines, you MUST ensure the student is authenticated. \n\n"
                       "### AUTHENTICATION PROTOCOL:\n"
                       "If a query requires personal data and the student is not authenticated, ask for their SID and PIN. "
                       "Once provided, use 'authenticate_student' to log them in."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    genie = ARUGenie()
    agent_executor = genie.get_agent()
    # Example interaction
    response = agent_executor.invoke({"input": "What is the policy for mitigation?", "chat_history": []})
    print(response["output"])
