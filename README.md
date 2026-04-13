# ARU Student Support Agent (ARU Genie)

## Project Overview
This repository contains an Agentic Retrieval-Augmented Generation (RAG) system designed to support ARU students. It can answer questions about academic regulations and provide personalized student information (marks, deadlines, etc.) through a basic authentication mechanism.

## Features
- **Agentic RAG**: Uses LangChain and OpenAI to intelligently route queries between general regulations and personalized data.
- **Knowledge Base**: Built from official ARU Academic Regulations and Student Rules PDFs.
- **Authentication**: Simple SID and PIN-based verification for personal records.
- **Personalization**: Access to dummy student records (marks, deadlines, timetables, mitigation status).

## Project Structure
- `data/raw_docs/`: Original ARU regulation PDFs.
- `data/student_records/`: Dummy student data (CSV).
- `data/vector_store/`: FAISS index for document retrieval.
- `src/`: Core implementation (Agent, Vector Store, Student Data).
- `notebooks/`: Interactive demonstration notebook.
- `main.py`: CLI entry point for the agent.

## Getting Started

### 1. Set up the Environment
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your OpenAI API key:
```text
OPENAI_API_KEY=your_api_key_here
```

### 3. Build the Knowledge Base
Ensure you have the PDFs in `data/raw_docs/` and run:
```bash
python src/vector_store.py
```

### 4. Run the Agent
Use the CLI to interact with ARU Genie:
```bash
python main.py
```
Or open the `notebooks/Demo.ipynb` for an interactive walkthrough.

## Example Interactions
- **Query**: "What is the policy for mitigation?" -> *Uses search_regulations tool.*
- **Query**: "When is my next deadline?" -> *Asks for SID/PIN if not authenticated.*
- **Auth**: "SID 12345678, PIN 4821" -> *Authenticates and provides the record.*

## Creative Features
- **Mitigation Eligibility Advisor**: The agent doesn't just retrieve policy; it **reasons** through the student's personal circumstances. By combining the **Mitigation Policy** (RAG) with the student's **Next Deadline** (Records), it provides personalized advice on whether they qualify and how to apply.
