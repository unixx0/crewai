# Calculator Agent Using CrewAI


## Overview

This mini project gave me idea about custom tools in crewAI and also helped me to know how and when the Agent calls the tools.
This project helped me gain practical insight into CrewAI’s agent–tool interaction workflow.
---

## Libraries Used
- CrewAI - for building and managing agents
- crewai-tools - for tool management
- OS - for environment variable handling
- dotenv - for loading environment variables
- pydantic - for structured output
---


## Setup and Installation
### 1. Install Required Libraries
```bash
pip install crewai crewai-tools
pip install dotenv
pip install pydantic
```

### 2. Clone the repository

```bash
git clone https://github.com/unixx0/crewai
```

### 3. Navigate to the Calculator directory
```bash
cd Calculator
```

### 4. Configure Environment Variables
- Create .env inside Calculator directory
- Add your Hugging Face API key with the following name:
```env
hugging=your_huggingface_api_key
```

### 5. Run the Program
```bash
python calculator.py
```
## What you will learn from this miniproject
- How to define custom tools in crewai
- How agent chooses proper tool based on user input
- Basic agent orchestration using CrewAI



