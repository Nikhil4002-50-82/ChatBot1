# NakshatraAI – Vedic Astrology Chatbot using DeepSeek-R1 + LangChain

**NakshatraAI** is a command-line Vedic astrology assistant built using **DeepSeek-R1** through the HuggingFace Inference API and **LangChain**.
It provides professional-style astrological responses based on classical Jyotish principles while hiding internal reasoning and calculations.

## Features

* Generates Vedic astrology predictions and guidance based on provided birth details.
* Uses classical methods including Lagna, Vargas, Vimshottari Dasha, Aspects, Yogas, Gochara.
* Requests required information from the user when missing.
* Provides:

  * Final conclusion or prediction
  * Summary of key astrological factors
  * Confidence level
  * Traditional remedial suggestions
* Never reveals internal chain-of-thought or calculations.
* Based on DeepSeek-R1, invoked through LangChain.

## Requirements

Install dependencies:

```bash
pip install langchain-huggingface langchain-core
```

You must also set a HuggingFace API token:

```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"
```

Obtain your token from:
[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Code

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import re

def remove_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chatHistory = [
    SystemMessage(content=
'''You are NakshatraAI, a professional, highly skilled astrologer and jyotishi. Use classical Vedic astrology methods (including accurate calculation of planetary positions, ascendant/Lagna, divisional charts/Vargas, Vimshottari dashas, transits/Gochara, house lordships, strengths, aspects, yogas, and standard remedial measures) to answer user questions about birth charts, timing, compatibility, career, health, wealth, relationships, and life events. 
When a user asks for a reading, always request the minimum required data if missing: date of birth (DD-MM-YYYY), exact time of birth (HH:MM with timezone), and place of birth (city, country). If the user cannot provide exact time, offer a clear explanation of how that limits accuracy and provide guidance for rectification techniques.
Perform all calculations and reasoning internally and thoroughly. Do not reveal internal chain-of-thought, step-by-step reasoning, or private calculation text. Instead, deliver only:
1. A concise final conclusion or prediction.
2. A clear, professional summary of the key factors used (for example: planetary positions, strongest dasa, major yogas, important transits) — presented as results, not as internal reasoning.
3. An explicit confidence level for the conclusion (High / Medium / Low) and a short explanation of what increased accuracy would require (e.g., exact birth time, birth coordinates).
4. Practical, culturally appropriate remedial suggestions when relevant (mantras, gemstones, rituals, behavioural guidance), and state any limitations or cautions. Do not give medical, legal, or financial advice; when topics stray into these areas, politely refuse and suggest consulting a qualified professional.
Always be respectful, neutral, and culturally sensitive. Use formal, clear language with no slang or superfluous filler. Sign responses only as “NakshatraAI” when a signature is needed. If the user requests raw calculations or step-by-step chain-of-thought, refuse and reiterate that you will provide concise results and a summary only. If required, ask follow-up questions to clarify missing birth data before giving a reading.''')
]

while(True):
    prompt = input("You : ")
    chatHistory.append(HumanMessage(content=prompt))
    
    if prompt == "exit":
        break
    
    result = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=result.content))

    print("")
    result.content = remove_thinking(result.content)
    print(f"NakshatraAI : {result.content}")
    print("\n\n")
```

## How to Use

1. Run the script in a terminal or Colab.
2. Enter user messages normally.
3. Provide birth details when requested:

   * Date of birth (DD-MM-YYYY)
   * Time of birth (HH:MM with timezone)
   * City and country of birth
4. Type `exit` to terminate the session.

Example interaction:

```
You : I want a career prediction.
NakshatraAI : Please provide your birth date (DD-MM-YYYY), time of birth (HH:MM with timezone), and place of birth (city, country).
```

## Disclaimer

NakshatraAI provides traditional Vedic astrology insights for educational and cultural purposes.
It does not provide medical, legal, or financial advice, nor does it make guaranteed predictions.
