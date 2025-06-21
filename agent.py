import os
import json
from dotenv import load_dotenv

# --- Core LangChain and Groq Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# --- 1. Load Environment Variables ---
# This loads the GROQ_API_KEY from your .env file
load_dotenv()

# Check if the API key is available
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# --- 2. Simulate Local Data File ---
# In a real-world app, you would load this from a file e.g., with open('smite_data.json') as f:
# For this single-file requirement, we'll store it as a multi-line string.
# This data is a simplified example of what you might scrape or find.
SMITE_DATA_JSON = """
{
  "Zeus": {
    "title": "God of the Sky",
    "class": "Mage",
    "type": "Ranged, Magical",
    "stats": {
      "win_rate_percent": 51.5,
      "pick_rate_percent": 8.2,
      "ban_rate_percent": 12.5
    },
    "abilities": {
      "Passive - Overcharge": "Zeus's basic attacks apply a charge. Enemies with 3 charges detonate, taking bonus damage.",
      "1 - Chain Lightning": "A bouncing lightning bolt that hits multiple enemies, applying a charge with each hit.",
      "2 - Aegis Assault": "Throws his shield, which creates a static field. Basic attacks hitting the shield create an explosion.",
      "3 - Detonate Charge": "Detonates all active charges on enemies, dealing significant damage based on the number of charges.",
      "4 - Lightning Storm": "Creates a large electrical storm that strikes all enemies within it for several seconds, applying charges."
    },
    "common_build": {
      "starter": ["Conduit Gem", "Lost Artifact"],
      "core_items": ["Book of Thoth", "Spear of Desolation", "Soul Reaver"],
      "situational_items": ["Divine Ruin (for anti-heal)", "Obsidian Shard (for tank shred)", "Mantle of Discord (for defense)"]
    },
    "strengths": ["High Area of Effect (AOE) damage", "Strong team fight presence", "Good at zoning enemies"],
    "weaknesses": ["Very low mobility (no escape)", "Vulnerable to ganks", "Reliant on landing abilities to apply charges"]
  },
  "Anubis": {
    "title": "God of the Dead",
    "class": "Mage",
    "type": "Ranged, Magical",
    "stats": {
      "win_rate_percent": 50.8,
      "pick_rate_percent": 9.5,
      "ban_rate_percent": 7.1
    },
    "abilities": {
      "Passive - Sorrow": "Anubis gains lifesteal and protections from items. Also, his abilities steal protections from the target.",
      "1 - Plague of Locusts": "A cone of locusts that deals damage over time. Anubis is stationary while casting.",
      "2 - Mummify": "Throws a bandage that stuns the first enemy god it hits. This is his key setup ability.",
      "3 - Grasping Hands": "Creates a circle on the ground that slows and damages enemies inside.",
      "4 - Death Gaze": "Fires a powerful beam of energy, dealing massive damage over time. Anubis is immune to knockback while casting."
    },
    "common_build": {
      "starter": ["Sands of Time", "Tiny Trinket"],
      "core_items": ["Bancroft's Talon", "Spear of the Magus", "Typhon's Fang"],
      "situational_items": ["Divine Ruin (for anti-heal)", "Charon's Coin (for mana and movement)", "Gem of Isolation (to enhance slows)"]
    },
    "strengths": ["Extremely high single-target damage", "Excellent lifesteal and sustain", "Powerful objective shred (Gold Fury, Fire Giant)"],
    "weaknesses": ["Completely immobile and self-roots on two abilities", "Very easy to gank and kill", "Relies heavily on landing his 'Mummify' stun"]
  },
  "Neith": {
    "title": "Weaver of Fate",
    "class": "Hunter",
    "type": "Ranged, Physical",
    "stats": {
      "win_rate_percent": 49.9,
      "pick_rate_percent": 11.0,
      "ban_rate_percent": 1.5
    },
    "abilities": {
      "Passive - Broken Weave": "When a god dies or Neith uses an ability, a 'Broken Weave' is left on the ground. Her other abilities have special effects on these weaves.",
      "1 - Spirit Arrow": "A long-range arrow that passes through enemies and roots the first god hit. If it passes through a weave, it detonates for AOE damage and roots all enemies in the area.",
      "2 - Unravel": "Damages and debuffs enemies in a cone. If it hits a weave, it heals Neith.",
      "3 - Back Flip": "Neith does a backflip, dealing damage and slowing enemies. Leaves a Broken Weave where she started.",
      "4 - World Weaver": "Fires a global arrow that seeks the targeted enemy god, dealing high damage and stunning them. Can be blocked by other gods."
    },
    "common_build": {
      "starter": ["Leather Cowl", "Spiked Gauntlet"],
      "core_items": ["Transcendence", "Asi", "Deathbringer", "The Executioner"],
      "situational_items": ["Dominance (for tank shred)", "Toxic Blade (for anti-heal)", "Magi's Cloak (against heavy crowd control)"]
    },
    "strengths": ["Global ultimate pressure", "Good lane sustain with her heal", "Safe laning phase with her Back Flip escape", "Built-in root (crowd control)"],
    "weaknesses": ["Falls off in late-game damage compared to hyper-carry hunters", "Ultimate can be blocked by enemy players", "Reliant on weaves for maximum utility"]
  }
}
"""

# --- 3. Data Retrieval Function ---
# This function acts as our "tool" to get data for a specific god.
def get_god_data(god_name: str) -> str:
    """
    Retrieves the statistics and information for a specific god from the JSON data.
    Performs a case-insensitive search.
    """
    print(f"--- Searching for data on '{god_name}'... ---")
    # Load the JSON data from our string
    data = json.loads(SMITE_DATA_JSON)
    
    # Find the god with a case-insensitive key search
    for key, value in data.items():
        if key.lower() == god_name.lower():
            print("--- Data found! Summarizing with Groq... ---")
            # Return the found data as a formatted string
            return json.dumps(value, indent=2)
            
    # If the loop finishes without finding the god
    print("--- God not found in the local data. ---")
    return "God not found. Please try one of the available gods: Zeus, Anubis, or Neith."

# --- 4. LangChain Agent/Chain Setup ---
# This is the "brain" of our agent. It tells the LLM how to behave and what to do.
def create_smite_agent_chain():
    """
    Creates the LangChain Expression Language (LCEL) chain for the agent.
    """
    # Define the LLM we want to use. `llama3-8b-8192` is fast and capable.
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
    
    # The prompt template is the instruction manual for the LLM.
    # It defines the context, the task, and the input variables.
    prompt_template = PromptTemplate(
        template="""
        You are an expert Smite 2 analyst and guide writer. Your goal is to help players quickly understand and learn a god.
        You will be given raw statistical data and ability information for a specific god.
        
        Your task is to synthesize this data into a clear, concise, and actionable player guide. The guide should be easy for a new or intermediate player to understand.
        
        Please structure your response in the following format:
        
        **God:** {god_name}
        **Class:** [God's Class]
        **Summary:** A brief 2-3 sentence overview of the god's playstyle, key strengths, and critical weaknesses based on the provided data.
        
        **Ability Playstyle:**
        - **Passive:** A quick tip on how to best utilize the passive.
        - **Ability 1:** A tip on when and how to use this ability.
        - **Ability 2:** A tip on when and how to use this ability.
        - **Ability 3:** A tip on when and how to use this ability.
        - **Ultimate:** A tip on the best situations to use the ultimate.
        
        **Recommended Build Guide:**
        - **Core Items:** Explain why the core items are essential for this god.
        - **Situational Items:** Briefly explain when a player should buy the situational items.
        
        **Gameplay Tips:**
        - **Early Game:** How to approach the first few minutes of the match.
        - **Mid & Late Game:** What the god's role is in team fights and objective control.
        - **Key Weakness to Cover:** A final piece of advice on how to mitigate the god's main weakness.
        
        Here is the raw data for the god:
        ---
        {god_data}
        ---
        """,
        input_variables=["god_name", "god_data"],
    )

    # This is the LCEL chain. It defines the flow of data.
    # 1. It takes a dictionary with "god_name" as input.
    # 2. `get_god_data` is called to fetch the data for that god.
    # 3. The `god_name` and the fetched `god_data` are passed to the prompt template.
    # 4. The formatted prompt is sent to the LLM (Groq).
    # 5. The LLM's response is parsed into a clean string.
    chain = (
        {
            "god_data": lambda x: get_god_data(x["god_name"]),
            "god_name": lambda x: x["god_name"].capitalize()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return chain

# --- 5. Main Execution and User Interaction Loop ---
if __name__ == "__main__":
    print("=================================================")
    print("      Welcome to the Smite 2 God Guide Agent!     ")
    print("=================================================")
    print("Type a god's name to get a guide, or 'exit' to quit.")
    print("-" * 50)

    # Create the agent chain
    agent_chain = create_smite_agent_chain()

    while True:
        # Get user input
        user_input = input("\nEnter a god's name: > ")

        # Check for exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using the Smite 2 God Guide Agent. Good luck in your games!")
            break
        
        # Check for empty input
        if not user_input.strip():
            print("Please enter a god's name.")
            continue
            
        # Invoke the chain with the user's input
        # The chain handles calling the get_god_data function internally
        response = agent_chain.invoke({"god_name": user_input})

        # Print the final, formatted response
        print("\n" + "="*20 + f" Guide for {user_input.capitalize()} " + "="*20)
        print(response)
        print("="* (42 + len(user_input)))