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

# --- 2. Load Local Data Files ---
# Load legacy data from string, and new data from god_item_description.json
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

# --- 3. Load god_item_description.json Data ---
GOD_ITEM_DESCRIPTION = []
GOD_ITEM_DESCRIPTION_BY_GOD = {}
GOD_ITEM_DESCRIPTION_PATH = os.path.join(os.path.dirname(__file__), 'god_item_description.json')
try:
    with open(GOD_ITEM_DESCRIPTION_PATH, encoding='utf-8') as f:
        GOD_ITEM_DESCRIPTION = json.load(f)
        # Build a lookup by god name (uppercase)
        for entry in GOD_ITEM_DESCRIPTION:
            god_name = entry.get('god', '').upper()
            if god_name:
                GOD_ITEM_DESCRIPTION_BY_GOD[god_name] = entry
except Exception as e:
    print(f"Warning: Could not load god_item_description.json: {e}")


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


def get_god_item_description(god_name: str):
    """
    Returns the item description entry for a god from god_item_description.json (case-insensitive).
    """
    if not god_name:
        return None
    return GOD_ITEM_DESCRIPTION_BY_GOD.get(god_name.upper())


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

    # --- Helper: Extract god name from user input ---
    import re
    def find_god_in_text(text):
        # Load god names from both local data sources
        god_db = json.loads(SMITE_DATA_JSON)
        god_names = set(god_db.keys())
        god_names.update(GOD_ITEM_DESCRIPTION_BY_GOD.keys())
        # Search for any god name in the user input (case-insensitive)
        for god in god_names:
            pattern = r'\b' + re.escape(god) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                return god
        return None

    # --- Helper: Extract intent and details from user input ---
    def extract_intent(text, god_data=None):
        text = text.lower()
        # Abilities: check for ability names
        if god_data and "abilities" in god_data:
            for ability_name in god_data["abilities"]:
                if ability_name.lower().split(" - ")[0] in text or ability_name.lower() in text:
                    return ("specific_ability", ability_name)
        # Item sections
        if "starter" in text:
            return ("starter_items", None)
        if "core" in text:
            return ("core_items", None)
        if "situational" in text or "situational items" in text:
            return ("situational_items", None)
        # Stats
        if "win rate" in text:
            return ("stat", "win_rate_percent")
        if "pick rate" in text:
            return ("stat", "pick_rate_percent")
        if "ban rate" in text:
            return ("stat", "ban_rate_percent")
        # Class, type, title, lore
        if "class" in text:
            return ("class", None)
        if "type" in text:
            return ("type", None)
        if "title" in text:
            return ("title", None)
        if any(word in text for word in ["lore", "story", "background"]):
            return ("lore", None)
        # Weakness
        if any(word in text for word in ["weakness", "counter", "vulnerab"]):
            return ("weakness", None)
        # Abilities (all)
        if any(word in text for word in ["abilit", "skill", "kit"]):
            return ("abilities", None)
        # Build
        if any(word in text for word in ["build", "item", "items", "buy"]):
            return ("build", None)
        # Role/class
        if any(word in text for word in ["role"]):
            return ("role", None)
        # Summary
        if any(word in text for word in ["summary", "overview", "about", "guide"]):
            return ("summary", None)
        return ("full", None)

    conversation_history = []
    last_god_name = None
    print("\nYou can now chat with the agent! Ask about gods, items, or Smite tips. Type 'exit' or 'quit' to leave.")
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Agent: Thank you for using the Smite 2 God Guide Agent. Good luck in your games!")
            break
        if not user_input.strip():
            print("Agent: Please enter a question or a god's name.")
            continue

        # Store user message
        conversation_history.append({"role": "user", "content": user_input})

        god_name = find_god_in_text(user_input)
        if god_name:
            last_god_name = god_name
        elif last_god_name:
            god_name = last_god_name
        else:
            print("Agent: I couldn't find a god name in your question. Please mention the god you want to know about!")
            continue

        god_db = json.loads(SMITE_DATA_JSON)
        god_data = god_db.get(god_name)
        god_item_desc = get_god_item_description(god_name)
        if not god_data and not god_item_desc:
            print(f"Agent: Sorry, I couldn't find any data for {god_name}.")
            continue

        # Fine-grained intent and detail extraction
        # If god_data is missing but god_item_desc is present, use a minimal dict for intent extraction
        intent, detail = extract_intent(user_input, god_data or {"abilities": {}, "common_build": {}})
        response = None
        if intent == "specific_ability" and detail:
            ability = god_data.get("abilities", {}).get(detail)
            if ability:
                response = f"{god_name} - {detail}: {ability}"
            else:
                response = f"Sorry, I couldn't find info for {detail} on {god_name}."
        elif intent == "starter_items":
            starter = god_data.get("common_build", {}).get("starter")
            if starter:
                response = f"{god_name} starter items: {', '.join(starter)}"
            else:
                response = f"Sorry, I couldn't find starter items for {god_name}."
        elif intent == "core_items":
            core = god_data.get("common_build", {}).get("core_items")
            if core:
                response = f"{god_name} core items: {', '.join(core)}"
            else:
                response = f"Sorry, I couldn't find core items for {god_name}."
        elif intent == "situational_items":
            situational = god_data.get("common_build", {}).get("situational_items")
            if situational:
                response = f"{god_name} situational items: {', '.join(situational)}"
            else:
                response = f"Sorry, I couldn't find situational items for {god_name}."
        elif intent == "stat" and detail:
            stat_val = god_data.get("stats", {}).get(detail)
            if stat_val is not None:
                response = f"{god_name} {detail.replace('_', ' ')}: {stat_val}"
            else:
                response = f"Sorry, I couldn't find that stat for {god_name}."
        elif intent == "class":
            response = f"{god_name} class: {god_data.get('class', 'N/A')}"
        elif intent == "type":
            response = f"{god_name} type: {god_data.get('type', 'N/A')}"
        elif intent == "title":
            response = f"{god_name} title: {god_data.get('title', 'N/A')}"
        elif intent == "lore":
            lore = god_data.get("lore", "")
            if lore:
                response = f"Lore for {god_name}: {lore}"
            else:
                response = f"Sorry, I couldn't find lore for {god_name}."
        elif intent == "weakness":
            tips = "\n".join([
                god_data.get("summary", ""),
                god_data.get("abilities", {}).get("Key Weakness to Cover", ""),
                god_data.get("gameplay_tips", {}).get("Key Weakness to Cover", "")
            ])
            if not tips.strip():
                for v in god_data.get("abilities", {}).values():
                    if "weakness" in v.lower():
                        tips += v + "\n"
            if "weaknesses" in god_data and god_data["weaknesses"]:
                tips += "\n" + ", ".join(god_data["weaknesses"])
            if tips.strip():
                response = f"{god_name}'s main weakness: {tips.strip()}"
            else:
                response = f"Sorry, I couldn't find a specific weakness for {god_name}."
        elif intent == "abilities":
            abilities = god_data.get("abilities", {})
            if abilities:
                response = f"{god_name}'s abilities:\n" + "\n".join([f"{k}: {v}" for k,v in abilities.items()])
            else:
                response = f"Sorry, I couldn't find abilities for {god_name}."
        elif intent == "build":
            # Try to get richer build info from god_item_description.json
            god_item_desc = get_god_item_description(god_name)
            if god_item_desc:
                items = god_item_desc.get('items', [])
                starter = god_item_desc.get('starter_detailed', {}).get('name', god_item_desc.get('starter'))
                relic = god_item_desc.get('relic_detailed', {}).get('name', god_item_desc.get('relic'))
                item_names = ', '.join([item['name'] for item in items]) if items else 'N/A'
                response = f"Build for {god_name} ({god_item_desc.get('role','')}):\nStarter: {starter}\nRelic: {relic}\nItems: {item_names}"
            else:
                build = god_data.get("recommended_build", "")
                common_build = god_data.get("common_build", {})
                if build:
                    response = f"Recommended build for {god_name}: {build}"
                elif common_build:
                    response = f"Build for {god_name}:\n"
                    if 'starter' in common_build:
                        response += f"  Starter: {', '.join(common_build['starter'])}\n"
                    if 'core_items' in common_build:
                        response += f"  Core Items: {', '.join(common_build['core_items'])}\n"
                    if 'situational_items' in common_build:
                        response += f"  Situational Items: {', '.join(common_build['situational_items'])}"
                else:
                    response = f"Sorry, I couldn't find a recommended build for {god_name}."
        elif intent == "role":
            response = f"{god_name} is a {god_data.get('class', 'N/A')} ({god_data.get('type', 'N/A')})"
        elif intent == "summary":
            response = god_data.get("summary", f"Here's a quick overview of {god_name}.")
        else:
            # Full guide fallback
            if god_data:
                response = f"**God:** {god_name}\n**Class:** {god_data.get('class', 'N/A')}\n**Type:** {god_data.get('type', 'N/A')}\n"
                if "summary" in god_data:
                    response += f"\n{god_data['summary']}\n"
                if "abilities" in god_data:
                    response += "\nAbilities:\n"
                    for k,v in god_data["abilities"].items():
                        response += f"- {k}: {v}\n"
                if "recommended_build" in god_data:
                    response += f"\nRecommended Build: {god_data['recommended_build']}\n"
                if "gameplay_tips" in god_data:
                    response += "\nGameplay Tips:\n"
                    for k,v in god_data["gameplay_tips"].items():
                        response += f"- {k}: {v}\n"
            elif god_item_desc:
                response = f"**God:** {god_name}\n**Role:** {god_item_desc.get('role', 'N/A')}\n"
                starter = god_item_desc.get('starter_detailed', {}).get('name', god_item_desc.get('starter'))
                relic = god_item_desc.get('relic_detailed', {}).get('name', god_item_desc.get('relic'))
                items = god_item_desc.get('items', [])
                item_names = ', '.join([item['name'] for item in items]) if items else 'N/A'
                response += f"Starter: {starter}\nRelic: {relic}\nItems: {item_names}\n"
            else:
                response = f"Sorry, I couldn't find any data for {god_name}."
        conversation_history.append({"role": "agent", "content": response})

        print(f"Agent: {response}")
        print("="* (42 + len(user_input)))