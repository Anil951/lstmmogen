import os
import sys
import re
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
import importlib
# Import current configurations
from ActionConditionedLSTM.config import ACTION_CLASSES, IOFILES_DIR
import segment_image
import ActionConditionedLSTM.generate_action_animations as generate_action_animations
import animate_image
import composer

# Load environment variables
load_dotenv(".env")




@tool
def get_all_valid_actions() -> List[str]:
    """Returns the list of valid actions that the deep learning model has the capability to generate."""
    print(f"\n[Tool Executing] get_all_valid_actions -> {ACTION_CLASSES}")
    return ACTION_CLASSES


def update_config_variable(name: str, value: any) -> str:
    """
    Dynamically updates and persists a configuration variable in ActionConditionedLSTM/config.py.
    This function must be called to set target configurations prior to running the pipelines.
    
    Parameters:
        - name: The configuration variable name to modify (e.g., 'TARGET_IMAGE_NAME', 'ACTION_CHOICE', 'OVERLAP_SKELETON', 'COMPOSER_GIF_SEQUENCE', 'COMPOSER_OUTPUT_PATH', 'ANIMATED_CHARACTER_PATH').
        - value: The new value for the variable (strings, booleans, or lists/sequences).
    """
    config_path = os.path.join(os.path.dirname(__file__), "ActionConditionedLSTM", "config.py")
    if not os.path.exists(config_path):
        return f"Error: Config file not found at {config_path}"
        
    with open(config_path, "r") as f:
        content = f.read()
        
    # We want to search for "^name = ..." and replace it with its Python representation
    pattern = re.compile(rf"^({name}\s*=\s*)(.*)$", re.MULTILINE)
    
    if isinstance(value, str) and (os.path.isabs(value) or "\\" in value or "/" in value):
        filename = os.path.basename(value)
        repr_value = f"os.path.join(IOFILES_DIR, {repr(filename)})"
    else:
        repr_value = repr(value)
    
    if not pattern.search(content):
        # Variable not found in the file, append it to the end
        new_content = content.rstrip() + f"\n\n# Dynamically added by Agent\n{name} = {repr_value}\n"
    else:
        new_content = pattern.sub(lambda m: f"{m.group(1)}{repr_value}", content)
        
    with open(config_path, "w") as f:
        f.write(new_content)
        
    print(f"\n[Config updater Executing] update_config_variable(name='{name}', value={repr_value}) -> Persisted to disk.")
    return f"Successfully updated config variable '{name}' to {repr_value}"

def reload_config():
    """Helper to reload config modules across all namespaces in sys.modules."""
    for m_name in list(sys.modules.keys()):
        if m_name == "config" or m_name == "ActionConditionedLSTM.config":
            try:
                importlib.reload(sys.modules[m_name])
            except Exception:
                pass


@tool
def segment_character(character_img_name: str) -> str:
    """
    Runs the segmentation pipeline for a given character image file.
    It registers the target image, performs Watershed segmentation, and saves the cached parts.
    
    Parameters:
        - character_img_name: The name of the character image file inside iofiles (e.g., 'sketch.jpg')
    """
    print(f"\n[Tool Executing] segment_character(image='{character_img_name}')")
    
    # 1. Update config variable first
    update_config_variable("TARGET_IMAGE_NAME", character_img_name)
    reload_config()
    
    # 2. Call segment_image.main() directly in-process
    try:
        importlib.reload(segment_image)
        segment_image.main()
    except Exception as e:
        print(f"   -> Segmentation Failed: {e}")
        return f"Error running segment_image.main(): {e}"
        
    print("   -> Character segmentation complete and cached.")
    return "Character segmented and cached successfully!"


@tool
def generate_motion_for_action(action_name: str, output_motion_name: str = "") -> str:
    """
    Generates a motion sequence file based on an action class and saves it to a unique .npy file.
    
    Parameters:
        - action_name: The action class name (e.g. 'walk', 'run', 'jump_vertical', 'throw_both_hands', 'boxing_right_left')
        - output_motion_name: (Optional) The desired filename for the output .npy motion file (e.g. 'walk_motion.npy'). If omitted, defaults to '{action_name}_motion.npy'.
    """
    if not output_motion_name:
        output_motion_name = f"{action_name}_motion.npy"
    if not output_motion_name.endswith(".npy"):
        output_motion_name += ".npy"
        
    print(f"\n[Tool Executing] generate_motion_for_action(action='{action_name}', output_motion='{output_motion_name}')")
    
    # 1. Update config variables
    update_config_variable("ACTION_CHOICE", action_name)
    motion_path = os.path.join(IOFILES_DIR, output_motion_name)
    update_config_variable("GENERATED_MOTION_PATH", motion_path)
    reload_config()
    
    # 2. Call generate_action_animations.main() directly in-process
    try:
        importlib.reload(generate_action_animations)
        generate_action_animations.main()
    except Exception as e:
        print(f"   -> Motion Generation Failed: {e}")
        return f"Error running generate_action_animations.main(): {e}"
        
    print(f"   -> Motion generation complete. Saved to {output_motion_name}")
    return f"Successfully generated motion sequence for action '{action_name}' and saved to '{output_motion_name}'!"


@tool
def animate_character(output_gif_name: str, character_img_name: str = "", motion_npy_name: str = "") -> str:
    """
    Warps the specified motion sequence (.npy) onto the specified character's segmented sketch and renders the animation GIF.
    
    Parameters:
        - output_gif_name: The desired filename for the output GIF (e.g. 'character_walk.gif')
        - character_img_name: (Required for multi-character stories) The character image filename to use for this animation (e.g., 'sketch.jpg').
        - motion_npy_name: (Required) The motion .npy filename to use for this animation (e.g., 'walk_motion.npy' or 'boxing_motion.npy').
    """
    print(f"\n[Tool Executing] animate_character(output='{output_gif_name}', character='{character_img_name}', motion='{motion_npy_name}')")
    
    # 1. Update target character image if specified
    if character_img_name:
        update_config_variable("TARGET_IMAGE_NAME", character_img_name)
        
    # 2. Update motion path if specified
    if motion_npy_name:
        motion_path = os.path.join(IOFILES_DIR, motion_npy_name)
        update_config_variable("GENERATED_MOTION_PATH", motion_path)
        
    # 3. Update config output path
    output_path = os.path.join(IOFILES_DIR, output_gif_name)
    update_config_variable("ANIMATED_CHARACTER_PATH", output_path)
    reload_config()
    
    # 4. Call Animate_Image.main() directly in-process
    try:
        importlib.reload(animate_image)
        animate_image.main()
    except Exception as e:
        print(f"   -> Animation Rendering Failed: {e}")
        return f"Error running Animate_Image.main(): {e}"
        
    print(f"   -> Animation rendered successfully to {output_path}")
    return f"Successfully generated animation GIF: {output_gif_name}"


@tool
def stitch_animations(gif_sequence: List[str], output_gif_name: str) -> str:
    """
    Stitches a list of GIF files sequentially in chronological order into a final combined GIF.
    
    Parameters:
        - gif_sequence: A list of GIF filenames/paths in the exact sequence they should occur (e.g. ['walk.gif', 'run.gif'])
        - output_gif_name: The desired filename for the final combined GIF (e.g. 'story_combined.gif')
    """
    print(f"\n[Tool Executing] stitch_animations(sequence={gif_sequence}, output='{output_gif_name}')")
    
    # 1. Update config variables
    output_path = os.path.join(IOFILES_DIR, output_gif_name)
    update_config_variable("COMPOSER_GIF_SEQUENCE", gif_sequence)
    update_config_variable("COMPOSER_OUTPUT_PATH", output_path)
    reload_config()
    
    # 2. Call composer.compose_gifs directly in-process
    try:
        importlib.reload(composer)
        composer.compose_gifs(gif_sequence, output_path)
    except Exception as e:
        print(f"   -> Stitching Failed: {e}")
        return f"Error running composer.compose_gifs(): {e}"
        
    print(f"   -> Sequence stitched successfully to {output_path}")
    return f"Successfully stitched sequence into: {output_gif_name}"


# List of tools bound to the agent
tools = [
    get_all_valid_actions,
    segment_character,
    generate_motion_for_action,
    animate_character,
    stitch_animations
]

SYSTEM_PROMPT = """You are an autonomous AI Animation Director.
Your objective is to parse any given storyline prompt and autonomously orchestrate its conversion into a complete animated sequence using your tools.

DIRECTOR DUTIES & WORKFLOW:

1. ACTION DISCOVERY:
   - Always call `get_all_valid_actions()` first to fetch the exact list of valid action classes supported by the animation model.
   - Match story events strictly to available valid action classes.

2. CHARACTER IDENTIFICATION & SEGMENTATION:
   - Identify all unique character image filenames/sketches mentioned in the user's storyline prompt.
   - Call `segment_character(character_img_name)` for EVERY unique character image found in the story.

3. MOTION GENERATION:
   - For each action event in the story, call `generate_motion_for_action(action_name, output_motion_name)` specifying a unique output `.npy` filename (e.g. 'walk_motion.npy', 'boxing_motion.npy', 'run_motion.npy', 'jump_motion.npy').

4. ANIMATION RENDERING:
   - For each event in the story sequence in chronological order:
     Call `animate_character(output_gif_name, character_img_name, motion_npy_name)` specifying:
     - `output_gif_name`: a unique descriptive GIF filename (e.g. 'A_walk.gif', 'B_boxing.gif').
     - `character_img_name`: the exact image filename of the character performing this action.
     - `motion_npy_name`: the exact motion `.npy` filename generated for this event in Step 3 (e.g. 'walk_motion.npy').

5. STITCHING & COMPOSITION:
   - Collect all individual rendered GIF filenames in their exact chronological order.
   - Call `stitch_animations(gif_sequence, output_gif_name)` to combine them into the final output GIF.
"""

# Initialize Groq LLM
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set in environment (.env)")

model_name = os.getenv("GROQ_MODEL")
llm = ChatGroq(
    groq_api_key=api_key,
    model_name=model_name,
    temperature=0.0, # Greedy generation for predictable tool calling
    max_tokens=4096,
)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MessagesState):
    """The main LLM node that makes autonomous decisions."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = llm_with_tools.invoke(messages)
    
    print("\n" + "="*40)
    print("AGENT DECISION:")
    if response.content:
        print(f"Text:\n{response.content}")
    if response.tool_calls:
        print("Tools Called:")
        for tc in response.tool_calls:
            print(f" - {tc['name']} with args: {tc['args']}")
        
    return {"messages": [response]}


# Build state graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# Save visual workflow graph
try:
    png_data = graph.get_graph().draw_png()
    with open(os.path.join(IOFILES_DIR, "langgraph_workflow.png"), "wb") as f:
        f.write(png_data)
except Exception as e:
    print(f"Could not generate LangGraph diagram: {e}")


if __name__ == "__main__":
    # Storyline containing 2-3 actions from available classes to test the agent
    storyline = """The character A (f99a3cd6-93c4-43aa-a155-e76b03578dd8_resized.jpg) starts by walking down the street. Then another character B (006027f5-13d6-43e8-a3b8-223f377428b7_resized.jpg) is practising boxing. A sees B and starts running towards B. B being exicted starts to jump as he is seeing his old friend A.
    """
    
    print(f"Starting Graph Execution for Story: {storyline.strip()}")
    for event in graph.stream({"messages": [HumanMessage(content=storyline)]}, stream_mode="values"):
        pass