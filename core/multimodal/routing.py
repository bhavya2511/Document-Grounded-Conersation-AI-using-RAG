from .graph import ExtractionState

def route_block(state: ExtractionState) -> str:
    """
    Conditional edge function to route to the correct processing node
    based on the current block's type.
    """
    block = state.get("current_block")
    if not block:
        return "merge_node" # No more blocks to process
        
    block_type = block.get("type", "").lower()
    
    if block_type in ["text"]:
        return "process_text_node"
    elif block_type in ["title"]:
        return "process_title_node"
    elif block_type in ["table"]:
        return "process_table_node"
    elif block_type in ["figure", "image"]:
        return "process_figure_node"
    else:
        # Fallback for unknown block types (treat as text if possible)
        return "process_text_node"
