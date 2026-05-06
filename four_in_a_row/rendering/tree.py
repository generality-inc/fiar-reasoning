from dataclasses import dataclass
from typing import Any, List, Tuple, Union, Optional
import graphviz
from IPython.display import Image, display

Nested = Union[List[Any], Tuple[Any, ...], Any]

@dataclass(eq=False)
class Node:
    label: str
    children: List["Node"]
    status: str = "normal"
    result: Optional[str] = None
    board_state: Optional[str] = None

def _parse_head(head: Any) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Parse head and return (label, status, result, board_state)"""
    board_state = None
    
    # Check if head is a dict with board_state
    if isinstance(head, dict):
        board_state = head.get("board_state")
        # Use label from dict or convert dict to string
        lbl = head.get("label", str(head))
        status = head.get("status", "normal")
        result = head.get("result")
        return lbl, status, result, board_state
    
    if isinstance(head, tuple):
        lbl = str(head[0])
        if len(head) >= 2 and head[1] == "illegal":
            return lbl, "illegal", None, None
        if len(head) >= 2 and head[1] == "final":
            res = head[2] if len(head) >= 3 else None
            return lbl, "final", (None if res is None else str(res)), None
        return lbl, "normal", None, None
    return str(head), "normal", None, None

def to_node(x: Nested) -> Node:
    if isinstance(x, (list, tuple)) and x:
        first = x[0]
        # Check if first element is START (either as string or dict with label "START")
        is_start = (first == "START" or 
                   (isinstance(first, dict) and first.get("label") == "START"))
        if not is_start:
            x = list(x)
            x.insert(0, "START")
    return _to_node(x)

def _to_node(x: Nested) -> Node:
    if isinstance(x, (list, tuple)):
        if not x:
            return Node("∅", [])
        head, *tail = x
        if isinstance(head, int) and tail and isinstance(tail[0], int):
            lbl = f"{int(head)},{int(tail[0])}"
            kids = [_to_node(c) for c in tail[1:]]
            return Node(lbl, kids)
        lbl, st, res, board_state = _parse_head(head)
        kids = [_to_node(c) for c in tail]
        return Node(lbl, kids, status=st, result=res, board_state=board_state)
    return Node(str(x), [])

def _extract_fen(board_state: str) -> str:
    """Extract FEN from board_state string. Returns 'FEN: ...' format"""
    if not board_state:
        return "START"
    try:
        start_marker = "FEN: "
        end_marker = "\n\n"
        start_idx = board_state.find(start_marker)
        if start_idx == -1:
            return "START"
        # Keep the "FEN: " prefix
        fen_start = start_idx
        start_idx += len(start_marker)
        end_idx = board_state.find(end_marker, start_idx)
        if end_idx == -1:
            # Try just newline
            end_idx = board_state.find("\n", start_idx)
            if end_idx == -1:
                return "START"
        return board_state[fen_start:end_idx].strip()
    except:
        return "START"

def _node_style(n: Node, node_size: float = 0.3, font_size: int = 11, depth: int = 0) -> dict:
    # Alternate shape based on depth: circle for odd depths, box for even depths (after START)
    shape = "circle" if depth % 2 == 1 else "box"
    
    base = {
        "fontname": "Arial",
        "fontsize": str(font_size),
        "fontcolor": "black",
        "style": "",
        "shape": shape,
        "width": str(node_size),
        "height": str(node_size),
        "fixedsize": "true",
        "penwidth": "2",
        "color": "#000000",
    }

    if n.label == "START" or n.board_state:
        # START node or node with board_state (FEN) - make it wider to fit FEN string
        base.update({
            "fillcolor": "none",
            "fontcolor": "black",
            "shape": "box",
            "style": "",
            "width": str(node_size * 3.5),
            "height": str(node_size * 1),
            "fontsize": str(font_size + 1),
            "fontname": "Arial-Bold",
            "penwidth": "2",
            "color": "#000000",
            "fixedsize": "false"
        })
    elif n.status == "final":
        r = (n.result or "").lower()
        if r == "win":
            base.update({"fillcolor": "#00FF41", "fontcolor": "black", "style": "filled", "penwidth": "0"})
        elif r == "loss":
            base.update({"fillcolor": "#FF073A", "fontcolor": "white", "style": "filled", "penwidth": "0"})
        elif r == "tie":
            base.update({"fillcolor": "#FFD700", "fontcolor": "black", "style": "filled", "penwidth": "0"})
        else:
            base.update({"fillcolor": "none", "fontcolor": "black"})
    elif n.status == "illegal":
        base.update({"fillcolor": "#666666", "fontcolor": "white", "style": "filled,dashed", "penwidth": "1.5"})
    else:
        base.update({"fillcolor": "none", "fontcolor": "black"})
    
    return base

def draw_tree(root: Node, title=None, filename=None, format="png", dpi="150",
              node_size=0.3, font_size=11, splines_style="polyline", force_center=False, show_legend=False, fig_size="16,10"):
    
    dot = graphviz.Digraph(
        comment=title or "Game Tree",
        format=format,
        engine="dot",
        graph_attr={
            "bgcolor": "#FFFFFF",
            "color": "#FFFFFF",
            "penwidth": "0",
            "rankdir": "TB",
            "ranksep": "0.35",
            "nodesep": "0.5",
            "splines": splines_style,
            "dpi": dpi,
            "size": f"{fig_size}!",
            "ratio": "fill",
            "margin": "0.3",
            "pad": "0.3",
        },
        node_attr={
            "fontname": "Arial",
            "fontsize": "11",
        },
        edge_attr={
            "color": "#000000",
            "penwidth": "2",
            "dir": "none",
        }
    )
    dot.attr(penwidth="0", color="#FFFFFF")

    # Add title at top
    if title:
        dot.attr(label=title + "\\n\\n", labelloc="t", labeljust="c",
                fontsize=str(int(font_size * 2)), fontcolor="#000000", fontname="Arial")
    
    node_counter = [0]
    root_id = None
    
    def count_subtree_width(n: Node) -> int:
        if not n.children:
            return 1
        return sum(count_subtree_width(c) for c in n.children)
    
    def add_nodes(n: Node, parent_id=None, add_balancers=False, depth=0):
        nonlocal root_id
        node_id = f"node_{node_counter[0]}"
        node_counter[0] += 1
        
        if parent_id is None:
            root_id = node_id
        
        style = _node_style(n, node_size=node_size, font_size=font_size, depth=depth)
        
        # Use FEN from board_state if available for START node
        label = n.label
        if n.label == "START" and n.board_state:
            label = _extract_fen(n.board_state)
        
        dot.node(node_id, label, **style)
        
        if parent_id is not None:
            edge_style = {"color": "#000000", "penwidth": "2"}
            
            if n.status == "illegal":
                edge_style.update({"style": "dashed", "color": "#666666"})
            elif n.status == "final":
                r = (n.result or "").lower()
                if r == "win":
                    edge_style.update({"color": "#00FF41", "penwidth": "3"})
                elif r == "loss":
                    edge_style.update({"color": "#FF073A", "penwidth": "3"})
                elif r == "tie":
                    edge_style.update({"color": "#FFD700", "penwidth": "3"})
            
            dot.edge(parent_id, node_id, **edge_style)
        
        # Add balancing invisible nodes if requested and this is root
        if add_balancers and parent_id is None and n.children:
            widths = [count_subtree_width(c) for c in n.children]
            total_width = sum(widths)
            max_width = max(widths) if widths else 0
            
            # Add invisible nodes to balance
            if max_width > 0 and total_width < max_width * len(n.children):
                needed = max_width * len(n.children) - total_width
                for i in range(needed):
                    invis_id = f"invisible_{i}"
                    dot.node(invis_id, "", style="invis", width="0", height="0")
                    dot.edge(node_id, invis_id, style="invis")
        
        for child in n.children:
            add_nodes(child, node_id, False, depth + 1)
    
    add_nodes(root, add_balancers=force_center)

    if root_id:
        dot.graph_attr['root'] = root_id

    if show_legend:
        leg_items = [
            ('__leg_regular__', 'Regular', '#FFFFFF', '#000000', 'black', 'solid'),
            ('__leg_illegal__', 'Illegal', '#666666', '#AAAAAA', 'white', 'dashed'),
            ('__leg_win__',     'Win',     '#00FF41', '#00FF41', 'black', 'solid'),
            ('__leg_loss__',    'Loss',    '#FF073A', '#FF073A', 'white', 'solid'),
            ('__leg_tie__',     'Tie',     '#FFD700', '#FFD700', 'black', 'solid'),
        ]
        with dot.subgraph() as leg:
            leg.attr(rank='sink')
            for nid, label, fill, color, fontcolor, style in leg_items:
                leg.node(nid, label, shape='box', style=f'filled,{style}',
                         fillcolor=fill, color=color, fontcolor=fontcolor,
                         fontname='Arial', fontsize=str(font_size), penwidth='1.5')
            for i in range(len(leg_items) - 1):
                leg.edge(leg_items[i][0], leg_items[i+1][0], style='invis')

    # Post-process image to add 16:9 aspect ratio and optionally legend
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    
    # Render the graph to PNG first
    if filename:
        output_path = dot.render(filename, cleanup=True, format='png')
        print(f"Saved base to: {output_path}")
    else:
        # Generate temp file for processing
        with tempfile.NamedTemporaryFile(suffix='.gv', delete=False, mode='w') as f:
            temp_path = f.name
            f.write(dot.source)
        output_path = dot.render(temp_path, cleanup=True, format='png')
    
    # Open the rendered image and trim any dark border left by graphviz
    import numpy as np
    img = Image.open(output_path)
    arr = np.array(img)
    dark = np.all(arr < 20, axis=2)
    rows = np.where(~np.all(dark, axis=1))[0]
    cols = np.where(~np.all(dark, axis=0))[0]
    if len(rows) and len(cols):
        img = img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    width, height = img.size
    
    # Calculate target height for 16:9 aspect ratio
    target_height = int(width * 9 / 16)
    
    # If image is already taller than 16:9, use current height + padding
    if height > target_height:
        legend_height = int(font_size * 3) if show_legend else int(font_size * 1.5)
        new_height = height + legend_height
    else:
        # Add black bar to reach 16:9 ratio
        legend_height = int(font_size * 3) if show_legend else 0
        new_height = max(target_height, height + legend_height)
    
    # Create new image with white background
    new_img = Image.new('RGB', (width, new_height), color='#FFFFFF')
    
    # Paste original image at top
    new_img.paste(img, (0, 0))
    
    # Draw legend at bottom right if requested
    if show_legend:
        draw = ImageDraw.Draw(new_img)
        
        # Legend parameters
        legend_font_size = font_size
        cell_width = max(50, int(font_size * 4.5))
        cell_height = max(20, int(font_size * 1.8))
        padding = max(6, int(font_size * 0.5))
        border_width = 2
        
        # Try to load font
        try:
            font = ImageFont.truetype("Arial.ttf", legend_font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", legend_font_size)
            except:
                font = ImageFont.load_default()
        
        # Legend items: (label, bg_color, text_color, border_color, border_style)
        legend_items = [
            ("Regular", "#FFFFFF", "#000000", "#000000", "solid"),
            ("Illegal", "#666666", "#FFFFFF", "#AAAAAA", "dashed"),
            ("Win", "#00FF41", "#000000", None, None),
            ("Loss", "#FF073A", "#FFFFFF", None, None),
            ("Tie", "#FFD700", "#000000", None, None),
        ]
        
        # Calculate total legend width
        total_width = len(legend_items) * (cell_width + padding * 2)
        
        # Position at bottom right with some margin
        margin = 20
        start_x = width - total_width - margin
        start_y = new_height - cell_height - padding * 2 - margin
        
        # Draw each legend item
        x = start_x
        for label, bg_color, text_color, border_color, border_style in legend_items:
            # Draw cell background
            draw.rectangle(
                [x, start_y, x + cell_width + padding * 2, start_y + cell_height + padding * 2],
                fill=bg_color
            )
            
            # Draw border if needed
            if border_color:
                if border_style == "dashed":
                    # Draw dashed border
                    for i in range(0, cell_width + padding * 2, 10):
                        draw.line([x + i, start_y, x + min(i + 5, cell_width + padding * 2), start_y], 
                                 fill=border_color, width=border_width)
                        draw.line([x + i, start_y + cell_height + padding * 2, 
                                  x + min(i + 5, cell_width + padding * 2), start_y + cell_height + padding * 2], 
                                 fill=border_color, width=border_width)
                    for i in range(0, cell_height + padding * 2, 10):
                        draw.line([x, start_y + i, x, start_y + min(i + 5, cell_height + padding * 2)], 
                                 fill=border_color, width=border_width)
                        draw.line([x + cell_width + padding * 2, start_y + i, 
                                  x + cell_width + padding * 2, start_y + min(i + 5, cell_height + padding * 2)], 
                                 fill=border_color, width=border_width)
                else:
                    draw.rectangle(
                        [x, start_y, x + cell_width + padding * 2, start_y + cell_height + padding * 2],
                        outline=border_color, width=border_width
                    )
            
            # Draw text centered
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x + (cell_width + padding * 2 - text_width) // 2
            text_y = start_y + (cell_height + padding * 2 - text_height) // 2
            draw.text((text_x, text_y), label, fill=text_color, font=font)
            
            x += cell_width + padding * 2
    
    # Save vector PDF directly from graphviz (crisp at any zoom)
    if filename:
        pdf_path = dot.render(filename, cleanup=True, format='pdf')
    else:
        with tempfile.NamedTemporaryFile(suffix='.gv', delete=False, mode='w') as f:
            pdf_temp = f.name
            f.write(dot.source)
        pdf_path = dot.render(pdf_temp, cleanup=True, format='pdf')

    if show_legend:
        print(f"Saved with legend to: {pdf_path}")
    else:
        print(f"Saved to: {pdf_path}")

    # Display the raster PNG inline in the notebook
    display(Image.open(output_path))

    return pdf_path

def visualize_row_forest(df, row_index, trees_col="trees", title=None, filename=None,
                         node_size=0.7, font_size=20, splines_style="polyline", force_center=False, show_legend=False, fig_size="16,10"):
    forest = df.loc[row_index, trees_col]

    # Check if board_state exists in the dataframe and create START with it
    start_node = "START"
    if "board_state" in df.columns:
        board_state = df.loc[row_index, "board_state"]
        if board_state:
            start_node = {"label": "START", "status": "normal", "result": None, "board_state": board_state}
    
    root = to_node([start_node] + list(forest))
    return draw_tree(root, title=title, filename=filename,
                     node_size=node_size, font_size=font_size, splines_style=splines_style, force_center=force_center, show_legend=show_legend, fig_size=fig_size)
