import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def parse_html_table(html_str: str) -> str:
    """
    Parse an HTML table string (output from PaddleOCR) into a Markdown table.
    """
    if not html_str:
        return ""
        
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find('table')
        if not table:
            return ""
            
        rows = table.find_all('tr')
        if not rows:
            return ""
            
        parsed_table = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            parsed_row = [cell.get_text(strip=True).replace("|", "\\|") for cell in cells]
            if any(parsed_row):
                parsed_table.append(parsed_row)
                
        if not parsed_table:
            return ""
            
        cols = max(len(r) for r in parsed_table)
        for r in parsed_table:
            while len(r) < cols:
                r.append("")
                
        header = "| " + " | ".join(parsed_table[0]) + " |"
        sep = "| " + " | ".join(["---"] * cols) + " |"
        
        body = ""
        if len(parsed_table) > 1:
            body = "\n".join("| " + " | ".join(r) + " |" for r in parsed_table[1:])
            
        if body:
            return header + "\n" + sep + "\n" + body
        else:
            return header + "\n" + sep
            
    except Exception as e:
        logger.error(f"Error parsing HTML table: {e}")
        return html_str # Return raw if parsing fails
