import re

def remove_c_cpp_comments(source):


    pattern = re.compile(
        r'('
        r'\"(?:\\.|[^\"\\])*\"'  # Double-quoted strings
        r'|\'(?:\\.|[^\'\\])*\''  # Single-quoted characters
        r')'
        r'|' 
        r'('
        r'//.*?$'  # Single-line comments
        r'|'
        r'/\*[\s\S]*?\*/'  # Multi-line comments (non-greedy, across lines)
        r')',
        re.MULTILINE
    )

    def replacer(match):
        
        if match.group(1) is not None:
            return match.group(1)
        else:
            comment = match.group(2)
            if comment.startswith('//'):
                return ' ' if comment.strip() != '//' else '' # Handle edge case of empty comment "//"
            else:
                content_inside = comment[2:-2].strip()
                if not content_inside:
                    return '\n' if '\n' in comment else ' '
                else:
                    return ' '

    
    cleaned_source = pattern.sub(replacer, source)

    
    cleaned_source = re.sub(r'\n\s*\n', '\n', cleaned_source)

    return cleaned_source
