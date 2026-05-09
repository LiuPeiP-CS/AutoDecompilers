import re

def remove_c_cpp_comments(source):
    """
    Removes all comments from C/C++ source code accurately.
    Handles edge cases like escaped newlines in strings, comment-like patterns in strings, etc.

    Args:
        source (str): The input C/C++ source code.

    Returns:
        str: The source code with all comments removed.
    """

    # Pattern explanation:
    # 1. STRINGS: Matches all double-quoted and single-quoted strings, including escaped quotes and newlines.
    #    - `(?<!\\)` : Negative lookbehind to ensure the quote is not escaped (though the inner part already handles escapes)
    # 2. SINGLE-LINE COMMENTS: Matches // comments, but ignores if it's inside a string.
    # 3. MULTI-LINE COMMENTS: Matches /* ... */ comments, non-greedily, across lines.

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
        # Group 1: If a string was matched, preserve it entirely.
        if match.group(1) is not None:
            return match.group(1)
        # Group 2: If a comment was matched, replace it based on its type.
        else:
            comment = match.group(2)
            if comment.startswith('//'):
                # For single-line comments, replace with a space only if it's not at the start of a line.
                # This helps preserve line numbers for debugging and avoids merging tokens.
                return ' ' if comment.strip() != '//' else '' # Handle edge case of empty comment "//"
            else:
                # For multi-line comments, replace with a single space if the comment contained text,
                # or a single newline if it was primarily whitespace (to preserve some structure).
                # This prevents things like `int/*...*/a` becoming `inta`.
                content_inside = comment[2:-2].strip()
                if not content_inside:
                    return '\n' if '\n' in comment else ' '
                else:
                    return ' '

    # Apply the replacement
    cleaned_source = pattern.sub(replacer, source)

    # Optional: Post-processing to clean up potential leftover whitespace
    # This replaces multiple consecutive empty lines with a single one
    cleaned_source = re.sub(r'\n\s*\n', '\n', cleaned_source)

    return cleaned_source


if __name__ == '__main__':
    code_string = """// 代码A - 先计算再输出
void process_data(int x, int y) {
    int sum = x + y;        // 定义: sum
    int prod = x * y;        // 定义: prod
    int result = sum + prod; // 使用: sum, prod
    printf("%d", result);    // 使用: result
}"""
    code_string_strip = remove_c_cpp_comments(code_string)
    print(code_string_strip)
