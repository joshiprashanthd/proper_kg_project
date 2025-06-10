import re

NO_DEF_PATTERN = re.compile(r'<\|\s*NO_DEFINITION\s*\|>', re.IGNORECASE)
FIRST_DEF_PATTERN = re.compile(
    r'(?sm)^\s*1\.\s*'           
    r'(.*?)'                     
    r'(?=(?:\r?\n)\s*\d+\.|\Z)' 
)
TRIPLET_PATTERN = re.compile(
    r"""(?sx)                     
    \(Head:\s*'(?P<head>.*?)',\s*Definition:\s*'(?P<head_def>.*?)'\)
    \s*-\[(?P<rel>.*?)\]->
    \s*\(Tail:\s*'(?P<tail>.*?)',\s*Definition:\s*'(?P<tail_def>.*?)'\)
    """
)

def extract_first_definition(def_text: str) -> str:
    if NO_DEF_PATTERN.search(def_text):
        return '<|NO_DEFINITION|>'
    m = FIRST_DEF_PATTERN.search(def_text)
    if m:
        text = m.group(1).strip()
    else:
        text = def_text.strip()
    return text.rstrip('.')

def keep_first_definition(triplets: list[str]) -> list[str]:
    cleaned = []
    for line in triplets:
        m = TRIPLET_PATTERN.search(line)
        if not m:
            continue

        head, head_def = m.group('head'), m.group('head_def')
        tail, tail_def = m.group('tail'), m.group('tail_def')
        rel = m.group('rel')

        h_def1 = extract_first_definition(head_def)
        t_def1 = extract_first_definition(tail_def)

        cleaned.append(
            f"(Head: '{head}', Definition: '{h_def1}')"
            f"-[{rel}]->"
            f"(Tail: '{tail}', Definition: '{t_def1}')"
        )
    return cleaned