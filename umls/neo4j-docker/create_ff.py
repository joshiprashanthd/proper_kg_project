import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from pathlib import Path

DB_USER = 'root'
DB_PASS = 'admin'
DB_HOST = '127.0.0.1'
DB_PORT = '4000'
DB_NAME = 'umls'

engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8')

folder = Path("flat_files_umls_full_pref")
folder.mkdir(parents=True, exist_ok=True)

queries = {
#     "TUIs.csv": """SELECT DISTINCT TUI as "TUI\:ID", STY as "name", STN as "STN" from umls_full.MRSTY""",
#     "CUIs.csv": """SELECT DISTINCT CUI AS "CUI\:ID"
#                    FROM umls_full.MRCONSO 
#                    WHERE ISPREF = 'Y' AND STT = 'PF' AND TS = 'P' and LAT = 'ENG'""",
#     "SUIs.csv": """
#         SELECT DISTINCT SUI as "SUI\:ID", STR as "name" 
#         FROM umls_full.MRCONSO
#         WHERE LAT = 'ENG'
#     """,
#     "DEFs.csv": """
#         WITH CUIlist AS (
#             SELECT DISTINCT CUI 
#             FROM umls_full.MRCONSO 
#             WHERE ISPREF = 'Y' AND STT = 'PF' AND TS = 'P' and LAT = 'ENG'
#         )

#         SELECT DISTINCT MRDEF.ATUI as "ATUI\:ID", MRDEF.SAB, MRDEF.DEF 
#         FROM umls_full.MRDEF
#         INNER JOIN CUIlist ON MRDEF.CUI = CUIlist.CUI 
#         WHERE SUPPRESS <> 'O'
#           AND NOT (SAB LIKE 'MSH%' AND SAB <> 'MSH')
#           AND NOT (SAB LIKE 'MDR%' AND SAB <> 'MDR')
#     """,
#     "CUI-TUIs.csv": """SELECT DISTINCT CUI as '\:START_ID', TUI as '\:END_ID' from umls_full.MRSTY""",
#     "CUI-CUIs.csv": """
#         WITH SABlist AS (SELECT DISTINCT SAB from umls_full.MRCONSO where LAT = 'ENG') 
#         SELECT DISTINCT CUI2 AS "\:START_ID", CUI1 AS "\:END_ID", NVL(RELA, REL) AS "\:TYPE", MRREL.SAB 
#         FROM umls_full.MRREL
#         INNER JOIN SABlist ON MRREL.SAB = SABlist.SAB 
#         WHERE MRREL.SUPPRESS <> 'O' AND CUI1 <> CUI2 AND REL <> 'SIB'
#     """,
#     "CUI-SUIs.csv": """
#         SELECT DISTINCT CUI AS "\:START_ID", SUI AS "\:END_ID" 
#         FROM umls_full.MRCONSO 
#         WHERE ISPREF = 'Y' AND STT = 'PF' AND TS = 'P' and LAT = 'ENG'
#     """,
#     "DEFrel.csv": """SELECT DISTINCT ATUI AS "\:END_ID", CUI AS "\:START_ID" from umls_full.MRDEF where SUPPRESS <> 'O'""",
#     "CODEs.csv": """
#     WITH CUIlist AS (
#         SELECT DISTINCT CUI FROM umls_full.MRCONSO 
#         WHERE ISPREF = 'Y' AND STT = 'PF' AND TS = 'P' AND LAT = 'ENG'
#     )
#     SELECT DISTINCT CONCAT(MRCONSO.SAB, ' ', MRCONSO.CODE) AS "\:ID", 
#                     MRCONSO.SAB, MRCONSO.CODE
#     FROM umls_full.MRCONSO 
#     INNER JOIN CUIlist ON MRCONSO.CUI = CUIlist.CUI 
#     WHERE MRCONSO.LAT = 'ENG' AND SUPPRESS <> 'O'
# """,

#    "CUI-CODEs.csv": """
#     SELECT DISTINCT CUI AS "\:START_ID", 
#                     CONCAT(SAB, ' ', CODE) AS "\:END_ID" 
#     FROM umls_full.MRCONSO 
#     WHERE LAT = 'ENG' AND SUPPRESS <> 'O'
# """,
# "CODE-SUIs.csv": """
#     SELECT DISTINCT SUI AS "\:END_ID", 
#                     CONCAT(SAB, ' ', CODE) AS "\:START_ID", 
#                     TTY AS "\:TYPE", 
#                     CUI 
#     FROM umls_full.MRCONSO 
#     WHERE LAT = 'ENG' AND SUPPRESS <> 'O'
# """,

    # "NDCs.csv": """SELECT DISTINCT ATUI as "ATUI\:ID", ATV as "NDC" from umls_full.MRSAT where SAB = 'RXNORM' and ATN = 'NDC' and SUPPRESS <> 'O'""",
    # "NDCrel.csv": """SELECT DISTINCT ATUI as "\:END_ID", (SAB||' '||CODE) as "\:START_ID" from umls_full.MRSAT where SAB = 'RXNORM' and ATN = 'NDC' and SUPPRESS <> 'O'"""
    "Triplets.csv": """
WITH FilteredRelationships AS (
    -- Step 1: Select relevant relationships from MRREL.
    -- CUI2 is treated as the starting concept and CUI1 as the ending concept,
    -- consistent with the CUI-CUIs.tsv query structure (START_ID = CUI2, END_ID = CUI1).
    SELECT DISTINCT
        mr.CUI2 AS CUI_START,        -- This will be the concept associated with TUI1 and SAB1
        mr.CUI1 AS CUI_END,          -- This will be the concept associated with TUI2 and SAB2
        NVL(mr.RELA, mr.REL) AS REL_TYPE -- The relationship type
    FROM
        umls.MRREL mr
    INNER JOIN
        -- Ensure the relationship assertion comes from a source that has English terms
        (SELECT DISTINCT SAB FROM umls.MRCONSO WHERE LAT = 'ENG') EngSources
        ON mr.SAB = EngSources.SAB
    WHERE
        mr.SUPPRESS <> 'O'       -- Exclude suppressed relationships
        AND mr.CUI1 <> mr.CUI2   -- Exclude self-relationships (if not desired)
        AND mr.REL <> 'SIB'      -- Exclude sibling relationships (as in CUI-CUIs.tsv)
)
SELECT DISTINCT
    tui_start.TUI AS TUI1,          -- TUI of the starting concept
    fr.REL_TYPE AS REL,             -- Relationship type
    tui_end.TUI AS TUI2,            -- TUI of the ending concept
    sab_start.SAB AS SAB1,          -- Source vocabulary of the starting concept
    sab_end.SAB AS SAB2             -- Source vocabulary of the ending concept
FROM
    FilteredRelationships fr
INNER JOIN
    umls.MRSTY tui_start ON fr.CUI_START = tui_start.CUI -- Get TUI for the starting concept
INNER JOIN
    umls.MRSTY tui_end ON fr.CUI_END = tui_end.CUI       -- Get TUI for the ending concept
INNER JOIN
    umls.MRCONSO sab_start ON fr.CUI_START = sab_start.CUI -- Get SAB for the starting concept
    AND sab_start.LAT = 'ENG'                             -- Consider only English atoms for the concept's SAB
    AND sab_start.SUPPRESS <> 'O'                         -- Exclude suppressed atoms
INNER JOIN
    umls.MRCONSO sab_end ON fr.CUI_END = sab_end.CUI     -- Get SAB for the ending concept
    AND sab_end.LAT = 'ENG'                               -- Consider only English atoms for the concept's SAB
    AND sab_end.SUPPRESS <> 'O'                           -- Exclude suppressed atoms
ORDER BY
    TUI1, REL, TUI2, SAB1, SAB2; -- Optional: for consistent output ordering
"""
}

with engine.connect() as conn:
    for filename, sql in tqdm(queries.items()):
        print(f"Running: {filename}")
        df = pd.read_sql(text(sql), conn)
        df.to_csv(folder / filename, index=False)
