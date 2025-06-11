-- TUIs.tsv
SELECT DISTINCT TUI as 'TUI:ID', STY as 'name', STN as 'STN' from umls.MRSTY;

-- CUIs.tsv
SELECT DISTINCT CUI as 'CUI:ID' from umls.MRCONSO where umls.MRCONSO.ISPREF = 'Y' AND umls.MRCONSO.STT = 'PF' AND umls.MRCONSO.TS = 'P' and umls.MRCONSO.LAT = 'ENG';

-- CUI-TUIs.tsv
SELECT DISTINCT CUI AS ':START_ID', TUI AS ':END_ID' from umls.MRSTY;

-- CUI-CUIs.tsv
-- it takes all sabs which are english. then performs inner join with english sabs so that only those rows are selected which are english.
WITH SABlist AS (SELECT DISTINCT SAB FROM umls.MRCONSO WHERE umls.MRCONSO.LAT = 'ENG') 
SELECT DISTINCT CUI2 AS ':START_ID', CUI1 AS ':END_ID', NVL(RELA, REL) AS ':TYPE', umls.MRREL.SAB 
FROM umls.MRREL INNER JOIN SABlist ON umls.MRREL.SAB = SABlist.SAB 
WHERE umls.MRREL.SUPPRESS <> 'O' AND CUI1 <> CUI2 AND REL <> 'SIB';

-- CODEs.tsv
With CUIlist as (SELECT DISTINCT CUI from umls.MRCONSO where umls.MRCONSO.ISPREF = 'Y' AND umls.MRCONSO.STT = 'PF' AND umls.MRCONSO.TS = 'P' and umls.MRCONSO.LAT = 'ENG') SELECT DISTINCT (umls.MRCONSO.SAB||' '||umls.MRCONSO.CODE) as 'CodeID:ID', umls.MRCONSO.SAB, umls.MRCONSO.CODE from umls.MRCONSO inner join CUIlist on umls.MRCONSO.CUI = CUIlist.CUI where umls.MRCONSO.LAT = 'ENG' and SUPPRESS <> 'O' ;

-- CUI-CODEs.tsv
SELECT DISTINCT CUI as ':START_ID', (SAB||' '||CODE) as ':END_ID' from umls.MRCONSO where LAT = 'ENG' and SUPPRESS <> 'O';

-- SUIs.tsv
SELECT DISTINCT umls.MRCONSO.SUI as 'SUI:ID', umls.MRCONSO.STR as 'name' FROM umls.MRCONSO WHERE umls.MRCONSO.LAT = 'ENG';

-- CODE-SUIs.tsv
SELECT DISTINCT SUI as ':END_ID', (SAB||' '||CODE) as ':START_ID', TTY as ':TYPE', CUI as CUI from umls.MRCONSO where LAT = 'ENG' and SUPPRESS <> 'O';

-- CUI-SUIs.tsv
SELECT DISTINCT CUI as ':START_ID', SUI as ':END_ID' from umls.MRCONSO where umls.MRCONSO.ISPREF = 'Y' AND umls.MRCONSO.STT = 'PF' AND umls.MRCONSO.TS = 'P' and umls.MRCONSO.LAT = 'ENG';

-- DEFs.tsv
With CUIlist as (SELECT DISTINCT CUI from umls.MRCONSO where umls.MRCONSO.ISPREF = 'Y' AND umls.MRCONSO.STT = 'PF' AND umls.MRCONSO.TS = 'P' and umls.MRCONSO.LAT = 'ENG') SELECT DISTINCT umls.MRDEF.ATUI as 'ATUI:ID', umls.MRDEF.SAB, umls.MRDEF.DEF FROM umls.MRDEF inner join CUIlist on umls.MRDEF.CUI = CUIlist.CUI where SUPPRESS <> 'O' AND NOT (umls.MRDEF.SAB LIKE 'MSH%' AND umls.MRDEF.SAB <> 'MSH') AND NOT (umls.MRDEF.SAB LIKE 'MDR%' AND umls.MRDEF.SAB <> 'MDR');

-- DEFrel.tsv
SELECT DISTINCT ATUI as ':END_ID', CUI as ':START_ID' from umls.MRDEF where SUPPRESS <> 'O';

-- NDCs.tsv
SELECT DISTINCT ATUI as 'ATUI:ID', ATV as 'NDC' from umls.MRSAT where SAB = 'RXNORM' and ATN = 'NDC' and SUPPRESS <> 'O';

-- NDCrel.tsv
SELECT DISTINCT ATUI as ':END_ID', (SAB||' '||CODE) as ':START_ID' from umls.MRSAT where SAB = 'RXNORM' and ATN = 'NDC' and SUPPRESS <> 'O';

-- Triplets.csv (COLUMNS: SUI1, REL, SUI2, SAB1, SAB2) (SUI1, SUI2 are SUIs and REL is the relationship between concepts that points to SUI1 and SUI2) (SAB1 and SAB2 are related to the SUI1 and SUI2 respectively)


