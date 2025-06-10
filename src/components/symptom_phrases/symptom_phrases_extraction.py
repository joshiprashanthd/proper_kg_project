from src.utils import OpenAIModel
from src.types import SymptomPhrase
from pydantic import BaseModel, Field
from src.utils import logger

class ResponseFormat(BaseModel):
    symptom_phrases: list[SymptomPhrase] = Field(description="The list of symptom phrases.")

class SymptomPhrasesExtractor:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = OpenAIModel(model_name, 'cuda')
    
    def extract_symptom_phrases(self, text: str) -> list[SymptomPhrase]:
        prompt = """You are a medical and mental health expert. Your goal is to extract phrases from the patient's text that indicate symptoms and provide a detailed analysis for each phrase. Each analysis should include a descriptive symptom for the symptom or condition, identify potential hidden difficulties the patient might be experiencing, and provide additional context or information to better understand the phrase's implications. Symptom should reflect the specific symptom or condition (e.g., "Social withdrawal," "Chronic fatigue," "Intrusive thoughts").

Learn from the examples below:

Example 1:
Text:
I no longer hate myself in the same way. I feel somewhat better now that I’ve accepted my ugly appearance, shy personality, and few mental health issues. Before, I used to despise myself for these things. Although I am aware that I will never be loved, have a successful career, be a good friend, or find a girlfriend, I no longer feel horrible about myself because I accept how boring and ugly I am. The only thing that worries me is that I am more foolish than I care to acknowledge.

Output:
[]

Example 2:
Text:
I’m in pain all the time. Really, what’s the purpose of existence? I literally experience daily, unrelenting emotional pain. and the best i can do is distract myself from it. I literally experience constant emotional pain every single day. Most of the time, all I want to do is curl up into a ball and cry my eyes out, but I’m so disconnected that I can’t cry anymore.. I mean, I won’t kill myself, but when every day is agony, how is it possible for anyone to really think that I would want to live? Nothing ever seems to improve the situation, and I can feel my condition gradually deteriorating every day. I wish I could just vanish.

Output:
phrase: I literally experience daily, unrelenting emotional pain.
symptom: Persistent emotional distress
analysis: The patient describes a pervasive and unremitting sense of emotional pain occurring daily, suggesting a severe depressive state. Hidden difficulties may include an inability to engage in daily activities or maintain relationships due to overwhelming emotional burden. This could indicate major depressive disorder or dysthymia, potentially compounded by feelings of hopelessness.

phrase: I literally experience constant emotional pain every single day.
symptom: Chronic emotional pain
analysis: This reinforces the persistent nature of the patient’s emotional suffering, indicating a lack of relief or respite. Hidden difficulties might include impaired cognitive function or decision-making, as chronic pain can reduce mental clarity. The repetition of this sentiment suggests a deeply entrenched emotional state.

phrase: Most of the time, all I want to do is curl up into a ball and cry my eyes out, but I’m so disconnected that I can’t cry anymore.
symptom: Emotional numbness
analysis: The desire to withdraw and cry, paired with an inability to do so, points to emotional numbness, a common symptom of severe depression. Hidden difficulties may include social isolation or difficulty processing emotions, leading to a sense of detachment. This could also suggest anhedonia, where the patient struggles to experience pleasure or emotional release.

phrase: I wish I could just vanish.
symptom: Suicidal ideation
analysis: This phrase indicates passive suicidal thoughts, where the patient expresses a desire to cease existing without active intent to self-harm. Hidden difficulties may include a lack of purpose or motivation, increasing the risk of worsening mental health.

Example 3:
Text:
What just I experienced. I suddenly experienced a wave of extreme anxiety around 9:40 p.m. I had to lock every door, but I swear I saw someone when I peered outside. I swear I saw someone, but they would have to be taller than seven feet. My dad hurried to tell my parents after he went outside and saw nobody. This should normally calm people down, right? It didnt ease me. I still feel as though I’m going to die or am in danger. I couldnt stop shaking and I was only able to stop crying at 10 p.m. My chest and head hurt really bad now. Was this an extreme case of anxiety, or something else?

Output:
phrase: It didnt ease me.
symptom: Persistent fear
analysis: Despite external reassurance, the patient’s fear persists, indicating a heightened state of anxiety that is resistant to calming influences. Hidden difficulties may include hypervigilance or difficulty trusting their environment, potentially linked to an anxiety disorder or trauma response. This could impair their ability to feel safe in familiar settings.

phrase: I still feel as though I’m going to die or am in danger.
symptom: Sense of impending doom
analysis: This phrase suggests an intense, irrational fear of imminent harm, a hallmark of acute anxiety or panic attacks. Hidden difficulties may include physical symptoms like elevated heart rate or difficulty breathing, which reinforce the fear. This could indicate a panic disorder or acute stress response.

phrase: I couldnt stop shaking
symptom: Trembling
analysis: Uncontrollable shaking is a physical manifestation of extreme anxiety or panic. Hidden difficulties may include physical exhaustion or muscle tension, which could exacerbate discomfort. This symptom may be part of a broader anxiety disorder, and techniques like grounding.

phrase: My chest and head hurt really bad now.
symptom: Physical pain
analysis: Physical pain in the chest and head during an anxiety episode may indicate somatic symptoms of anxiety or a panic attack. Hidden difficulties could include misattribution of these symptoms to a medical emergency, increasing fear.

Example 4:
Text:
Stuck after graduation. August was when I graduated, and I still haven’t found employment in my field. I continue to work a few hours a week at my menial retail job, and because I am so severely depressed and lonely, I am afraid to apply to as many jobs as I should because I feel incompetent and afraid, literally frozen with fear. I’ll die over it. I spend everyday in a constant cycle of loneliness and anxiety and worry and panic and emptiness. I feel like I’m burdening my family more and more every day because I can’t seem to get a job and get my life started. I simply want to put an end to this years-long suffering because it keeps getting worse. I despise myself for it, Due to family issues, I’ve literally cut off from my friends and am dreading Christmas. and talk about why Im not working yet.

Output:
phrase: severely depressed and lonely
symptom: Depressive symptoms
analysis: The patient explicitly mentions severe depression and loneliness, indicating significant emotional distress. Hidden difficulties may include low self-esteem and lack of motivation, hindering job searches or social engagement. This could be indicative of major depressive disorder.

phrase: I feel incompetent and afraid, literally frozen with fear
symptom: Paralyzing fear
analysis: The patient’s fear of incompetence prevents job applications, suggesting a debilitating anxiety response. Hidden difficulties may include perfectionism or fear of rejection, leading to procrastination. This could be linked to social anxiety or generalized anxiety disorder.

phrase: constant cycle of loneliness
symptom: Chronic loneliness
analysis: Persistent loneliness suggests a lack of meaningful social connections, exacerbating depressive symptoms. Hidden difficulties may include difficulty initiating or maintaining relationships due to low self-worth.

phrase: anxiety and worry and panic and emptiness
symptom: Emotional distress
analysis: This mix of anxiety, worry, panic, and emptiness reflects a complex emotional state, likely involving both anxiety and depression. Hidden difficulties may include emotional overwhelm, making daily functioning challenging.

phrase: Due to family issues, I’ve literally cut off from my friends and am dreading Christmas
symptom: Social withdrawal
analysis: The patient’s isolation from friends due to family issues and dread of social events like Christmas indicate significant social withdrawal. Hidden difficulties may include unresolved family conflict or shame about their situation, further isolating them.

Extract phrases from the text relevant to the patient’s symptoms, assign a specific, descriptive label to each, and provide a detailed analysis including hidden difficulties and additional context to clarify the phrase’s implications.
Text:
{text}

Output:
"""
        prompt = prompt.format(text=text)
        response = self.model.generate_text(prompt, structured_format=ResponseFormat, max_tokens=4096)
        logger.info(f"Extracted {len(response.symptom_phrases)} symptom phrases for text: {text}")
        return response.symptom_phrases