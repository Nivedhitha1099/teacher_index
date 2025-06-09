import os
import streamlit as st #
import pdfplumber
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from dotenv import load_dotenv

load_dotenv()

def initialize_chat_model():
    """
    Initializes and returns a ChatAnthropic model.
    Expects LLMFOUNDRY_TOKEN environment variable to be set.
    """
    try:
        token = os.getenv("LLMFOUNDRY_TOKEN")
        if not token:
            if 'st' in globals():
                st.error("LLMFOUNDRY_TOKEN environment variable not found. Please set it.")
            else:
                print("Error: LLMFOUNDRY_TOKEN environment variable not found. Please set it.")
            return None

        chat_model = ChatAnthropic(
            anthropic_api_key=f'{token}:my-test-project',
            base_url="https://llmfoundry.straive.com/anthropic/",
            model_name="claude-3-haiku-20240307",
             max_tokens=4096
        )
        return chat_model
    except Exception as e:
        if 'st' in globals():
            st.error(f"Failed to initialize AI model: {str(e)}")
        else:
            print(f"Error: Failed to initialize AI model: {str(e)}")
        return None


def extract_text_with_page_numbers(pdf_path):
    """
    Extracts text from a PDF, associating each text block with its page number.
    Returns a list of dictionaries: [{"page": 1, "text": "..."}]
    """
    pages_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                text = page.extract_text()
                if text:
                    pages_content.append({"page": page_number, "text": text})
    except Exception as e:
        if 'st' in globals():
            st.error(f"Error extracting text from PDF: {str(e)}")
        else:
            print(f"Error extracting text from PDF: {str(e)}")
        return None
    return pages_content


def generate_teacher_guide_index_with_llm(chat_model, pdf_text_with_pages):
    """
    Sends extracted PDF text to the LLM with a prompt to generate a teacher guide index
    mimicking the structure of 'SK17 G1 Index (1).pdf'.
    Includes chunking logic for the input to manage large documents.
    Adds instructions for the LLM to indicate completion or truncation.
    """
    if not chat_model:
        return "LLM model not initialized."

    # Construct the input string for the LLM
    # THIS PROMPT IS NOW CRITICAL FOR MIMICKING THE TARGET FORMAT AND HANDLING COMPLETION.
    base_prompt = "Analyze the following text extracted from a student workbook. " \
                  "Your task is to create a 'Teacher's Guide Index' from this content. " \
                  "The index should be structured and formatted exactly like a professional index " \
                  "found in a teacher's guide, similar to the 'SK17 G1 Index (1).pdf' example.\n\n" \
                  "Follow these strict formatting rules:\n" \
                  "1.  **Alphabetical Order:** All top-level entries must be sorted alphabetically.\n" \
                  "2.  **Main Entries:** Identify broad educational concepts, skills, or topics. " \
                  "    Each main entry should be on its own line, followed by a colon if it has sub-entries.\n" \
                  "3.  **Sub-Entries:** Indent specific details or sub-topics under their main entries. " \
                  "    Each sub-entry should be on its own line, beginning with an indent.\n" \
                  "4.  **Page Referencing:** For each entry (main or sub), list the page numbers where the concept is found. " \
                  "    The format should be 'Page X, Page Y' or 'Pages X-Y' if a range. " \
                  "    **IMPORTANT:** If you can infer a 'Unit' (e.g., from context or numbering in the original document), " \
                  "    format it as 'Unit/Page' (e.g., '1/101'). If no unit is discernible, just use 'Page X'.\n" \
                  "5.  **Cross-Referencing ('See' and 'See also'):** Use 'See [Other Topic]' to direct users to the preferred or more comprehensive entry for a topic. " \
                  "    Place this on a new line after the entry, or as a standalone entry if the term itself is just a redirect. " \
                  "    For example: 'Abbreviations, understand. See Vocabulary.'\n" \
                  "6.  **Conciseness:** Keep entries brief and to the point.\n" \
                  "7.  **Punctuation:** Use consistent punctuation (e.g., comma-separated page numbers, period at end of some entries if they are sentences).\n\n" \
                  "**Crucial Instruction:** Ensure the index is exhaustive for the provided content. At the very end of the generated index, add the phrase '--- END OF INDEX ---'. " \
                  "If, for any reason, you are unable to complete the full index due to length, clearly state '--- INDEX INCOMPLETE ---' at the point of truncation.\n\n" \
                  "Example Format (Adhere strictly to this structure for *all* entries you generate):\n" \
                  "Assessment:\n" \
                  "  End-of-Unit Assessment: Page 101, Page 98\n" \
                  "  Informal Assessment: See Daily Routines\n" \
                  "Fluency:\n" \
                  "  Accuracy, read with. See Fluency.\n" \
                  "  Words correct per minute (WCPM): Page 55\n" \
                  "Phonics:\n" \
                  "  Long Vowels:\n" \
                  "    ai/ay: Page 4, Page 5\n" \
                  "  Short vowels: Page 122\n\n" \
                  """A - Abbreviations, understand. Accuracy, read with. Adjectives, understand. Adverbs, understand. Alphabetical order, use. Answer questions about literary and informational text. Antonyms, understand. Apostrophes, recognize and use. Art activities. Assessment End-of-Unit Assessment Informal Assessment Placement Test Progress tests Spelling tests

B - Base words, identify. Blend sounds to decode words.

C - Capitalization. Categorizing. Cause and effect, determine. Characters, understand. Commas, recognize and use. Compare and contrast text or pictures. Composition. Compound words, understand. Comprehension Skills Associate pictures with a story or poem Associate pictures or signs with sentences Associate pictures with words Comprehension Strategies Answer questions about informational text Answer questions about literary text Generate questions for investigation Monitor comprehension and use fix-up tips Prior knowledge, use Summarize Text structure, recognize Visualize

D - Daily Routines: Informal Assessment Decoding, Spelling, Handwriting routines Memory Word routines Decoding Describe people, places, things, or events. Descriptive language, appreciate and use. Details, recall. Dictionary skills. Directions, follow Oral Written Discussions, participate in. Drama activities. Draw conclusions.

E - Encoding. End punctuation. Exclamation marks. Expression, read with.

F - Fiction. Figurative language, appreciate and use. Final sounds. Fix-up tips. Fluency Accuracy, read with Expression, read with Natural or appropriate phrasing, read with Punctuation, observe Rate, read with appropriate Repeated words and phrases, read Rhythmically, read Speech balloons, read Stress, read with appropriate Typographical clues, observe Volume, read with appropriate

G - Games Decoding games “Bingoo” “Concentration” “Counting Rhyme” “Fiddlestick Rhyming Fun” “Fishing Game” “Hot Potato” Matching Game “Pass the Rhyme” “Rhymin’ Numbers” “Rhymin’ Pop Up” “Rhymin’ Simon” “Simon Says” “Spin the Bottle” “Word Catch” Spelling games “Bingoo” “Build a Monster Face” Fiddlestick Spelling Game “Fix the Spelling” “Hangman” Happy Land Game “Hot Potato” “Match It!” Memory Word Game “Memory Word Toss” “Pit Crew Races” “Spelling Fix-It” “Spin the Bottle” “Tic-Tac-Toe Trickers” Vocabulary games “Adjective I Spy” Beach Ball Game “Compound Concentration” “I Spy” “Memory Word Concentration” “Question Quiz Show” “Question Word Bingo” Race Car Game “Roll and Read” “Say the Opposite” Slap Memory Word Cards “Toss It” “What Did It Do?” Generate questions. Genre, identify. Grammar, Usage, and Mechanics Adjectives, understand Comparative adjectives, understand Superlative adjectives, understand Adverbs, understand Capitalization Beginning of sentences Proper nouns Titles Conjunctions and or Nouns, understand Plurals, understand Possessives, singular, recognize Prepositional phrases Pronouns, understand Punctuation Apostrophes, recognize and use Commas, recognize and use Exclamation marks, recognize and use Periods, recognize and use Question marks, recognize and use Quotation marks, recognize and use Question words, understand Sentences Complete sentences, recognize Types of sentences Commands (Imperative) Declarative Exclamatory Interrogative Verbs, understand Verbs, use to understand time of action Graphic organizers Charts Five Senses Organizer Reading Log Word webs

H - Handwriting Letters, form Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj Kk Ll Mm Nn Oo Pp Qq Rr Ss Tt Uu Vv Ww Xx Yy Zz Handwriting, Daily. Health and safety activities. High-frequency words. Homographs, understand. Homophones, understand. Idioms, understand.

I - Important ideas, determine. Independent Activities Initial sounds. Informational Text “How to Plant Carrots” Library Books Super-Duper Super Smart Informational Digital Read-Alouds Informational text words, discuss. Integrated Curriculum.

L - Language Arts. Lasting Lessons Asking Nicely Avoid Jumping to Conclusions Being a Good Sport Being Patient Calming Down When Upset Clearing Up Misunderstandings Deciding How to Play Together Doing the Right Thing for Its Own Sake Getting Good Ideas Giving It a Try Helping Others Helping Someone Feel Better Helping Your Community Keep Trying Keeping Fit Learning with Practice Looking Out for Others Making Good Use of Time Making Group Decisions Respecting Nature Responding to Teasing Reusing and Recycling Solving Problems Taking Care of Public Places Taking Responsibility Taking Turns and Working Together Talking About Fears Thanking Others Politely Thinking for Yourself Waiting Patiently Lesson taught by literary or informational text Letter Recognition Vowels and consonants, distinguish between Letter-sound correspondences Listening Directions, follow oral Informational text, listen and respond to Literary text, listen and respond to Multimedia text, listen to and discuss Purposes, listen for different Songs, listen to and discuss Literary Text Reader, main stories “Buster’s Surprise” “The Case of the Mystery Monster” “Fiddlesticks” “Fire!” “The Flat Cat” “The Foolish Giant” “For the Birds” “Get Fit” “Golly and the Vet” “Golly Helps” “Help!” “In a Pickle” “In Case of Rain” “The Lesson” “Lily’s Desert Project” “The Little Horse” “The Lost Mitt” “The Monster Under the Bus” “The Patch-it-up Shop” “Play Ball!” “Race Day” “Slumber Party” “The Spingle Spangle Talent Show” “A Super Day at Happy Land” “Tex McGraw’s Visit” “That Was Yesterday” “Toc’s Chicken Pox” “The Very Best Gift” “What a Pet!” “What Can You Get with a Nickel?” “The Wish” “Yuck! Yuck!” “Zoo Clue” Superkids Shorts Literature, respond to Locate information.

M - Main idea or topic, identify. Math activities. Mechanics. Memory Words a about again always any are be because been before both boy buy cold come coming could day do does done down eight find first for four from girl give good have he her here his hold how I kind know laugh light like live look many me my new no now of oh old once one only or our out over put right said she show some the their there they to too two very walk want warm was wash we were what when where which who why work would write you your Monitor comprehension and use fix-up tips. Multiple-meaning words, understand.

N - Natural or appropriate phrasing, read with. Nonfiction. Nouns, understand.

O - Onomatopoeia (words for sounds), recognize. Opinions, give and support. Oral directions, follow. Oral language.

P - Parts of a book, identify. Pattern words. Patterns in text, recognize. Periods, recognize and use. Personification, understand. Phonemic Awareness Blend letter-sounds Distinguish between short- and long-vowel sounds Identify letter-sounds in words Phonics Blend sounds to decode words Letter-sound correspondences Patterns, consonant and vowel Reading Rules Rhyming words. Segmenting. Trickers Word families Words with endings Phonological Awareness Rhyming words, identify and produce Physical education activities. Picture-text relationships, understand. Pictures Associate pictures with a story or poem Associate pictures or signs with sentences Associate pictures with words

P - Plays The Contest It’s So Hot Pleasant’s Pointers Plot: beginning, middle, and end, recognize. Plot: problem and solution, recognize. Plurals, understand. Poems and Rhymes “Ettabetta’s E-mail” “Ettabetta’s Radish Patch” “A Gift I Like” “Golly Went Sniffing” “My Happy Rainy Day” “Super-Duper Golly!” “Super Scrub-a-matic” “When the Superkids Pretend” Possessives, recognize. Predictions, make and confirm. Prefixes, understand. Prepositional phrases. Print and Book Awareness Parts of a book, identify Speech or thought balloons, understand use of Title of a literary or informational text, discuss

Q - Question marks, recognize and use. Question words, understand. Questions, generate. Quotation marks, recognize and use.

R - Rate, read at appropriate. Read-aloud texts Informational Text Super Smart Informational Digital Read-Alouds Literary Text The Superkids’ Summer Adventures Reading Rules Reality and fantasy, distinguish between. Rebuses, identify. Recite poems, rhymes or songs. References and resources, use. Repeated words and phrases, read. Respond to literature. Retell stories or information. Rhyming words. Rhythm and rhyme, recognize. Rhythmically read. Riddles, ask and answer.

S - Science activities. Segmenting. Sentences. Sequence of events or steps, understand. Sequence words, understand. Setting, describe. Skill extension, reinforcement, and reteaching. Social studies activities. Songs. Sound-symbol relationships. Spanish words. Speaking Describe people, places, things, or events Discussions, participate in Recite poems, rhymes or songs Retell stories or information Riddles, ask and answer Stories, compose Speech balloons, read Speech or thought balloons, understand use of Spelling Compound words, spell Contractions, spell Decodable words, spell Letters for sounds Spelling patterns Spelling rules Trickers Words with endings Spelling, Daily. Spelling tests Stories. Stories, tell or retell. Story/poem vocabulary. Stress, read with appropriate. Structural Analysis Base words, identify Compound words, form and understand Contractions, form and understand Prefixes re- mis- un- Suffixes -ed -en -er -es -est -ful -ier -iest -ing -less -ly -ness -or -s -y Syllables, recognize Study and Research Skills Alphabetical order, use Generate questions for investigation Locate information Notes, take from sources References and resources, use Table of contents, use

Suggested Teacher Read-Alouds. Summarize. Super-Duper mini-magazines. Super Smart Informational Digital Read-Alouds. Superkids’ names, recognize. Superkids Shorts. Syllables, recognize. Synonyms, understand.

T - Teacher Read-Aloud Suggestions. Ten-Minute Tuck-Ins: Activities for Differentiating Instruction Skill extension Skill reinforcement Text features and graphics, understand. Text structure, recognize. Title of a book, story, or poem, identify. Typographical clues, observe.

V - Verbs, understand. Verbs, use to understand time of action. Visualize. Vocabulary Abbreviations, understand Antonyms, understand Categorizing Compound words, understand Context clues, use Descriptive language, appreciate and use Figurative language, appreciate and use Homographs, understand Homonyms, understand Homophones, understand Idioms, understand Informational text words, discuss Interjections Literary text words, discuss Memory Words, understand Multiple-meaning words, understand Onomatopoeia (words for sounds), discuss Rebuses, identify Sequence words, understand Superkids’ names, recognize Synonyms, understand Words to Know

Vowels. Word Families. Writing Composition Products Address on an envelope Answers to questions Books Descriptions Directions (How-to) E-mails Fact cards Facts Friendly letter Illustrations Informative writing Labels Lists Memory Book Messages Notes Opinions Paragraphs Personal narratives Poetry Questions Research Questions Review (play or book) Riddles Sentences Sign Stories (Narratives) Summaries Skills Add details to pictures or writing Contribute ideas to group writing Edit writing Feedback, give Generate ideas before writing Organize ideas in sequential order Partners, work with Plan writing Publish writing Revise writing Select a topic to write about Set a purpose for writing Spelling, use temporary Staying on topic Text Types Informative/explanatory Narrative Opinion Writing Rubrics Descriptive Writing Explanatory Writing Informative and Correspondence Writing Informative Writing Narrative and Opinion Writing Narrative Writing Opinion Writing Poetry Written directions, follow """\
                  "Here is the content from the student workbook to create the index from:\n\n"

    MAX_LLM_INPUT_LENGTH = 900000  # Character limit for the combined prompt and PDF content
    input_text = base_prompt
    current_input_length = len(input_text)
    pdf_content_to_send = ""
    truncated_input = False

    for item in pdf_text_with_pages:
        page_content = f"--- Page {item['page']} ---\n{item['text']}\n\n"
        if current_input_length + len(page_content) > MAX_LLM_INPUT_LENGTH:
            truncated_input = True
            break
        pdf_content_to_send += page_content
        current_input_length += len(page_content)

    input_text += pdf_content_to_send
    if truncated_input:
        input_text += "\n\n--- Content truncated due to length limits for LLM input. Index generated based on available content. ---"

    try:
        messages = [HumanMessage(content=input_text)]
        llm_response = chat_model.invoke(messages)
        return llm_response.content
    except Exception as e:
        if 'st' in globals():
            st.error(f"Error interacting with LLM: {str(e)}")
        else:
            print(f"Error interacting with LLM: {str(e)}")
        return None

# --- 4. PDF Generation from LLM Output ---
def create_pdf_from_text(text_content, output_pdf_path="teacher_guide_index.pdf"):
    """
    Creates a basic PDF document from a given string content.
    The function already processes the LLM's output line by line, which can be seen
    as a form of chunking for presentation.
    """
    try:
        doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Define a custom style for the index content
        index_style = ParagraphStyle(
            'IndexStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,
            leftIndent=0 # Default for main entries
        )

        # A simple way to handle indentation for sub-entries based on leading spaces
        sub_entry_style = ParagraphStyle(
            'SubEntryStyle',
            parent=index_style,
            leftIndent=20 # Indent for sub-entries
        )

        flowables = []
        flowables.append(Paragraph("Teacher's Guide Index Generated by LLM", styles['h1']))
        flowables.append(Spacer(1, 0.2 * 2.54 * 72)) # 0.2 inch spacer

        # Check for LLM self-reported truncation
        llm_truncated_flag = False
        if "--- INDEX INCOMPLETE ---" in text_content:
            llm_truncated_flag = True
            st.warning("The LLM indicated that the index generation was incomplete due to length constraints. The generated PDF will contain the partial index.")
        elif "--- END OF INDEX ---" not in text_content:
            st.warning("The LLM did not include the '--- END OF INDEX ---' marker. The response might have been truncated by the LLM or ended prematurely.")

        # Split the text content into lines and apply basic styling
        for line in text_content.split('\n'):
            # Stop processing if we hit the LLM's explicit end marker
            if line.strip() == "--- END OF INDEX ---":
                break
            if line.strip() == "":
                flowables.append(Spacer(1, 0.1 * 2.54 * 72)) # Small spacer for empty lines
            elif line.startswith("   - ") or line.startswith("     "): # Simple check for indentation
                # Adjust for potentially malformed indentation if LLM is inconsistent
                clean_line = line.lstrip(' ').lstrip('- ').strip()
                flowables.append(Paragraph(clean_line, sub_entry_style))
            else:
                flowables.append(Paragraph(line.strip(), index_style))

        doc.build(flowables)
        if 'st' in globals():
            st.success(f"Successfully created PDF: {output_pdf_path}")
            if llm_truncated_flag:
                st.info("Please review the generated PDF for completeness, as the LLM reported an incomplete index.")
        else:
            print(f"Successfully created PDF: {output_pdf_path}")
            if llm_truncated_flag:
                print("Please review the generated PDF for completeness, as the LLM reported an incomplete index.")
        return output_pdf_path
    except Exception as e:
        if 'st' in globals():
            st.error(f"Error creating PDF: {str(e)}")
        else:
            print(f"Error creating PDF: {str(e)}")
        return None

# --- 5. Main Execution Flow (Streamlit App or standalone script) ---

# --- For Streamlit App ---
def run_streamlit_app():
    st.title("PDF to LLM-Generated Teacher Guide Index")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file to a temporary path
        temp_pdf_path = os.path.join("temp_uploaded_pdf.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write(f"Processing '{uploaded_file.name}'...")

        # Initialize LLM
        chat_model = initialize_chat_model()

        if chat_model:
            st.info("Extracting text from PDF...")
            pdf_text_data = extract_text_with_page_numbers(temp_pdf_path)

            if pdf_text_data:
                st.success("Text extracted. Sending to LLM...")

                llm_index_content = generate_teacher_guide_index_with_llm(chat_model, pdf_text_data)

                if llm_index_content:
                    st.subheader("Generated Index Content (from LLM):")
                    # Displaying a very long code block in Streamlit can be slow.
                    # You might consider showing only the first N lines or a scrollable area.
                    st.code(llm_index_content[:5000] + "...\n(truncated for display, download PDF for full content)" if len(llm_index_content) > 5000 else llm_index_content, language='text')

                    # 3. Create output PDF
                    output_pdf_filename = f"teacher_guide_index_{uploaded_file.name.replace('.pdf', '')}.pdf"
                    created_pdf_path = create_pdf_from_text(llm_index_content, output_pdf_filename)

                    if created_pdf_path and os.path.exists(created_pdf_path):
                        with open(created_pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download Generated Teacher Guide Index PDF",
                                data=pdf_file,
                                file_name=output_pdf_filename,
                                mime="application/pdf"
                            )
                        os.remove(created_pdf_path) # Clean up generated PDF
                else:
                    st.error("LLM failed to generate index content.")
            else:
                st.error("Failed to extract text from PDF.")

        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path) # Clean up temporary uploaded PDF


if __name__ == "__main__":
    run_streamlit_app()