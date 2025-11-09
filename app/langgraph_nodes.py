from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, List
import pandas as pd 
from app.profiler import parse_excel, basic_summary, profile_to_json
from app.llm_client import client
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from io import BytesIO
import ast
import os

BASE_DIR = os.path.dirname(__file__) 
REPORT_DIR = os.path.join(BASE_DIR, "generated_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# defining states
class SheetState(TypedDict):
    sheet_name: str
    summary: Dict[str,Any]
    profile: Dict[str,Any]
    df: pd.DataFrame
    insights:str
    visuals: Dict[str,Dict[str,Any]]
    pdf_path:str

class DataProfileState(TypedDict):
    filepath:str
    sheets:List[SheetState]

# defining graph
graph = StateGraph(DataProfileState)


def get_data_profile(state: DataProfileState) -> DataProfileState:
    filepath = state['filepath']
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding="utf-8")
        sheets = {"Sheet1":df}
    else:
        sheets = parse_excel(filepath)
    
    sheet_states: List[SheetState] = []
    for sheet_name, df in sheets.items():
        summary = basic_summary(df)
        profile  = profile_to_json(df)
        sheet_states.append({
            "sheet_name": sheet_name,
            "summary": summary,
            "profile": profile,
            "df": df
        })
    state['sheets'] = sheet_states

    return state


def generate_insights(state: DataProfileState) -> DataProfileState:

    # Prepare prompt content
    updated_sheets:List[SheetState] = []
    for sheet in state["sheets"]:
        
        sheet_name = sheet["sheet_name"]
        summary = sheet["summary"]
        profile = sheet["profile"]


        system_prompt = {
            "role": "system",
            "content": """
            You are a business insights agent. Your role is to generate clear, actionable, and relevant textual insights based on structured data.

            You have been provided with two key inputs:

            A basic summary of an Excel file, including row/column counts, data types, missing values, unique values, and sample entries.
            A data profile description, which includes detailed statistical and structural metadata about the Excel file.

            Your task is to analyze this information and generate at most 4 applicable business insights that can be inferred from the data.
            These insights should reflect patterns, anomalies, opportunities, risks, or strategic observations that would be useful to a business decision-maker.

            Each insight must be:

            Grounded in the data
            Clearly stated and context-aware
            Framed as a meaningful takeaway
            Suitable for visualization — meaning that each insight should be expressed in a way that allows a graph, chart, or dashboard element to be created from it later (e.g., trends, comparisons, distributions, correlations, rankings, outliers).

            Avoid generic statements. Focus on clarity, relevance, and impact.
            """
        }

        user_prompt = {
            "role": "user",
            "content": f"Sheet:{sheet_name}\n Summary:{summary}\n Profile:{profile}"
        }

        response = client.chat.completions.create(
            messages=[system_prompt, user_prompt],
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model="gpt-4o"
        )

        sheet['insights'] = response.choices[0].message.content
        updated_sheets.append(sheet)
        state["sheets"] = updated_sheets

    return state



def suggest_plots(state: DataProfileState) -> DataProfileState:

    updated_sheets:List[SheetState] = []
    for sheet in state['sheets']:

        user_prompt = {
            'role': 'user',
            'content': f"Business Insights: {sheet['insights']}"
        }

        df = sheet['df']
        df_columns = list(df.columns)

        system_prompt = {
            'role': 'system',
            'content': f"""
                    You are a data visualization assistant. Based on the business insights provided to you, your task is to generate at most 4 chart or plot suggestions that can help visualize those insights.

                    Your output must strictly be a JSON object structured as follows:

                    {{
                    "chart1": {{
                        "plot": "matplotlib code as a string",
                        "description": "A short explanation of what the chart reveals."
                    }},
                    "chart2": {{
                        "plot": "...",
                        "description": "..."
                    }}
                    }}

                    Requirements:
                    - Each chart must be based on a specific insight.
                    - Use diverse chart types (e.g., bar, line, pie, scatter, histogram, box plot, heatmap, etc.).
                    - The "plot" field must contain valid Python matplotlib code as a string that can be executed to generate the chart.
                    - The "description" field should briefly explain what the chart shows and why it’s useful.
                    - Use the following DataFrame: {df}
                    - Use only the column names of the DataFrame: {df_columns}
                    - Do not invent or assume any other columns.
                    - Do not include placeholder data — assume the data is already loaded in df.
                    - Focus on clarity, variety, and relevance to the insights.
                    - Always assign the figure to a variable using fig = plt.figure() and plot on that figure. Do not rely on implicit figure creation.
                    """
                        }
        response = client.chat.completions.create(
            messages=[system_prompt, user_prompt],
            max_tokens=4096,
            temperature=0,
            top_p=1.0,
            model="gpt-4o"
        )
        raw = response.choices[0].message.content

        # Remove Markdown code block markers
        if raw.startswith("```json"):
            raw = raw[len("```json"):].strip()
        if raw.endswith("```"):
            raw = raw[:-len("```")].strip()

        # Now safely parse
        sheet['visuals']= ast.literal_eval(raw)
        updated_sheets.append(sheet)
        state["sheets"] = updated_sheets

    return state

def make_plots(state: DataProfileState) -> DataProfileState:

    updated_sheets:List[SheetState] = []
    for sheet in state['sheets']:
        df = sheet["df"]
        images_with_descriptions = []
        for chart_name, chart in sheet["visuals"].items():
            plot_code = chart.get("plot","").replace("plt.show()", "")
            description = chart.get("description","")

                        # Execute the matplotlib code
            local_scope = {"df": df, "plt": plt, "pd": pd}
            try:
                exec(plot_code, {}, local_scope)
            except Exception as e:
                print(f"Error executing plot code for {chart_name}: {e}")
                continue


            # Get the most recent figure created by the executed code
            figs = [plt.figure(n) for n in plt.get_fignums()]
            if figs:
                fig = figs[-1]
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                images_with_descriptions.append((buf, description))
                plt.close(fig)
            else:
                print(f"No figure generated for {chart_name} in {sheet['sheet_name']}")
        
        pdf_filename = os.path.join(REPORT_DIR, f"{sheet['sheet_name']}.pdf")

        c = canvas.Canvas(pdf_filename,pagesize=A4)
        width, height = A4

        for img_buf, description in images_with_descriptions:
            img = ImageReader(img_buf)
            img_width = 5.5 * inch
            img_height = 4.5 * inch
            x = (width - img_width) / 2
            y = height - img_height - 100

            # Draw image
            c.drawImage(img, x, y, width=img_width, height=img_height, preserveAspectRatio=True)

            # Draw description below image
            text_x = 50
            text_y = y - 40
            c.setFont("Helvetica", 11)
            c.drawString(text_x, text_y, description)

            c.showPage()

        c.save()
        sheet["pdf_path"] = os.path.basename(pdf_filename)
        updated_sheets.append(sheet)
        state["sheets"] = updated_sheets

    return state


