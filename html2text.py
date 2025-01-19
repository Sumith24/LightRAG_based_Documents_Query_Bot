from bs4 import BeautifulSoup
import os
import re

# html_string = ""

# for file in os.listdir('dummy_data'):
#     with open(os.path.join(os.getcwd(), f"dummy_data\\{file}"), "r", encoding="utf-8") as doc_f:
#         h_text = doc_f.read()
#         html_string = html_string + '\n'+ h_text

# # clean_text = ' '.join(BeautifulSoup(html_string, "html.parser").stripped_strings)

# text = '\n'.join(BeautifulSoup(html_string, "html.parser").findAll(string=True))

# # print(clean_text)

# print(text)



def extract_html_content_to_text(html_file_path, output_txt_path):
    """
    Reads text and table data from an HTML file and saves it to a .txt file in a structured format.

    Args:
        html_file_path (str): Path to the input HTML file.
        output_txt_path (str): Path to save the extracted content as a .txt file.
    """
    def clean_text(text):
        """
        Removes continuous dots from text (e.g., '................ label1' -> 'label1').
        """
        return re.sub(r'^\.*\s*', '', text)  # Remove leading dots and spaces

    with open(html_file_path, "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        # Extract all text (ignoring tags) and write it
        txt_file.write("Extracted Text Content:\n")
        txt_file.write("=" * 50 + "\n")
        txt_file.write(soup.get_text(separator="\n",strip=True) + "\n\n")

        # Extract all tables
        tables = soup.find_all("table")
        for i, table in enumerate(tables, start=1):
            txt_file.write(f"Table {i}:\n")
            txt_file.write("=" * 50 + "\n")
            # import pdb;pdb.set_trace();
            # Extract rows
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["th", "td"])  # Extract both header and data cells
                cell_text = [clean_text(cell.get_text(separator=" ",strip=True)) for cell in cells]
                txt_file.write("\t\t".join(cell_text) + "\n")  # Tab-separated format


            txt_file.write("\n")  # Add a newline between tables

    print(f"HTML content has been successfully extracted to {output_txt_path}")


h_file = r"dummy_data/EXP530.1-84.71 - R530.1 TCU2 PRs- Build2.html"
t_file = r"dummy_data/EXP530.1-84.70_Build2_PRs.txt"

extract_html_content_to_text(h_file, t_file)
