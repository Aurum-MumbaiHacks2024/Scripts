

import requests
import xml.etree.ElementTree as ET
import csv

# Step 1: Fetch the XML from the URL
url = 'https://portal.amfiindia.com/RssNAV.aspx?mf=71'  # Replace with your actual URL
response = requests.get(url)
xml_content = response.content

# Step 2: Parse the XML content
root = ET.fromstring(xml_content)

# Step 3: Extract relevant data
data = []
for item in root.findall('.//item'):
    title = item.find('title').text
    description = item.find('description').text
    pubDate = item.find('pubDate').text

    # Extract NAV and Date from description
    nav = None
    date = None
    if description:
        # Parse the HTML table in the description
        desc_root = ET.fromstring(f"<root>{description}</root>")
        nav_element = desc_root.find('.//b[. = " NAV "]/following-sibling::text()[1]')
        date_element = desc_root.find('.//b[. = " Date "]/following-sibling::text()[1]')
        nav = nav_element.strip() if nav_element is not None else None
        date = date_element.strip() if date_element is not None else None

    # Append the data
    data.append({
        'Title': title,
        'NAV': nav,
        'Date': date,
        'Publication Date': pubDate,
    })

# Step 4: Write the data to a CSV file
csv_file = 'mutual_fund_data.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Title', 'NAV', 'Date', 'Publication Date'])
    writer.writeheader()
    writer.writerows(data)

print(f"Data has been written to {csv_file}")

import requests
import xml.etree.ElementTree as ET
import csv
import html

# Step 1: Fetch the XML from the URL
url = 'https://portal.amfiindia.com/RssNAV.aspx?mf=71'  # Replace with your actual URL
response = requests.get(url)
xml_content = response.content

# Step 2: Parse the XML content
root = ET.fromstring(xml_content)

# Step 3: Extract relevant data
data = []
for item in root.findall('.//item'):
    title = item.find('title').text
    description = item.find('description').text
    pubDate = item.find('pubDate').text

    # Extract NAV and Date from description
    nav = None
    date = None
    if description:
        # Escape HTML entities before parsing
        description = html.unescape(description)

        # Parse the HTML table in the description
        try:
            desc_root = ET.fromstring(f"<root>{description}</root>")
            nav_element = desc_root.find('.//b[. = " NAV "]/

# Install necessary libraries
!pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL to scrape
url = "https://portal.amfiindia.com/RssNAV.aspx?mf=71"

# Set headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
response = requests.get(url, headers=headers)

# Parse the XML content
soup = BeautifulSoup(response.content, 'xml')

# Initialize lists to store scraped data
names, categories, navs = [], [], []

# Extract and print each item for inspection
for item in soup.find_all('item'):
    try:
        name = item.find('AmfiName').text if item.find('AmfiName') else "N/A"
        category = item.find('Category').text if item.find('Category') else "N/A"
        nav = float(item.find('Nav').text.strip()) if item.find('Nav') else None

        if nav and nav >= 15:
            names.append(name)
            categories.append(category)
            navs.append(nav)
    except Exception as e:
        print(f"Error: {e}")

# Create DataFrame
df = pd.DataFrame({'Name': names, 'Category': categories, 'NAV': navs})

# Check if DataFrame is populated
print(df)

# Save to CSV
df.to_csv('Mutual_Fund_NAVs.csv', index=False)

from google.colab import files
files.download('Mutual_Fund_NAVs.csv')

# Install required libraries
!pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL to fetch all RSS feeds
rss_feed_url = "https://www.amfiindia.com/rss-feeds"

# Set headers to avoid blocking by the server
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Send a request to get the main RSS feed page
response = requests.get(rss_feed_url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract all individual RSS feed URLs from the main page
rss_links = [a['href'] for a in soup.find_all('a', href=True) if 'RssNAV.aspx' in a['href']]

print(f"Found {len(rss_links)} RSS links")  # Debugging step to confirm RSS links

# Initialize lists to store the scraped data
data = []

# Loop through each RSS feed link and extract data
for link in rss_links:
    print(f"Scraping {link}...")
    rss_response = requests.get(link, headers=headers)
    rss_soup = BeautifulSoup(rss_response.content, 'xml')

    # Loop through all <item> tags and extract relevant data
    for item in rss_soup.find_all('item'):
        try:
            name = item.find('AmfiName').text if item.find('AmfiName') else "N/A"
            category = item.find('Category').text if item.find('Category') else "N/A"
            nav = float(item.find('Nav').text.strip()) if item.find('Nav') else None

            # Store the data only if NAV is available and >= 15
            if nav and nav >= 15:
                data.append([name, category, nav])
        except Exception as e:
            print(f"Error processing item: {e}")

# Create a DataFrame from the scraped data
df = pd.DataFrame(data, columns=['Name', 'Category', 'NAV'])

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('Mutual_Fund_NAVs.csv', index=False)

# Download the CSV in Colab (if running on Colab)
from google.colab import files
files.download('Mutual_Fund_NAVs.csv')

import requests
from bs4 import BeautifulSoup
import pandas as pd

rss_feed_url = "https://portal.amfiindia.com/RssNAV.aspx?mf=62"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Fetch the main page and extract RSS links
response = requests.get(rss_feed_url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
rss_links = [a['href'] for a in soup.find_all('a', href=True) if 'RssNAV.aspx' in a['href']]

# Initialize list to collect data
data = []

# Loop through RSS feeds and extract data
for link in rss_links:
    print(f"Scraping {link}...")
    rss_response = requests.get(link, headers=headers)
    rss_soup = BeautifulSoup(rss_response.content, 'xml')

    # Debug: Print the structure of the RSS feed
    print(rss_soup.prettify())  # Add this to inspect the structure

    for item in rss_soup.find_all('item'):
        try:
            # Update these tags based on the actual XML structure
            name = item.find('schemeName').text if item.find('schemeName') else "N/A"
            category = item.find('category').text if item.find('category') else "N/A"
            nav = float(item.find('nav').text.strip()) if item.find('nav') else None

            if nav and nav >= 15:
                data.append([name, category, nav])
        except Exception as e:
            print(f"Error processing item: {e}")

# Create DataFrame from the collected data
df = pd.DataFrame(data, columns=['Name', 'Category', 'NAV'])

print(df)

# Save to CSV if data is not empty
if not df.empty:
    df.to_csv('Mutual_Fund_NAVs.csv', index=False)
else:
    print("No data found!")

import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the RSS feed
url = "https://portal.amfiindia.com/RssNAV.aspx?mf=62"

# Send a request to the URL and get the response
response = requests.get(url)
rss_content = response.content

# Parse the XML content
soup = BeautifulSoup(rss_content, 'xml')

# Initialize a list to collect data
data = []

# Iterate over each item in the RSS feed
for item in soup.find_all('item'):
    try:
        # Extract the required fields
        name = item.find('AmfiName').text if item.find('AmfiName') else "N/A"
        category = item.find('Category').text if item.find('Category') else "N/A"
        nav = float(item.find('NAV').text.strip()) if item.find('NAV') else None

        # Skip funds with NAV less than 15
        if nav is not None and nav >= 15:
            data.append([name, category, nav])
    except Exception as e:
        print(f"Error processing item: {e}")

# Create a DataFrame from the collected data
df = pd.DataFrame(data, columns=['Name', 'Category', 'NAV'])

# Display the DataFrame
print(df)

# Save to CSV if data is not empty
if not df.empty:
    df.to_csv('Mutual_Fund_NAVs.csv', index=False)
else:
    print("No data found!")

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL to scrape
url = "https://portal.amfiindia.com/RssNAV.aspx?mf=62"

# Fetch the page content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'xml')  # Parse as XML

# Initialize a list to hold the extracted data
data = []

# Find all item elements in the RSS feed
for item in soup.find_all('item'):
    try:
        # Extract the required fields
        name = item.find('AmfiName').text.strip() if item.find('AmfiName') else "N/A"
        category = item.find('Category').text.strip() if item.find('Category') else "N/A"
        nav = item.find('NAV').text.strip() if item.find('NAV') else None

        # Append to data if NAV is available
        if nav:
            data.append([name, category, nav])
    except Exception as e:
        print(f"Error processing item: {e}")

# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=['Name', 'Category', 'NAV'])

# Save to CSV
df.to_csv('Mutual_Fund_NAVs.csv', index=False)

# Print the DataFrame for verification
print(df)

import requests
import csv
from bs4 import BeautifulSoup
from datetime import datetime

def scrape_amfi_nav():
    """
    Scrapes NAV data from AMFI portal and saves it to CSV
    """
    for i in range(1, 80):
        url = f"https://portal.amfiindia.com/RssNAV.aspx?mf={i}"

        try:
            # Send GET request to the URL
            response = requests.get(url)
            response.raise_for_status()

            # Parse the XML content
            soup = BeautifulSoup(response.content, 'xml')

            # Create/open CSV file
            with open('amfi_nav_data.csv', 'a', newline='') as file:  # Changed 'w' to 'a'
                writer = csv.writer(file)

                # Write header
                writer.writerow(['Name', 'Category', 'NAV'])

                # Find all item elements
                items = soup.find_all('item')

                for item in items:
                    # Extract required fields
                    description = item.description.text

                    # Parse the description to get category and NAV
                    desc_lines = description.split('\n')

                    name = item.title.text.strip()
                    category = ''
                    nav = ''

                    for line in desc_lines:
                        if 'Category' in line:
                            category = line.replace('Category', '').strip()
                        elif 'NAV' in line:
                            nav = line.replace('NAV', '').strip()

                    # Write to CSV
                    writer.writerow([name, category, nav])

            print("Data has been successfully scraped and saved to amfi_nav_data.csv")

        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
        except Exception as e:
            print(f"Error processing data: {e}")

if __name__ == "__main__":
    scrape_amfi_nav()

import requests
import csv
from bs4 import BeautifulSoup

def scrape_amfi_nav():
    """
    Scrapes NAV data from AMFI portal and saves it to CSV
    """
    url_template = "https://portal.amfiindia.com/RssNAV.aspx?mf={}"

    # Open CSV file in append mode
    with open('amfi_nav_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['Name', 'Category', 'NAV'])

        for i in range(62, 77):
            url = url_template.format(i)

            try:
                # Send GET request to the URL
                response = requests.get(url)
                response.raise_for_status()

                # Parse the XML content
                soup = BeautifulSoup(response.content, 'xml')

                # Find all item elements
                items = soup.find_all('item')

                for item in items:
                    # Extract required fields
                    name = item.title.text.strip()
                    description = item.description.text
                    desc_lines = description.split('\n')

                    category = ''
                    nav = ''

                    for line in desc_lines:
                        if 'Category' in line:
                            category = line.replace('Category', '').strip()
                        elif 'NAV' in line:
                            nav = line.replace('NAV', '').strip()

                    # Write to CSV
                    writer.writerow([name, category, nav])

            except requests.RequestException as e:
                print(f"Error fetching data for mf={i}: {e}")
            except Exception as e:
                print(f"Error processing data for mf={i}: {e}")

    print("Data has been successfully scraped and saved to amfi_nav_data.csv")

if __name__ == "__main__":
    scrape_amfi_nav()

import requests
import csv
from bs4 import BeautifulSoup

def scrape_amfi_nav():
    # Open CSV file once outside the loop in append mode
    with open('amfi_nav_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header once at the beginning
        writer.writerow(['MF ID', 'Name', 'Category', 'NAV'])

        for i in range(1, 6):
            """
            Scrapes NAV data from AMFI portal and saves it to a single CSV file.
            """
            url = f"https://portal.amfiindia.com/RssNAV.aspx?mf={i}"

            try:
                # Send GET request to the URL with a timeout of 20 seconds
                response = requests.get(url, timeout=20)
                response.raise_for_status()

                # Parse the XML content
                soup = BeautifulSoup(response.content, 'xml')

                # Find all item elements
                items = soup.find_all('item')

                for item in items:
                    # Extract required fields
                    name = item.title.text.strip()
                    description = item.description.text
                    category = ''
                    nav = ''

                    # Filter relevant lines from the description
                    for line in description.split('\n'):
                        if 'Category' in line:
                            category = line.replace('Category:', '').strip()
                        elif 'NAV' in line:
                            nav = line.replace('NAV:', '').strip()

                    # Write data for each item to the CSV
                    writer.writerow([i, name, category, nav])

                print(f"Data for MF ID {i} has been successfully scraped and saved.")

            except requests.Timeout:
                print(f"Skipping MF ID {i}: Page took too long to load.")
            except requests.RequestException as e:
                print(f"Error fetching data for MF ID {i}: {e}")
            except Exception as e:
                print(f"Error processing data for MF ID {i}: {e}")

if __name__ == "__main__":
    scrape_amfi_nav()

import requests
import csv
from bs4 import BeautifulSoup

def scrape_amfi_nav():
    """
    Scrapes NAV data from AMFI portal and saves it to CSV
    """
    url_template = "https://portal.amfiindia.com/RssNAV.aspx?mf={}"

    # Open CSV file in append mode
    with open('amfi_nav_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['Name', 'Category', 'NAV', 'Date'])

        for i in range(1, 80):
            url = url_template.format(i)

            try:
                # Send GET request to the URL
                response = requests.get(url)
                response.raise_for_status()

                # Parse the XML content
                soup = BeautifulSoup(response.content, 'xml')

                # Find all item elements
                items = soup.find_all('item')

                for item in items:
                    # Extract fields directly from XML
                    name = item.title.text.strip()
                    category = item.find('category').text.strip() if item.find('category') else 'N/A'
                    nav = item.find('nav').text.strip() if item.find('nav') else 'N/A'
                    date = item.pubDate.text.strip() if item.pubDate else 'N/A'

                    # Write to CSV
                    writer.writerow([name, category, nav, date])

            except requests.RequestException as e:
                print(f"Error fetching data for mf={i}: {e}")
            except Exception as e:
                print(f"Error processing data for mf={i}: {e}")

    print("Data has been successfully scraped and saved to amfi_nav_data.csv")

if __name__ == "__main__":
    scrape_amfi_nav()

import requests
import csv
from bs4 import BeautifulSoup

def scrape_amfi_nav():
    """
    Scrapes NAV data from AMFI portal and saves it to CSV
    """
    url_template = "https://portal.amfiindia.com/RssNAV.aspx?mf={}"

    # Open CSV file in append mode
    with open('amfi_nav_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['Name', 'Category', 'NAV', 'Date'])

        for i in range(1, 80):
            url = url_template.format(i)

            try:
                # Send GET request to the URL
                response = requests.get(url)
                response.raise_for_status()

                # Parse the XML content
                soup = BeautifulSoup(response.content, 'xml')

                # Find all item elements
                items = soup.find_all('item')

                for item in items:
                    # Extract fields directly from XML
                    name = item.title.text.strip()
                    date = item.pubDate.text.strip() if item.pubDate else 'N/A'

                    # Parse the description content as HTML to extract Category and NAV
                    description_html = BeautifulSoup(item.description.text, 'html.parser')
                    rows = description_html.find_all('tr')

                    # Initialize variables
                    category = 'N/A'
                    nav = 'N/A'

                    # Extract category and NAV from the rows
                    for row in rows:
                        header = row.find('b').text.strip() if row.find('b') else ''
                        value = row.find_all('td')[1].text.strip() if len(row.find_all('td')) > 1 else ''

                        if 'Category' in header:
                            category = value
                        elif 'NAV' in header:
                            nav = value

                    # Write to CSV
                    writer.writerow([name, category, nav, date])

            except requests.RequestException as e:
                print(f"Error fetching data for mf={i}: {e}")
            except Exception as e:
                print(f"Error processing data for mf={i}: {e}")

    print("Data has been successfully scraped and saved to amfi_nav_data.csv")

if __name__ == "__main__":
    scrape_amfi_nav()

