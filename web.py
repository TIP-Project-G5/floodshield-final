from selenium import webdriver
from bs4 import BeautifulSoup

# URL of the FAQ page
url = "https://www.swinburneonline.edu.au/faqs/"

# Initialize a Selenium web driver (make sure you have the appropriate driver installed)
driver = webdriver.Chrome()  # You can use other drivers like Firefox or Edge

# Open the web page
driver.get(url)

# Wait for the page to load (you might need to adjust the wait time)
driver.implicitly_wait(10)

# Get the page source after it has loaded
html = driver.page_source

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# Find all the FAQ items
faq_items = soup.find_all('div', class_='accordion faqs-group')

# Loop through each FAQ item and extract the questions and answers
for faq_item in faq_items:
    question = faq_item.find('div', class_='red pr-2').next_sibling.strip()
    answer = faq_item.find('div', class_='content').text.strip()
    print("Q:", question)
    print("A:", answer)
    print("\n")

# Close the Selenium web driver
driver.quit()
