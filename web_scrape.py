# BeautifulSoup web scraping to download images to local


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

# Directory to save images to
out_dir = 'input_images'

# Set up the Selenium driver
driver = webdriver.Chrome()

# Load the page with the dynamic content
url = 'https://www.gap.com/browse/product.do?pid=449909012&cid=15043&pcid=15043&vid=1#pdp-page-content'
driver.get(url)

# Wait for the page to load and the dynamic content to be generated
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'pdp-main-image-container')))

# Extract the URL of the main image on the page
main_image_url = driver.find_element_by_css_selector('.pdp-main-image-container img').get_attribute('src')

# Extract the URLs of all the other images on the page
other_image_urls = [img.get_attribute('src') for img in driver.find_elements_by_css_selector('.pdp-thumbnail-item img')]

# Combine the URLs into a single list
image_urls = [main_image_url] + other_image_urls

# Create a directory to store the downloaded images
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Download each image and save it to the 'input_images' directory
for url in image_urls:
    filename = os.path.join(out_dir, url.split('/')[-1])
    with open(filename, 'wb') as f:
        f.write(requests.get(url).content)

# Quit the Selenium driver
driver.quit()












# import requests
# from bs4 import BeautifulSoup
# import os




# # Send an HTTP request to the website and get the HTML response
# url = 'https://www.gap.com/browse/product.do?pid=449909012&cid=15043&pcid=15043&vid=1#pdp-page-content'
# response = requests.get(url)

# # Parse the HTML response using BeautifulSoup
# soup = BeautifulSoup(response.content, 'html.parser')

# # Directory to save images to
# out_dir = 'input_images'

# # Extract all the image URLs on the page
# image_urls = []
# for img in soup.find_all('img'):
#     #print('dumping everything:', img, dir(img))

#     image_url = img['src']
#     print(image_url)
#     # Prepend the base URL to the relative image URL
#     if not image_url.startswith('http'):
#         image_url = f"{url}/{image_url}"
#     image_urls.append(image_url)

# # Create a directory to store the downloaded images
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)

# # Download each image and save it to the 'images' directory
# for url in image_urls:
#     filename = os.path.join(out_dir, url.split('/')[-1])

#     # Before saving here, I may want to filter to try to only get relevant images. Somehow, the "main" image on the site


#     with open(filename, 'wb') as f:
#         f.write(requests.get(url).content)