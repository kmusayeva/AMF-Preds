from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service



driver_path = '/user/kmusayev/home/chromedriver'

driver = webdriver.Chrome()

url = 'https://www.globalamfungi.com/'

driver.get(url)

driver.switch_to.frame(driver.find_element("id", "shinyframe"))

element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Taxon search']")))

element.click()

element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//a[text()='Species']")))

element.click()


dropdown = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "id_search-search_key_species-selectized")))

dropdown.click()

# Wait for the dropdown options to appear
options = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "#pymdqzbs8z .option")))


for option in options:
    # Click on the dropdown option
    option.click()
    
    # Click on the "Search" button
    search_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "id_search-buttSearch")))
    search_button.click()
    
    # Click on the "Metadata" tab
    metadata_tab = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "ui-tab-808")))
    metadata_tab.click()
    
    # Click on the "Download metadata" link
    download_link = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "id_results-results_samples-downloadData")))
    download_link.click()




driver.switch_to.default_content()


