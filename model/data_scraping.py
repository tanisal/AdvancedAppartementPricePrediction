from bs4 import BeautifulSoup
import numpy
import pandas as pd
from selenium import webdriver

# Giving path of the chrome driver and setting a selenium driver
chrome_driver = r"C:\chromedriver\chromedriver"
driver = webdriver.Chrome(chrome_driver)


def all_pages_list_of_ads(pages):
    """Function to extract the links of all the ads for the given number of pages"""
    # WE go to the first page
    driver.get(r"https://www.imot.bg/pcgi/imot.cgi?act=3&slink=82wfp9&f1=1")
    # Click on the go on page pop menu
    driver.find_element_by_xpath(
        "/html/body/div[5]/div[2]/div[1]/div[2]/div[2]/button[1]/p"
    ).click()
    # Create an empty list to store all the links from the loop
    all_links = []
    # traverse through all the links in a page, append the links in a list and then go on a another page
    for page in range(1, pages + 1):
        url = r"https://www.imot.bg/pcgi/imot.cgi?act=3&slink=82wfp9&f1={}".format(page)
        driver.get(url)
        # Creating a soup object with the page source
        soup = BeautifulSoup(driver.page_source, "html.parser")
        # Finding all the needed links on the page
        links = soup.find_all("a", {"class": "photoLink"})
        # Traversing through the first page and storing everything in the main list
        for row in links[2:]:
            all_links.append(row["href"])
    return all_links


def html_data(url):
    """Function to get the html data for every single ad, returns soup object"""

    driver.get(f"https:{url}")
    soup = BeautifulSoup(driver.page_source, "html.parser")
    return soup


def get_price(soup):
    """Function returning the price foe the given soup object as an argument"""
    try:
        # Looking for cena in the html
        return soup.find("div", id="cena").text.strip()
    except:
        return numpy.nan


def sqrm_price(soup):
    """Function returning the square meter per price value for the soup object given"""
    try:
        return soup.find("span", id="cenakv").text
    except:
        return numpy.nan


def get_sqrm(soup):
    """Function returning square meter value for the soup object given"""
    try:
        adParams = soup.find("div", class_="adParams")
        div = list(adParams.children)[1]
        strong = list(div.children)[1]
        return strong.text
    except:
        return numpy.nan


def get_floor(soup):
    """Function for the floor value for the soup object given"""
    try:
        adParams = soup.find("div", class_="adParams")
        div = list(adParams.children)[3]
        strong = list(div.children)[1]
        return strong.text
    except:
        return numpy.nan


def get_build(soup):
    """Function for the build method for the soup object given"""
    try:
        adParams = soup.find("div", class_="adParams")
        div = list(adParams.children)[4]
        return div.text
    except:
        return numpy.nan


def get_title(soup):
    """Function returning the details for the soup object given"""
    try:
        return soup.find("div", class_="title").text
    except:
        return numpy.nan


def get_location(soup):
    """Function returning the locations for the soup object given"""
    try:
        return soup.find("div", class_="location").text
    except:
        return numpy.nan


# Assigning the number of pages for the links for soup objects
app_links = all_pages_list_of_ads(25)


def get_house_data(app_links):
    """Function to combine all the info we need, with the defined functions above"""
    # We create an empty list for the info we are gathering
    appartament_data = []
    # Loop through all the ads pages
    for url in app_links:
        soup = html_data(url)
        price = get_price(soup)
        sqrm_cena = sqrm_price(soup)
        title = get_title(soup)
        sqrm = get_sqrm(soup)
        floor = get_floor(soup)
        build = get_build(soup)
        location = get_location(soup)
        appartament_data.append([location, price, sqrm, sqrm_cena, title, floor, build])
    return appartament_data


# Create data frame with all the gathered data
df = pd.DataFrame(get_house_data(app_links))
# Naming the columns
df.columns = ["location", "price", "m2", "price/m2", "details", "floor", "build"]

# Saving everythng into csv file
df.to_csv(r"imotibg_varna.csv", index=False)
print("Created CSV File")
