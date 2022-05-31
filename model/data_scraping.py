from bs4 import BeautifulSoup
import numpy
import pandas as pd
from selenium import webdriver

#Giving path of the chrome driver and setting a selenium driver
chrome_driver = r'C:\chromedriver\chromedriver'
driver = webdriver.Chrome(chrome_driver)

#Function to extract the links of all the ads for the given number of pages
def all_pages_list_of_ads(pages):
    #WE go to the first page
    driver.get(r'https://www.imot.bg/pcgi/imot.cgi?act=3&slink=82sgje&f1=1')
    #Click on the go on page pop menu
    driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[1]/div[2]/div[2]/button[1]/p').click()
    #Create an empty list to store all the links from the loop
    all_links=[]
    #traverse through all the links n a page, append the links in a list and then go on a another page
    for page in range(1,pages+1):
        url =r'https://www.imot.bg/pcgi/imot.cgi?act=3&slink=82sgje&f1={}'.format(page)
        driver.get(url)
        #Creating a soup object with the page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        #Finding all the needed links on the page 
        links=soup.find_all('a',{'class':'photoLink'})
        #Traversing through the first page and storing everything in the main list
        for row in links[2:]:
            all_links.append(row['href'])
    return all_links




#Function to get the html data for every single ad
def html_data(url):
    driver.get(f'https:{url}')
    soup = BeautifulSoup(driver.page_source,'html.parser')
    return soup

def get_price(soup):
    try:
        return soup.find('div', id='cena').text.strip()
    except:
        return numpy.nan

def sqrm_price(soup):
    try:
        return soup.find('span',id='cenakv').text
    except:        return numpy.nan

def get_sqrm(soup):
    try:
        adParams=soup.find('div',class_='adParams')
        div=list(adParams.children)[1]
        strong=list(div.children)[1]
        return strong.text
    except:
        return numpy.nan


def get_floor(soup):
    try:
        adParams=soup.find('div',class_='adParams')
        div=list(adParams.children)[3]
        strong=list(div.children)[1]
        return strong.text
    except:
        return numpy.nan


def get_build(soup):
    try:
        adParams=soup.find('div',class_='adParams')
        div=list(adParams.children)[4]
        return div.text
    except:
        return numpy.nan


def get_title(soup):
    try:
        return soup.find('div',class_='title').text
    except:
        return numpy.nan

def get_location(soup):
    try:
        return soup.find('div',class_='location').text
    except:
        return numpy.nan

app_links= all_pages_list_of_ads(25)

def get_house_data(app_links):
    appartament_data=[]
    for url in app_links:
        soup = html_data(url)
        price= get_price(soup)
        sqrm_cena=sqrm_price(soup)
        title=get_title(soup)
        sqrm=get_sqrm(soup)
        floor=get_floor(soup)
        build=get_build(soup)
        location=  get_location(soup)
        appartament_data.append([location,price,sqrm,sqrm_cena,title,floor,build])
    return appartament_data



df =pd.DataFrame(get_house_data(app_links))

df.columns=['location','price','m2','price/m2','details','floor','build']


df.to_csv(r"imotibg_Varna.csv",index=False)
print("Created CSV File")

