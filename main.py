from csv import writer
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd


df =pd.DataFrame()


result,app_no=[],1

for page in range(1,6):
    headers={
            "accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "accept - language" : "en-US,en;q=0.9,bg;q=0.8",
            "connection": "keep-alive",
            "cockies":"LOCALE=bg; _gcl_au=1.1.1595615601.1652778402; _gid=GA1.2.2016846164.1652778402; cookieconsent_status=dismiss; PHPSESSID=7b1525b506a2c780a568378122d8cfee; _gat_gtag_UA_201170889_1=1; _ga=GA1.2.1137365934.1652778401; _ga_ZGMJEDYHKC=GS1.1.1652778401.1.1.1652778689.0",
            "dnt": "1",
            #"host": "imoti-shumen.imot.bg",
            "referer": f"https://realistimo.com/bg/buy/trakiya-shumen-bg/?page={page}",
            "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="101", "Google Chrome";v="101"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36',

    }


    url = f'https://realistimo.com/bg/buy/trakiya-shumen-bg/?page={page}'

    #Get url request
    response= requests.get(url,headers=headers)

    #Create a soup object
    soup=BeautifulSoup(response.text,"html.parser")

    #Finding the content,we look for in the html file of the website
    ads= soup.find_all('div', class_='card-property-wrap mb-6 flex-col-sm-6 flex-col-lg-4')



    #Itarate over all ads on the given page
    for ad in ads:
        type=ad.find('span',class_='type').text.replace('\n','').strip()
        price=ad.find('span',class_='price').text.replace('\n','').strip()
        sqrm=ad.find('div',class_='secondary').text.replace('\n','')[0:4]
        total_rooms=ad.find('div',class_='secondary').text.replace('\n','')[7:8]
        thumbnail_text=ad.find('div',class_='thumbnail-text').text.replace('\n','').strip()
        details=ad.find('div',class_='secondary').text.replace('\n','')
        


        data={'App_no':str(app_no),'type': type , 'price': price,
            'sqrm':sqrm,'total_rooms':total_rooms,'thumbnail_text':thumbnail_text,
            'details':details
        }
        result.append(data)
        app_no+=1


def clear_eur(price):
    if len(price.split())>2:
        return int("".join(price.split()[0:2]))
    else:
        return int(price)


def lv_eur(price):
    if price.split()[2].lower()=='лева':
        eur = round((int("".join(price.split(" ")[0:2])))/1.952)
    else:
        return price
    return str(eur)

#Create a dataframe
df = pd.DataFrame(result)

#Select just the appartaments
df1 = df[df['type']=='Апартамент']
df2=df1.copy()
#Aplly function to convert lvs into eur
df2['price']=df2['price'].apply(lv_eur)
df3=df2.copy()

#Convert all the prices into eur integers
df3['price']=df3['price'].apply(clear_eur)
df3

# open('realistimo.json', 'w', encoding='utf-8') as f:
#         json.dump(result, f, indent=8, ensure_ascii=False)
#         print("Created Json File")


