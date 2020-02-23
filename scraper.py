# Written by Abrahan Nevarez
# Scrapes an image and uploads image and details to a PostgreSQL database
from ui import mainWindow
from PySide2 import QtWidgets, QtCore, QtGui
import json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urlparse
import urllib
from selenium.webdriver.chrome.options import Options 
import os, shutil
class mainWindow(mainWindow.Ui_MainWindow, QtWidgets.QMainWindow):
     
    def __init__(self):
        super(mainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.seleniumLaunch)
    
    # Parses the data from the page
    def grabData(self, windowAddress):
        link = urlparse(windowAddress).path
        filteredoutlink = re.findall(r'(dp/\S+)', link)
        # initialize an empty string 
        str1 = " " 
    
        # return string   
        newLink = str1.join(filteredoutlink)
        newLink = newLink.replace("dp/", "")
        reviews = []
        asin = newLink.replace("/ref=sr_1_", "")
        asin = asin[:-1]
        html = urllib.request.urlopen(windowAddress).read()
        soup = BeautifulSoup(html, 'html.parser')
        html = soup.prettify('utf-8')
        product_json = {}
        # This block of code will help extract the Prodcut Title of the item
        for spans in soup.findAll('span', attrs={'id': 'productTitle'}):
            name_of_product = spans.text.strip()
            product_json['name'] = name_of_product
            break

        product_json['catagories'] = []
        for span in soup.findAll('span', class_ = 'nav-a-content'):
            if span.text:
                catagories = span.text.strip()
                product_json['catagories'] = catagories
                print(catagories)
                break
        
        # This block of code will help extract the image of the item in dollars
        # This block of code will help extract the average star rating of the product
        for i_tags in soup.findAll('i',
                                attrs={'data-hook': 'average-star-rating'}):
            for spans in i_tags.findAll('span', attrs={'class': 'a-icon-alt'}):
                product_json['star-rating'] = spans.text.strip()
                break
        # This block of code will help extract the number of customer reviews of the product
        for spans in soup.findAll('span', attrs={'id': 'acrCustomerReviewText'
                                }):
            if spans.text:
                review_count = spans.text.strip()
                product_json['customer-reviews-count'] = review_count
                break
        # # This block of code will help extract top specifications and details of the product
        # product_json['details'] = []
        # for ul_tags in soup.findAll('ul',
        #                             attrs={'class': 'a-unordered-list a-vertical a-spacing-none'
        #                             }):
        #     for li_tags in ul_tags.findAll('li'):
        #         for spans in li_tags.findAll('span',
        #                 attrs={'class': 'a-list-item'}, text=True,
        #                 recursive=False):
        #             product_json['details'].append(spans.text.strip())

        # This block of code will help extract the short reviews of the product

        # product_json['short-reviews'] = []
        # for a_tags in soup.findAll('a',
        #                         attrs={'class': 'a-size-base a-link-normal review-title a-color-base a-text-bold'
        #                         }):
        #     short_review = a_tags.text.strip()
        #     product_json['short-reviews'].append(short_review)
        # This block of code will help extract the long reviews of the product
        product_json['long-reviews'] = []
        for divs in soup.findAll('div', attrs={'data-hook': 'review-collapsed'
                                }):
            long_review = divs.text.strip()
            product_json['long-reviews'].append(long_review)
        
        product_json['asin'] = []
        product_json['asin'].append(asin)
        result = ""
        nameOfFile =  result.join(asin)
        nameOfFile = nameOfFile.replace("[]", "")
        # print(nameOfFile)
        # Saving the scraped data in json format
        # Checks for json animal path
        jsonDir = "jsonData"
        jsonDirFile = "jsonData/" + product_json["catagories"]

    
        # Check parent folder to prevent creation of improper directories
        if not os.path.exists(jsonDir):
            os.makedirs(jsonDir)
        if not os.path.exists(jsonDirFile):
            os.makedirs(jsonDirFile)

        # Opens up the JSON file
        file = open(nameOfFile + ".json", "w")

        # Dumps the data to the file
        json.dump(product_json, file, indent = 1)
        file.close()

        

        # Move to local directory for json
        shutil.move(file.name, jsonDirFile)
        print ('----------Extraction of data is complete. Check json file.----------')
    
    def seleniumLaunch(self, index):
        # Grabs the animal text from user input and number of pics we'll use
        animalText = self.animalSearch.text()
        product_json = {}
        # Engages the web driver
        driver = webdriver.Chrome()
        window_original = driver.window_handles[0]

        # Begin the web driver by starting on google
        driver.get("https://amazon.com")

        # # Searches for animal that is specified by user
        search = driver.find_element_by_id('twotabsearchtextbox')

        # # Adds a # to get direct animal image results
        search.send_keys(animalText)
        search.send_keys(Keys.RETURN)
        try:
            xpath = self.xpath.text()
            elems = driver.find_element_by_xpath(xpath)
            elems.click()
        except:
            driver.close()
            print("Seems we've hit something captain! And it isn't items...")
        
        # Grabs the current URL
        windowAddress = driver.current_url
        result = requests.get(windowAddress)
        src = result.content
    
        # Instantate beautifulsoup object
        data = BeautifulSoup(src, 'lxml')

        self.grabData(windowAddress)
        
        
        driver.close()
        
if __name__ == "__main__":
    app = QtWidgets.QApplication()
    qt_app = mainWindow()
    qt_app.show()
    app.exec_()