'''
Obtaining Data From MMI Database.

# Libraries Used
	xml,os,csv

# Variables Used
	parser = XML Parser created to parse XML files in MMI Database.
	sessions = list of files present in Sessions folder in MMI Database.
'''

import xml.sax              #Import sax parser for XML files
import os
import csv

#Open the CSV file and create a CSV writer to write into it.
csvfile = open("emos.csv", "wb")
print "Creating a writer and writing header row..."
data = csv.writer(csvfile, delimiter = ',')
data.writerow(['Person Id']+['AU Code '+str(i) for i in range(1,65)]+['Emotion'])

#ContentHandler Class defined for handling the specific XML files using SAX parser. 
class xml_file_handler(xml.sax.ContentHandler):
	def __init__(self,p_id):
		self.AU = []
		self.id = p_id
		self.currentData = ""
	def startElement(self, tag, attributes):
		self.currentData = tag
		if tag == "ActionUnit":
			x=attributes["Number"]
			print x, self.id
			try:
				self.AU.append(int(x))
			except:
				self.AU.append(int(x[:-1]))
	def endDocument(self):
		if len(self.AU)!=0:
			lau=['' for i in range(1,65)]
			for i in self.AU:
				lau[i-1]=3
			l=[self.id]+lau
			data.writerow(l)

#Initialize the SAX Parser
parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)

#Parse through the Sessions Folder to obtain AU codes and Emotion Label.
sessions = [iterator for iterator in os.walk("Sessions")]
for i in range(1,len(sessions)):
	xml_file=sessions[i][0]+'\\'+sessions[i][2][1]	
	Handler = xml_file_handler(sessions[i][0].split('\\')[1])
	parser.setContentHandler( Handler )
   	parser.parse(xml_file)
