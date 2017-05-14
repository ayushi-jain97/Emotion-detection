import os
import csv
print "Opening FACS folder..."
facs = [iterator for iterator in os.walk("FACS")]
print "Opening Emotions folder..."
emos = [iterator for iterator in os.walk("Emotion")]
l = len(emos)
print "Creating a blank csv file emos.csv..."
csvfile = open("emos.csv", "wb")
print "Creating a writer and writing header row..."
data = csv.writer(csvfile, delimiter = ',')
data.writerow(['Person Id','Person SubID']+['AU Code '+str(i) for i in range(1,65)]+['Emotion'])
data_to_be_written=[]
for i in range(l):
	node_facs = facs[i]
	node_emos = emos[i]
	path_facs = node_facs[0].split('\\')
	path_emos = node_emos[0].split('\\')
	if(len(path_facs)==3):
		print "Data for "+path_facs[1]+" "+path_facs[2]
		pid = path_facs[1]
		psid = path_facs[2]
		writing = [pid,psid]+['']*65
		f_facs = node_facs[2]
		f_emos = node_emos[2]
		if(f_facs):
			f=open(path_facs[0]+"\\"+path_facs[1]+"\\"+path_facs[2]+"\\"+f_facs[0],'rb')
			f_data = f.readlines()
			for line in f_data:
				line_data = line.strip().split('   ')
				if line_data[0]:
					writing[int(float(line_data[0]))+1]=int(float(line_data[1]))
			f.close()
		if(f_emos):
			f=open(path_emos[0]+"\\"+path_emos[1]+"\\"+path_emos[2]+"\\"+f_emos[0],'rb')
			f_data = f.readlines()
			for line in f_data:
				line_data = line.strip()
				writing[66]=int(float(line_data))
			f.close()
		data_to_be_written.append(writing)

data.writerows(data_to_be_written)
