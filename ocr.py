import numpy as np
import cv2
from os import listdir
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import numpy as np
import time

def make_rows(contours, thresh_y = 0.7):
	contoursBBS = {}
	height_list=[]
	for contour in contours:
	    [x, y, w, h] = cv2.boundingRect(contour)
	    height_list.append(h)
	height_list.sort()
	#contours with height less than min_height will be discarded
	min_height = height_list[int(len(height_list)/2)]*0.7
	print("min_height: ",min_height)
	#finding suitable line height
	alpha = int(len(height_list)*0.3)
	line_height = 1.2*sum(height_list[alpha:len(height_list)-alpha])/(len(height_list)-2*alpha)

	for contour in contours:
	    [x, y, w, h] = cv2.boundingRect(contour)
	    if h< min_height : continue
	    cnt= [x,y,w,h]
	    search_key= y
	    #check if current contour is part of any existing row
	    if contoursBBS: 
	    	text_row = min(contoursBBS.keys(), key = lambda key: abs(key-search_key))
	    	#if diff btw nearest row and y is greater than the threshhold 
	    	#if(abs(text_row-y) > h*thresh_y):
	    	if(abs(text_row-y) > line_height):
	    		contoursBBS[y]=[]
	    		contoursBBS[y].append(cnt)
	    	else :  contoursBBS[text_row].append(cnt)
	    #else make new row
	    else: contoursBBS[y]=[cnt]
	
	#sort contours
	for row in contoursBBS:
		contoursBBS[row].sort(key = lambda x: x[0])
	
	return contoursBBS

def detect_line(rect,x1,x2,y1,y2,w1,w2,h1,h2):
	x1=x1+w1+1
	y=int((y1+h1)/2 + (y2+h2)/2)
	pos_edge=0
	neg_edge=0
	for i in range(x1,x2):
		if (int(rect[y][i][0])+int(rect[y][i][1])+int(rect[y][i][2]) - int(rect[y][i-2][0])-int(rect[y][i-2][1])-int(rect[y][i-2][2]))/2 >= 80 : pos_edge=1
		if (int(rect[y][i][0])+int(rect[y][i][1])+int(rect[y][i][2]) - int(rect[y][i-2][0])-int(rect[y][i-2][1])-int(rect[y][i-2][2]) ) /2  <= -80 : neg_edge=1
		if(pos_edge and neg_edge): 
			print("line detected between ",x1+w1," ",x2)
			return True
	return False
#a utility function to merge two words based on their nearness 
def merge_boxes(rect, contoursBBS, thresh_x = 0.3, thresh_y = 0.3):
	merge_cnt={}
	i=0
	for key in contoursBBS:
		j=1
		i=0
		de=[]
		merge_cnt[key]=[]
		[x1,y1,w1,h1]=contoursBBS[key][i]
		new_width = w1
		new_height = h1
		miny=y1
		#iterating through row to see if current contour can be merged with previous
		while j< len(contoursBBS[key]):

			[x2,y2,w2,h2]=contoursBBS[key][j]
			if( abs(y1-y2)<h1*thresh_y and abs(x1+new_width-x2) < h1*thresh_x and abs(new_height-h2)<h2*thresh_y and not(detect_line(rect,x1,x2,miny,y2, new_width,-1,new_height,h2)and detect_line(rect,x1,x2,miny,y2, new_width,-1,int(new_height/2),int(h2/2)) ) ):
				miny=min(miny,y2)
				new_width= x2-x1+w2
				new_height= max(new_height, y2+h2-miny)
				j+=1
				if j==len(contoursBBS[key]):
					merge_cnt[key].append([x1,miny,new_width,new_height])
			else:
				merge_cnt[key].append([x1,miny,new_width,new_height])
				i=j
				j+=1
				[x1,y1,w1,h1]=contoursBBS[key][i]
				new_width = w1
				new_height = h1
				miny=y1
				if j==len(contoursBBS[key]):
					merge_cnt[key].append(contoursBBS[key][j-1])
		if(len(contoursBBS[key])==1):merge_cnt[key].append(contoursBBS[key][0])
	#print("merged")
	#for i in sorted (merge_cnt) : 
	#    print ((i, merge_cnt[i]), end =" ") 

	return merge_cnt

def stretch_columns(img):
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
	img = cv2.erode(img, structure,iterations=1) 
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
	x = cv2.dilate(img, structure,iterations=2) 
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
	x = cv2.dilate(x, structure,iterations=1) 
	contours, hierarchy = cv2.findContours(x, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)

	cv2.imwrite("columnstretched.jpg",x)
	return x

def segment_columns(img,shape,contours):
	mask = np.zeros((shape[0],shape[1]),np.uint8)
	for key in contours:
		for i in contours[key]:
			[x,y,w,h]= i
			mask = cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1) 
			print(y," ",y+h," ",x," ",x+w," is 255")

	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (4,1))
	mask = cv2.erode(mask, structure,iterations=1)
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
	mask = cv2.dilate(mask, structure,iterations=2)
	cont, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)
	for c in cont:
		[x,y,w,h]= cv2.boundingRect(c)
		img = cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,255), 3) 

	cv2.imwrite("blocks.jpg",img)
	return cont
def ignore_lines(img,save_dir,file_name):

	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	ret= min(255,int(1.5*ret))
	
	#after thresholding, to connect pixels in weak images
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
	x = cv2.dilate(bw, structure,iterations=1) 
	
	y=bw

	# Applying dilation on vertical lines
	rows = x.shape[0]
	kernel = np.ones((3,3),np.uint8)
	vertical_size = rows / 80
	vertical_size=int(vertical_size)

	# Create structure element for extracting vertical lines through morphology operations
	print("defining vertical lines...",flush=True)
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vertical_size))

	# Apply morphology operations
	x = cv2.erode(x, verticalStructure)
	x = cv2.dilate(x, verticalStructure) 
	cv2.imwrite(save_dir+ 'X_' + file_name,x)

	#applying mask of vertical lines to image
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
	x = cv2.dilate(x, rect_kernel, iterations = 1)


	#applying dilation to horizontal lines
	print("defining horizontal lines...",flush=True)
	cols = y.shape[1]
	horizontal_size = cols / 90
	horizontal_size=int(horizontal_size)

	# Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

	# Apply morphology operations
	y = cv2.erode(y, horizontalStructure)
	y = cv2.dilate(y, horizontalStructure) 

	# Applying dilation on horizontal lines
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
	y = cv2.dilate(y, rect_kernel, iterations = 1)

	cv2.imwrite(save_dir+'Y_' + file_name,y)

	#applying vertical and horizontal line masks
	ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU ) 
	bw = bw + x + y
	
	cv2.imwrite(save_dir+'lines_removed_'+file_name,bw)

	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
	
	# Appplying dilation on the threshold image 
	ret, thresh1 = cv2.threshold(bw, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
	cv2.imwrite(save_dir+'dilated_'+file_name, dilation)
	# Finding contours 
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)
	column_dilation = stretch_columns(dilation)
	print("preprocessing done.",flush=True)
	return contours,hierarchy, bw




keys = ['P.O.#','Sales Rep. Name','Ship Date','Ship Via','Terma','Due Date','# / Taxable','Description','Quantity','Unit Price','Line Total']


def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) 
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return distance[row][col]



def get_text(save_dir,file_name, write_ = False):
	read_dir = save_dir + file_name
	# Mention the installed location of Tesseract-OCR in your system 
	#pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
	print(read_dir)
	# Read image from which text needs to be extracted 
	img = cv2.imread(read_dir) 
	rect= img
	img2 = img
	#remove lines and form contours
	contours, hierarchy, img = ignore_lines(img,save_dir,file_name)
	# Creating a copy of image 
	#rect = img2.copy() 
	# A text file is created and flushed 
	file = open("recognized.txt", "a") 
	file_ = open(save_dir+"recognized.txt","a")
	file.write("")
	file_.write("")

	#assigning rows to contours
	contoursBBS = make_rows(contours)

	#combining contours on the bases of contour thresh x
	merge_cnt = merge_boxes(rect, contoursBBS, thresh_x = 1.0, thresh_y = 0.6)
	column_contours = segment_columns(img2,img.shape,merge_cnt)

	print("recognizing text",flush=True)

	# Looping through the identified contours 
	# Then rectangular part is cropped and passed on 
	# to pytesseract for extracting text from it 
	# Extracted text is then written into the text file 
	# Open the file in append mode 
	croptime=0
	tesstime=0
	tt=time.time()
	key_nodes=[]
	text_val={}
	node_number=0
	node_columns={}

	for cnt in sorted(merge_cnt): 
		#print(cnt)
		#cv2.line(rect,(0,cnt),(len(rect[0])-1,cnt),(255,0,0),2)
		for contour in merge_cnt[cnt] :
			node_number+=1
			[x, y, w, h] = contour
			if h<10 : continue
			# Drawing a rectangle on copied image 
			rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 1) 
			
			# Cropping the text block for giving input to OCR 
			#offset= int (h*cnt_thresh_y)
			#y=y-offset
			#x=x-offset
			start = time.time()
			cropped = img[max(0,y-2) :y + h + 2 , max(0,x - 2)  : x + w + 2]  
			end = time.time()
			croptime+= end-start
			# Apply OCR on the cropped image 

			text = ""
			# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
			text = pytesseract.image_to_string(cropped, lang='eng', config='--psm 6')
			p=text.split(' ')
			for tex in p:
				tex=tex.lower()
				if(tex in keys):
					rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 0, 255), 1)
					key_nodes.append(node_number-1)
					break
				else:
					for k in keys:
						#print(k + " " + tex)
						if(k in tex):
							rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 0, 255), 1)
							key_nodes.append(node_number-1)
							break
						if(len(text)>=1 and levenshtein_ratio_and_distance(k,tex)>0.8 ): 			
							rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 0, 255), 1)
							key_nodes.append(node_number-1)
							# if(':' in text):
							# 	key, val = text.split(':',1)
							# 	break
							break
			text_val[node_number-1] = text	
			end2 = time.time()
			tesstime += end2-end
			# Appending the text into file 
			if text!="":
				if write_: file.write(text)
				file_.write(text)
				if write_: file_.write("\n") 
				file.write("\n") 

	tt = time.time()- tt
	print("croptime: ",croptime, "  tesstime: ",tesstime,"   tt: ",tt)
	# Close the file 
	file.close 
	cv2.imwrite(save_dir+'boxed_'+file_name, rect)


def main():
	
    dir_path = 'D:/Qureai/Problem_3/ocr_main/'
    file_name = '1.png'
    get_text(dir_path,file_name,write_ = False)
			
if __name__ == '__main__':
	main()