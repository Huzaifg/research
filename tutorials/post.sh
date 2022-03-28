#!/bin/bash

python3 diag_plot.py "$1" "$2" > dump.txt

#Print the first 2 lines as it is
head -2 dump.txt


#Remove those 2 lines from the file
# sed -e '1,2d' < dump.txt

	



while read -r line; 
do
	# If there is a line with float 64, remove that float 64
	if [[ "$line" == *"float64"* ]]; 
		then var1=`echo "$line" | rev |  cut -d' ' -f3- | rev | sed 's/ //g'`; var2=`echo "$line"| rev | cut -d' ' -f1 | rev` ; echo "$var1" "$var2";
	#If there is a line with xarray.Dataset remove that xarray.dataset
	elif [[ "$line" == *"xarray.Dataset"* ]];
		then var3=`echo "$line" | cut -d'<' -f1` ; 
		echo "$var3";
	#For the data array ones
	elif [[ "$line" == *"xarray.DataArray"* ]];
		then var4=`echo "$line" | cut -d'<' -f1` ;
		echo "$var4";
	#For all the numbers stuck in the array()
	elif [[ "$line" == *"array("* ]];
		then var5=`echo "$line" | cut -d'(' -f2 | cut -d')' -f1` ;
		echo "$var5";
	#Print all the divergences statements as is
	elif [[ "$line" == *"divergences"* ]];
		then echo "$line";

fi; 
done < dump.txt 


#DElete the dump file
rm dump.txt
