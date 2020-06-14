#!/bin/bash
gunzip -c aclImdb_v1.tar.gz | tar xopf -
cd aclImdb
mkdir movie_data
for split in train test;
do
  for sentiment in pos neg;
  do   
    for file in $split/$sentiment/*; 
    do
              cat $file >> movie_data/full_${split}.txt; 
              echo >> movie_data/full_${split}.txt; 
	     # Esta línea agrega archivos que contienen las revisiones originales si lo desea
             # echo $file | cut -d '_' -f 2 | cut -d "." -f 1 >> combined_files/original_${split}_ratings.txt; 
    done;
  done;
done;