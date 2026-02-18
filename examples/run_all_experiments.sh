#!/bin/sh

for i in icare10 icare100 FS-SVM cox rsf100 rsf10 gb10 gb100
do
  for j in True False
  do
    echo Hecktor22 $i $j
    python3 Hecktor22.py $i $j
  done
done