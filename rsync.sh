#!/bin/bash

inotifywait -m -q -r -e  modify,create,delete --format '%w%f' ~/GraphGPS | while read FILE
do
	  var=$(date)
	  echo "$var : something happened on $FILE"
	  prefix=~/GraphGPS/
	  FILE_P=${FILE#$prefix}
	  echo "rsync to /gpfswork/rech/tbr/uho58uo/graphit_MOF/$FILE_P"
     	  # rsync -azc  --exclude=".git/*" -e "ssh -i ~/.ssh/id_rsa_jeanzay" $FILE  ump88gx@jean-zay.idris.fr:/gpfswork/rech/tbr/ump88gx/GraphGPS/$FILE_P
done
