search_dir=/gpfswork/rech/tbr/ump88gx/EJ_logs/
#ids = [695442, 695456, 695479, 695485, 695491, 695499, 695501, 695523]
for file in $search_dir/*.out
do
	#echo "$file" >> n_params.txt
	echo "--------------------------------------------------------------------------" >> n_params.txt
	cat $file | grep "  name: " >> n_params.txt
	cat $file | grep "  project: ZINC_seeds" >> n_params.txt
	cat $file | grep "Num parameters:" >> n_params.txt
done
