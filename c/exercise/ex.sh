#!/bin/bash
#set -xv
exec 1>mylog.txt 2>&1
size=(128000 2621440 6553600 10485760)
kernel=(4 8 16 32)
#i=0
#times=1
for k in ${kernel[@]};
do
	echo "**********************************************************************************"
	echo  "current kernel is $k"
	i=0
	times=49
	while [ $i -le $times ]
	do
		let 'i++'
		echo "###### the results are shown as follow ######"
		echo "current time is $i"
		for s in ${size[@]};
		do
    			./homework $s $k
			echo "current size is $s"
			
        
		done
	done
done
