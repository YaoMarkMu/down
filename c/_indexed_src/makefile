#DEBUG=0
CC=gcc
CFLAGS= -pthread -Wall -lm 

ifeq ($(DEBUG), 1)
	CFLAGS += -g 
else
	CFLAGS += -O3
endif 

TARGET=p13_show_stack p14_show_tid p26_hello p30_detach p32_return_stack_ptr p38_kway_merge_sort p49_bind_affinity p51_vec_sum p54_shared_data p66_shared_data_mutex p70_bank p74_deadlock
ALL: $(TARGET)

$(TARGET): %: %.c
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -rf *.o $(TARGET) *.s
