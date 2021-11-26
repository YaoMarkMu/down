BONUS=0
CC=gcc
CFLAGS= -pthread -Wall -lm -lnuma -O3

ifeq ($(BONUS), 1)
	CFLAGS += -DBUILD_BONUS=1
endif 

TARGET=homework
ALL: clean $(TARGET)

$(TARGET): %: %.c
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -rf *.o $(TARGET) *.s
