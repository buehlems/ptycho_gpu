

OBJS=main.o ptycho_petsc.o

.SUFFIXES: .o .c

PROG=petsc_driver.x

# PETSC_ROOT=/opt/petsc
PETSC_ROOT=/home/gpu28/opt/petsc

CFLAGS="-I$(PETSC_ROOT)/include"

.c.o:
	mpicc -c $(CFLAGS) $<


all: $(PROG)

$(PROG): $(OBJS) 
	mpicc -o $(PROG) $(OBJS) -L$(PETSC_ROOT)/lib -lpetsc -Wl,-rpath,$(PETSC_ROOT)/lib

main.o: ptycho_petsc.h

clean:
	rm -f $(PROG) $(OBJS) main.o
	
