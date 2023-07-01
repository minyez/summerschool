CC = cc
CXX = CC
FC = ftn

CSRCSUFFIX = c
CXXSRCSUFFIX = cpp
FSRCSUFFIX = f90

COBJSUFFIX = c
CXXOBJSUFFIX = C
FOBJSUFFIX = f

C_SRCS = $(wildcard *.$(CSRCSUFFIX))
CPP_SRCS = $(wildcard *.$(CXXSRCSUFFIX))
F_SRCS = $(wildcard *.$(FSRCSUFFIX))

C_OBJS = $(patsubst %.$(CSRCSUFFIX),%_$(COBJSUFFIX),$(C_SRCS))
CPP_OBJS = $(patsubst %.$(CXXSRCSUFFIX),%_$(CXXOBJSUFFIX),$(CPP_SRCS))
F_OBJS = $(patsubst %.$(FSRCSUFFIX),%_$(FOBJSUFFIX),$(F_SRCS))

all: $(F_OBJS) $(CPP_OBJS) $(C_OBJS)

%_$(FOBJSUFFIX): %.$(FSRCSUFFIX)
	$(FC) $< -o $@

%_$(COBJSUFFIX): %.$(CSRCSUFFIX)
	$(CC) $< -o $@

%_$(CXXOBJSUFFIX): %.$(CXXSRCSUFFIX)
	$(CXX) $< -o $@

clean:
	rm -f *_$(CXXOBJSUFFIX) *_$(FOBJSUFFIX) *_$(COBJSUFFIX)
