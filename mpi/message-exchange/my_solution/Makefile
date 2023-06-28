FC = ftn
CXX = CC

FSUFFIX = F90

F_SRCS = $(wildcard *.$(FSUFFIX))
CPP_SRCS = $(wildcard *.cpp)

F_OBJS = $(patsubst %.$(FSUFFIX),%_f,$(F_SRCS))
CPP_OBJS = $(patsubst %.cpp,%_C,$(CPP_SRCS))

all: $(F_OBJS) $(CPP_OBJS)

%_f: %.$(FSUFFIX)
	$(FC) $< -o $@

%_C: %.cpp
	$(CXX) $< -o $@
