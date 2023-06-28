FC = ftn
CXX = CC

FSRCSUFFIX = F90
FOBJSUFFIX = f
CXXOBJSUFFIX = C

F_SRCS = $(wildcard *.$(FSRCSUFFIX))
CPP_SRCS = $(wildcard *.cpp)

F_OBJS = $(patsubst %.$(FSRCSUFFIX),%_f,$(F_SRCS))
CPP_OBJS = $(patsubst %.cpp,%_C,$(CPP_SRCS))

all: $(F_OBJS) $(CPP_OBJS)

%_$(FOBJSUFFIX): %.$(FSRCSUFFIX)
	$(FC) $< -o $@

%_$(CXXOBJSUFFIX): %.cpp
	$(CXX) $< -o $@

clean:
	rm -f *_$(CXXOBJSUFFIX) *_$(FOBJSUFFIX)
